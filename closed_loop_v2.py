"""Замкнутый контур v2: коннектом дрозофилы (FlyWire) <-> тело (NeuroMechFly/MuJoCo).

Отличия от MVP от 01.08.2026, который был разобран и признан несостоявшимся:

  1. Мозг — валидированная TorchModel из fly-brain/code/run_pytorch.py, а не
     самописная копия с потерянными множителями wScale и time_factor_mem.
  2. Вход от тела идёт в ПРЯМЫХ возбуждающих пресинаптических партнёров P9,
     взятых из коннектома. Раньше он шёл в Sugar GRNs, откуда до P9 нет пути:
     измерение дало ровно 0 спайков P9.
  3. Обратная связь — 6 настоящих флагов касания лапок из штатного сенсора
     MuJoCo (sim.get_ground_contact_info), а не общее число контактов сцены.
  4. Выход мозга реально управляет телом через HybridTurningController, который
     принимает пару нисходящих сигналов (левый/правый) и модулирует амплитуды
     CPG по сторонам. Раньше cmd_signal вычислялся и никуда не шёл.
  5. Нормировка взята из измеренной тюнинг-кривой P9, а не с потолка.

Схема одного внешнего цикла (15 мс):
    тело -> 6 флагов касания -> частота Пуассона на драйверы P9 (лево/право)
         -> мозг 150 шагов по 0.1 мс -> частота спайков P9 L/R
         -> нормировка -> descending_signal[2] -> CPG -> 150 шагов физики
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

PROJECT = "/mnt/d/временное использование федей/мозг мухи"
sys.path.insert(0, f"{PROJECT}/fly-brain/code")

os.environ.setdefault("MUJOCO_GL", "egl")

from benchmark import path_comp, path_con, path_wt  # noqa: E402
import run_pytorch as rp  # noqa: E402

from flygym.compose import FlatGroundWorld  # noqa: E402
from flygym.simulation import Simulation  # noqa: E402
from flygym.utils.math import Rotation3D  # noqa: E402
from flygym_demo.complex_terrain import (  # noqa: E402
    HybridControllerObservation,
    HybridTurningController,
    PreprogrammedSteps,
    apply_locomotion_action,
    make_locomotion_fly,
)

# ---------------------------------------------------------------------------
# Параметры связки. Все взяты из измерений, источник указан.
# ---------------------------------------------------------------------------

P9_LEFT = 720575940627652358
P9_RIGHT = 720575940635872101

DT_BRAIN_MS = 0.1          # шаг сети, как в статье Shiu et al.
SYNC_MS = 15.0             # окно синхронизации мозг<->тело
N_DRIVERS = 20             # сколько сильнейших возбуждающих входов P9 стимулируем

# Из tools/p9_tuning_curve.py: при входе 200 Гц на драйверы P9 даёт ~175 Гц,
# при 0 Гц — ровно 0 Гц. 180 Гц берём за верх рабочего диапазона.
P9_REF_HZ = 180.0

# Постоянная времени сглаживания командного сигнала.
# Зачем: за окно 15 мс P9 выдаёт всего 1-3 спайка, поэтому мгновенная оценка
# частоты квантуется шагом 1/0.015 = 66.7 Гц — команда получает всего три
# уровня (0.37, 0.74, 1.0). Это измерено в первом прогоне closed_loop_v2.
# Экспоненциальное сглаживание по нескольким окнам восстанавливает разрешение
# и заодно отражает реальную низкочастотную фильтрацию нисходящих команд
# синапсами и мышцами.
TAU_CMD_MS = 100.0

# Диапазон обратной связи. Нижняя граница ненулевая, иначе при полном отрыве
# лапок мозг замолкает намертво и контур больше не может сам себя запустить.
FB_BASE_HZ = 50.0
FB_SPAN_HZ = 150.0

# Доля касаний в разомкнутом контроле. Измерено в прогонах: при ходьбе на земле
# в среднем 4.1 лапки из 6, то есть примерно 0.68 на сторону. Держим контроль в
# той же рабочей точке, чтобы разница между прогонами объяснялась наличием
# обратной связи, а не сдвигом средней частоты входа.
OPEN_LOOP_DUTY = 0.68


def build_brain(device):
    """Загрузка коннектома, поиск драйверов P9, сборка валидированной модели."""
    print("[мозг] читаю список нейронов...")
    df_comp = pd.read_csv(path_comp, index_col=0)
    flyid2i = {j: i for i, j in enumerate(df_comp.index)}
    n = len(flyid2i)
    idx_l, idx_r = flyid2i[P9_LEFT], flyid2i[P9_RIGHT]
    print(f"[мозг] нейронов: {n}, P9 left={idx_l}, P9 right={idx_r}")

    print("[мозг] ищу прямых возбуждающих партнёров P9 по коннектому...")
    conn = pd.read_parquet(path_con, columns=[
        "Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity",
    ])
    W = sp.coo_matrix(
        (conn["Excitatory x Connectivity"].to_numpy().astype(np.float32),
         (conn["Postsynaptic_Index"].to_numpy(), conn["Presynaptic_Index"].to_numpy())),
        shape=(n, n),
    ).tocsr()
    del conn

    def top_drivers(idx):
        row = W.getrow(idx).tocoo()
        pos = row.data > 0
        cols, vals = row.col[pos], row.data[pos]
        order = np.argsort(-vals)[:N_DRIVERS]
        return cols[order].tolist(), vals[order].tolist()

    drv_l, w_l = top_drivers(idx_l)
    drv_r, w_r = top_drivers(idx_r)
    del W
    print(f"[мозг] драйверы P9 left:  {len(drv_l)} шт, веса {w_l[0]:.0f}..{w_l[-1]:.0f}")
    print(f"[мозг] драйверы P9 right: {len(drv_r)} шт, веса {w_r[0]:.0f}..{w_r[-1]:.0f}")

    print("[мозг] загружаю матрицу весов...")
    weights = rp.get_weights(str(path_con), str(path_comp), str(path_wt), csr=True).to(device)

    stim_all = sorted(set(drv_l) | set(drv_r))
    model = rp.TorchModel(
        1, n, DT_BRAIN_MS, rp.MODEL_PARAMS, weights,
        exc_indices=stim_all, device=device,
    )
    return model, n, idx_l, idx_r, drv_l, drv_r


def build_body(with_camera=False):
    """Тело: NeuroMechFly, ноги с позиционными приводами, плоский грунт."""
    print("[тело] собираю модель...")
    fly = make_locomotion_fly()
    # Камеру надо добавить ДО компиляции модели. set_renderer принимает сам
    # элемент MJCF, а не строку: при подключении мухи к миру имя камеры
    # получает префикс пространства имён, и поиск по строке "trackcam" падает.
    cam = fly.add_tracking_camera(name="trackcam") if with_camera else None
    world = FlatGroundWorld()
    world.add_fly(
        fly,
        spawn_position=[0.0, 0.0, 0.5],
        spawn_rotation=Rotation3D("quat", [1, 0, 0, 0]),
        add_ground_contact_sensors=True,
    )
    sim = Simulation(world)
    sim.reset()
    print(f"[тело] шаг физики: {sim.timestep * 1000:.4f} мс")
    print(f"[тело] порядок лапок: {fly.get_legs_order()}")
    return sim, fly, cam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=100,
                    help="число внешних циклов по 15 мс")
    ap.add_argument("--open-loop", action="store_true",
                    help="контроль: обратная связь отключена, вход постоянный")
    ap.add_argument("--tag", type=str, default="closed",
                    help="метка прогона для имён выходных файлов")
    ap.add_argument("--video", action="store_true",
                    help="писать mp4 со следящей камерой")
    ap.add_argument("--tau", type=float, default=TAU_CMD_MS,
                    help="постоянная сглаживания команды, мс (0 — без сглаживания)")
    ap.add_argument("--fb-base", type=float, default=FB_BASE_HZ,
                    help="частота входа при нуле касаний, Гц")
    ap.add_argument("--fb-span", type=float, default=FB_SPAN_HZ,
                    help="прибавка при всех касаниях, Гц")
    ap.add_argument("--seed", type=int, default=0, help="seed пуассоновского входа")
    ap.add_argument("--perturb", action="store_true",
                    help="сенсорное возмущение: в средней трети прогона вход от "
                         "тела принудительно занижен до fb_base, как будто лапки "
                         "потеряли опору. Проверка причинности мозг->тело.")
    ap.add_argument("--perturb-side", choices=("both", "left", "right"), default="both",
                    help="какую сторону глушить при возмущении. Одностороннее "
                         "возмущение проверяет, даёт ли асимметрия P9 поворот.")
    args = ap.parse_args()
    fb_base, fb_span = args.fb_base, args.fb_span

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 78)
    print(f" ЗАМКНУТЫЙ КОНТУР v2  ({'РАЗОМКНУТЫЙ КОНТРОЛЬ' if args.open_loop else 'замкнутый'})")
    print("=" * 78)
    print(f"device={device}, циклов={args.cycles}, окно={SYNC_MS} мс")
    print(f"обратная связь: {fb_base:.0f}..{fb_base + fb_span:.0f} Гц, seed={args.seed}")

    model, n_neurons, idx_l, idx_r, drv_l, drv_r = build_brain(device)
    cond, delay_buf, spikes, v, refrac = model.state_init()

    sim, fly, cam = build_body(with_camera=args.video)
    if args.video:
        sim.set_renderer(cam, camera_res=(360, 480),
                         playback_speed=0.2, output_fps=25)
        print("[тело] запись видео включена")
    legs = fly.get_legs_order()
    n_left = sum(1 for leg in legs if str(leg).startswith("l"))
    print(f"[тело] левых лапок: {n_left} из {len(legs)}")

    controller = HybridTurningController(
        timestep=sim.timestep,
        preprogrammed_steps=PreprogrammedSteps(),
    )
    controller.reset(seed=0)

    print("[тело] прогрев 50 мс...")
    sim.warmup(0.05)

    inner_brain = int(SYNC_MS / DT_BRAIN_MS)
    inner_phys = int(round(SYNC_MS / 1000.0 / sim.timestep))
    print(f"[цикл] шагов мозга={inner_brain}, шагов физики={inner_phys}")

    rates = torch.zeros(1, n_neurons, device=device)
    p9_idx = torch.tensor([idx_l, idx_r], dtype=torch.long, device=device)
    p9_acc = torch.zeros(2, device=device)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)

    # Экспоненциальное сглаживание частоты P9. alpha=1 означает отсутствие
    # сглаживания (мгновенная оценка по одному окну).
    alpha = 1.0 if args.tau <= 0 else 1.0 - float(np.exp(-SYNC_MS / args.tau))
    ema_l = ema_r = None
    print(f"[цикл] сглаживание команды: tau={args.tau:.0f} мс, alpha={alpha:.3f}")

    rows = []
    headers = [
        "cycle", "t_sec",
        "contact_l0", "contact_l1", "contact_l2",
        "contact_r0", "contact_r1", "contact_r2",
        "n_contacts", "fb_rate_left_hz", "fb_rate_right_hz",
        "p9_spikes_left", "p9_spikes_right", "p9_left_hz", "p9_right_hz",
        "p9_left_hz_ema", "p9_right_hz_ema", "cmd_left", "cmd_right",
        "thorax_x_mm", "thorax_y_mm", "thorax_z_mm",
        "cpg_phase_0", "cpg_phase_3", "perturbed", "heading_deg",
    ]

    body_order = fly.get_bodysegs_order()
    bodyseg_cls = type(fly).BODY_SEGMENT_CLASS
    thorax_idx = body_order.index(bodyseg_cls("c_thorax"))
    thorax_body_id = sim._internal_bodyids_by_fly[fly.name][thorax_idx]

    t0 = time.perf_counter()
    with torch.no_grad():
        for cycle in range(args.cycles):
            # --- 1. тело -> мозг: 6 флагов касания лапок ---
            contact_found, *_ = sim.get_ground_contact_info(fly.name)
            flags = (np.asarray(contact_found) > 0).astype(int)
            left_flags, right_flags = flags[:n_left], flags[n_left:]

            if args.open_loop:
                # контроль: вход не зависит от тела, но держим ту же среднюю
                # долю касаний, что наблюдается при ходьбе, иначе сравнение
                # пойдёт по разной рабочей точке, а не по наличию связи
                fb_l = fb_r = fb_base + fb_span * OPEN_LOOP_DUTY
            else:
                fb_l = fb_base + fb_span * left_flags.sum() / max(len(left_flags), 1)
                fb_r = fb_base + fb_span * right_flags.sum() / max(len(right_flags), 1)

            in_perturb = args.perturb and (args.cycles // 3 <= cycle < 2 * args.cycles // 3)
            if in_perturb:
                if args.perturb_side in ("both", "left"):
                    fb_l = fb_base
                if args.perturb_side in ("both", "right"):
                    fb_r = fb_base

            rates.zero_()
            rates[:, drv_l] = fb_l
            rates[:, drv_r] = fb_r

            # --- 2. мозг: 150 шагов по 0.1 мс ---
            # Счётчик копится на GPU: снимать .item() на каждом шаге — это
            # синхронизация с устройством 300 раз за цикл, впустую.
            p9_acc.zero_()
            for _ in range(inner_brain):
                cond, delay_buf, spikes, v, refrac = model(
                    rates, cond, delay_buf, spikes, v, refrac, generator=gen
                )
                p9_acc += spikes[0, p9_idx]
            sp_l, sp_r = (int(x) for x in p9_acc.tolist())

            t_win = SYNC_MS / 1000.0
            hz_l, hz_r = sp_l / t_win, sp_r / t_win

            # --- 3. сглаживание и нормировка по измеренной тюнинг-кривой ---
            if ema_l is None:
                ema_l, ema_r = hz_l, hz_r
            else:
                ema_l += alpha * (hz_l - ema_l)
                ema_r += alpha * (hz_r - ema_r)

            cmd_l = float(np.clip(ema_l / P9_REF_HZ, 0.0, 1.0))
            cmd_r = float(np.clip(ema_r / P9_REF_HZ, 0.0, 1.0))
            descending = np.array([cmd_l, cmd_r], dtype=float)

            # --- 4. мозг -> тело: 150 шагов физики под управлением CPG ---
            for _ in range(inner_phys):
                obs = HybridControllerObservation.from_sim(sim, fly.name)
                action = controller.step(descending, obs)
                apply_locomotion_action(sim, fly.name, action)
                sim.step()
                if args.video:
                    sim.render_as_needed()

            pos = sim.get_body_positions(fly.name)[thorax_idx]
            phases = controller.cpg_network.curr_phases

            rows.append([
                cycle, round(cycle * SYNC_MS / 1000.0, 4),
                *left_flags.tolist(), *right_flags.tolist(),
                int(flags.sum()), round(fb_l, 1), round(fb_r, 1),
                sp_l, sp_r, round(hz_l, 1), round(hz_r, 1),
                round(ema_l, 2), round(ema_r, 2),
                round(cmd_l, 4), round(cmd_r, 4),
                round(float(pos[0]), 4), round(float(pos[1]), 4), round(float(pos[2]), 4),
                round(float(phases[0] % (2 * np.pi)), 4),
                round(float(phases[n_left] % (2 * np.pi)), 4),
                int(in_perturb),
                round(float(np.degrees(np.arctan2(
                    sim.mj_data.xmat[thorax_body_id].reshape(3, 3)[1, 0],
                    sim.mj_data.xmat[thorax_body_id].reshape(3, 3)[0, 0]))), 2),
            ])

            if cycle % 10 == 0 or cycle == args.cycles - 1:
                print(f"  [{cycle:3d}/{args.cycles}] касаний={int(flags.sum())} "
                      f"вход={fb_l:.0f}/{fb_r:.0f}Гц  P9={hz_l:.0f}/{hz_r:.0f}Гц  "
                      f"cmd={cmd_l:.3f}/{cmd_r:.3f}  x={pos[0]:.3f} y={pos[1]:.3f} z={pos[2]:.3f}")

    elapsed = time.perf_counter() - t0
    print(f"\nготово за {elapsed:.1f} с ({elapsed / args.cycles:.2f} с на цикл)")

    out_csv = f"{PROJECT}/output/closed_loop_v2_{args.tag}.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    print(f"лог: {out_csv}")

    if args.video:
        video_path = f"{PROJECT}/output/closed_loop_v2_{args.tag}.mp4"
        sim.renderer.save_video(video_path)
        print(f"видео: {video_path}")

    # --- сводка по критериям приёмки ---
    df = pd.DataFrame(rows, columns=headers)
    print("\n" + "=" * 78)
    print(" КРИТЕРИИ ПРИЁМКИ")
    print("=" * 78)
    dx = df["thorax_x_mm"].iloc[-1] - df["thorax_x_mm"].iloc[0]
    dy = df["thorax_y_mm"].iloc[-1] - df["thorax_y_mm"].iloc[0]
    dist = float(np.hypot(dx, dy))
    print(f"  смещение груди:      dx={dx:+.3f} мм, dy={dy:+.3f} мм, путь={dist:.3f} мм")
    print(f"  cmd_left:  среднее={df['cmd_left'].mean():.4f}  ст.откл={df['cmd_left'].std():.4f}"
          f"  диапазон {df['cmd_left'].min():.3f}..{df['cmd_left'].max():.3f}")
    print(f"  cmd_right: среднее={df['cmd_right'].mean():.4f}  ст.откл={df['cmd_right'].std():.4f}"
          f"  диапазон {df['cmd_right'].min():.3f}..{df['cmd_right'].max():.3f}")
    print(f"  насыщение cmd в 1.0: {int((df['cmd_left'] >= 0.999).sum())} из {len(df)} циклов")
    print(f"  различных уровней cmd_left: {df['cmd_left'].nunique()} из {len(df)} циклов")
    print(f"  касаний за цикл:     среднее={df['n_contacts'].mean():.2f}, "
          f"уникальных значений={df['n_contacts'].nunique()}")
    if df["n_contacts"].nunique() > 1 and df["cmd_left"].std() > 0:
        # Мозг отвечает не мгновенно, поэтому смотрим и сдвинутую корреляцию:
        # касания в цикле t против команды в цикле t+lag.
        for lag in (0, 1, 2, 3):
            c = df["n_contacts"].corr(df["cmd_left"].shift(-lag))
            print(f"  корреляция касаний с cmd_left (сдвиг {lag}): {c:+.3f}")

    if args.perturb:
        print("\n  --- ступенчатое сенсорное возмущение ---")
        df["speed"] = np.hypot(df["thorax_x_mm"].diff(), df["thorax_y_mm"].diff()) / (SYNC_MS / 1000.0)
        thirds = {
            "до":     df[df["cycle"] < args.cycles // 3],
            "во время": df[df["perturbed"] == 1],
            "после":  df[df["cycle"] >= 2 * args.cycles // 3],
        }
        print(f"  глушим сторону: {args.perturb_side}")
        print(f"  {'фаза':>10s} {'циклов':>8s} {'cmd_L':>8s} {'cmd_R':>8s} "
              f"{'скорость, мм/с':>16s} {'поворот, град':>15s}")
        for name, part in thirds.items():
            if len(part) < 2:
                continue
            turn = part["heading_deg"].iloc[-1] - part["heading_deg"].iloc[0]
            turn = (turn + 180) % 360 - 180
            print(f"  {name:>10s} {len(part):>8d} {part['cmd_left'].mean():>8.3f} "
                  f"{part['cmd_right'].mean():>8.3f} {part['speed'].mean():>16.2f} "
                  f"{turn:>+15.1f}")


if __name__ == "__main__":
    main()
