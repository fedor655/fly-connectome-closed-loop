"""Зрительный замкнутый контур: глаза мухи -> LC9 -> DNp09 -> походка.

Почему именно так. Разбор входов DNp09 по аннотациям FlyWire показал, что
единственный крупный источник чистого возбуждения для него — зрительные
проекционные нейроны (+880 при 22.8% массы веса), а вход от ног через
восходящие даёт 1.8% и в сумме тормозит. Прямая проверка подтвердила:
стимуляция 1736 восходящих и 2656 механосенсорных нейронов не даёт DNp09
ни одного спайка, а LC9 зажигают его до 159 Гц, причём строго ипсилатерально —
левые LC9 действуют только на левый DNp09, перекрёстных наводок ноль.

Отсюда схема одного цикла 15 мс:

    левый глаз  -> темнота в поле зрения -> Пуассон на LC9 слева  (87 нейронов)
    правый глаз -> темнота в поле зрения -> Пуассон на LC9 справа (92 нейрона)
                        |
                мозг, 150 шагов по 0.1 мс
                        |
            частота DNp09 left / right -> сглаживание -> команда [0..1]^2
                        |
            HybridTurningController -> CPG -> 150 шагов физики

Честная оговорка. Внутриглазная ретинотопия НЕ реализована: весь глаз сводится
к одному числу, и все LC9 своей стороны получают одинаковую стимуляцию.
Настоящие LC-нейроны смотрят каждый в свой участок поля зрения. Здесь
используется только латеральное разделение, которое измерено и подтверждено.

Какое поведение из этого выйдет — вопрос к измерению, а не к замыслу: мы не
закладывали ни избегание, ни приближение, а просто соединили измеренный путь.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flypaths import ANNOTATIONS, add_fly_brain_to_path, out  # noqa: E402

add_fly_brain_to_path()
os.environ.setdefault("MUJOCO_GL", "egl")

from benchmark import path_comp, path_con, path_wt  # noqa: E402
import run_pytorch as rp  # noqa: E402

from flygym.compose import FlatGroundWorld  # noqa: E402
from flygym.simulation import Simulation  # noqa: E402
from flygym.utils.math import Rotation3D  # noqa: E402
from flygym.utils.mjcf import GEOM_TYPES  # noqa: E402
from flygym_demo.complex_terrain import (  # noqa: E402
    HybridControllerObservation,
    HybridTurningController,
    PreprogrammedSteps,
    apply_locomotion_action,
    make_locomotion_fly,
)

# Стороны по аннотациям FlyWire; в benchmark.py подписи перепутаны.
DNP09_LEFT = 720575940635872101
DNP09_RIGHT = 720575940627652358

DT_BRAIN_MS = 0.1
SYNC_MS = 15.0
TAU_CMD_MS = 100.0     # обосновано измерением d-prime, см. tools/p9_noise_floor.py

# Диапазон стимуляции LC9. Из tools/visual_to_dn.py: 87 левых LC9 на 50 Гц дают
# DNp09 32 Гц, на 200 Гц — 109 Гц. Работаем в этом диапазоне.
LC_BASE_HZ = 40.0
LC_SPAN_HZ = 180.0

PILLAR_R = 1.2
PILLAR_H = 8.0


class PillarWorld(FlatGroundWorld):
    """Плоский грунт и один тёмный столб сбоку от траектории."""

    def __init__(self, x: float, y: float) -> None:
        super().__init__(name="pillar_world", half_size=300)
        self.mjcf_root.worldbody.add_geom(
            type=GEOM_TYPES["cylinder"], name="pillar",
            size=[PILLAR_R, PILLAR_H / 2, 0.0],
            pos=[x, y, PILLAR_H / 2],
            rgba=[0.05, 0.05, 0.05, 1.0],
            contype=0, conaffinity=0,
        )


def build_brain(device):
    print("[мозг] читаю нейроны и аннотации...")
    comp = pd.read_csv(path_comp, index_col=0)
    flyid2i = {int(j): i for i, j in enumerate(comp.index)}
    n = len(flyid2i)

    ann = pd.read_csv(ANNOTATIONS, sep="\t", low_memory=False)
    ann["root_id"] = pd.to_numeric(ann["root_id"], errors="coerce")
    ann = ann.dropna(subset=["root_id"])
    ann["root_id"] = ann["root_id"].astype("int64")
    ann = ann[ann["root_id"].isin(flyid2i.keys())]

    lc9 = ann[ann["cell_type"] == "LC9"]
    lc_l = [flyid2i[i] for i in lc9.loc[lc9["side"] == "left", "root_id"]]
    lc_r = [flyid2i[i] for i in lc9.loc[lc9["side"] == "right", "root_id"]]
    idx_l, idx_r = flyid2i[DNP09_LEFT], flyid2i[DNP09_RIGHT]

    print(f"[мозг] нейронов {n}; LC9 слева {len(lc_l)}, справа {len(lc_r)}")
    print(f"[мозг] DNp09 left index={idx_l}, right index={idx_r}")

    print("[мозг] загружаю веса...")
    weights = rp.get_weights(str(path_con), str(path_comp), str(path_wt), csr=True).to(device)
    model = rp.TorchModel(1, n, DT_BRAIN_MS, rp.MODEL_PARAMS, weights,
                          exc_indices=sorted(set(lc_l) | set(lc_r)), device=device)
    return model, n, idx_l, idx_r, lc_l, lc_r


def build_body(pillar_x, pillar_y, with_camera):
    print("[тело] собираю модель со зрением...")
    fly = make_locomotion_fly()
    fly.add_vision()
    cam = fly.add_tracking_camera(name="trackcam") if with_camera else None
    world = PillarWorld(pillar_x, pillar_y)
    world.add_fly(fly, spawn_position=[0.0, 0.0, 0.5],
                  spawn_rotation=Rotation3D("quat", [1, 0, 0, 0]),
                  add_ground_contact_sensors=True)
    sim = Simulation(world)
    sim.reset()
    print(f"[тело] шаг физики {sim.timestep * 1000:.4f} мс, "
          f"столб в ({pillar_x:.1f}, {pillar_y:+.1f}) мм")
    return sim, fly, cam


def eye_darkness(readouts, baseline):
    """Свести показания каждого глаза к одному числу: насколько потемнело.

    Каждый омматидий отдаёт ненулевое значение только в своём канале
    (жёлтый или бледный), поэтому суммируем по каналам, а не усредняем.
    """
    per_om = readouts.sum(axis=2)          # (2 глаза, 721 омматидий)
    inten = per_om.mean(axis=1)            # средняя яркость по глазу
    dark = np.clip((baseline - inten) / np.maximum(baseline, 1e-6), 0.0, 1.0)
    return inten, dark


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=120)
    ap.add_argument("--pillar-x", type=float, default=12.0)
    ap.add_argument("--pillar-y", type=float, default=3.0,
                    help="сторона столба: >0 слева, <0 справа, 0 — прямо по курсу")
    ap.add_argument("--no-pillar", action="store_true",
                    help="контроль: убрать столб далеко, поле зрения пустое")
    ap.add_argument("--tag", type=str, default="vision")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--autocal", type=int, default=15,
                    help="циклов на замер базовой яркости и потолка DNp09")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 78)
    print(" ЗРИТЕЛЬНЫЙ ЗАМКНУТЫЙ КОНТУР: глаза -> LC9 -> DNp09 -> походка")
    print("=" * 78)
    px, py = (500.0, 500.0) if args.no_pillar else (args.pillar_x, args.pillar_y)
    print(f"device={device}, циклов={args.cycles}, "
          f"столб {'убран' if args.no_pillar else f'в ({px:.1f}, {py:+.1f})'}")

    model, n_neurons, idx_l, idx_r, lc_l, lc_r = build_brain(device)
    cond, delay_buf, spikes, v, refrac = model.state_init()

    sim, fly, cam = build_body(px, py, args.video)
    if args.video:
        sim.set_renderer(cam, camera_res=(360, 480), playback_speed=0.2, output_fps=25)

    controller = HybridTurningController(
        timestep=sim.timestep, preprogrammed_steps=PreprogrammedSteps())
    controller.reset(seed=0)
    sim.warmup(0.05)

    inner_brain = int(SYNC_MS / DT_BRAIN_MS)
    inner_phys = int(round(SYNC_MS / 1000.0 / sim.timestep))

    rates = torch.zeros(1, n_neurons, device=device)
    dn_idx = torch.tensor([idx_l, idx_r], dtype=torch.long, device=device)
    dn_acc = torch.zeros(2, device=device)
    gen = torch.Generator(device=device)
    gen.manual_seed(0)
    alpha = 1.0 - float(np.exp(-SYNC_MS / TAU_CMD_MS))

    body_order = fly.get_bodysegs_order()
    bodyseg_cls = type(fly).BODY_SEGMENT_CLASS
    thorax_idx = body_order.index(bodyseg_cls("c_thorax"))
    thorax_body_id = sim._internal_bodyids_by_fly[fly.name][thorax_idx]

    def step_body(cmd):
        for _ in range(inner_phys):
            obs = HybridControllerObservation.from_sim(sim, fly.name)
            apply_locomotion_action(sim, fly.name, controller.step(cmd, obs))
            sim.step()
            if args.video:
                sim.render_as_needed()

    def step_brain(rate_l, rate_r):
        nonlocal cond, delay_buf, spikes, v, refrac
        rates.zero_()
        rates[:, lc_l] = rate_l
        rates[:, lc_r] = rate_r
        dn_acc.zero_()
        for _ in range(inner_brain):
            cond, delay_buf, spikes, v, refrac = model(
                rates, cond, delay_buf, spikes, v, refrac, generator=gen)
            dn_acc += spikes[0, dn_idx]
        a, b = (float(x) / (SYNC_MS / 1000.0) for x in dn_acc.tolist())
        return a, b

    # --- калибровка: базовая яркость пустого поля и потолок каждого канала ---
    print(f"[калибровка] {args.autocal} циклов...")
    base_int, ceil_l, ceil_r = [], [], []
    with torch.no_grad():
        for c in range(args.autocal):
            step_body(np.array([0.8, 0.8]))
            ro = sim.get_ommatidia_readouts(fly.name)
            base_int.append(ro.sum(axis=2).mean(axis=1))
            a, b = step_brain(LC_BASE_HZ + LC_SPAN_HZ, LC_BASE_HZ + LC_SPAN_HZ)
            if c >= args.autocal // 2:
                ceil_l.append(a)
                ceil_r.append(b)
    baseline = np.array(base_int).mean(axis=0)
    ref_l = max(float(np.mean(ceil_l)), 1.0)
    ref_r = max(float(np.mean(ceil_r)), 1.0)
    print(f"[калибровка] базовая яркость: левый {baseline[0]:.4f}, правый {baseline[1]:.4f}")
    print(f"[калибровка] потолок DNp09: левый {ref_l:.1f} Гц, правый {ref_r:.1f} Гц")

    headers = ["cycle", "t_sec", "eye_left_int", "eye_right_int",
               "dark_left", "dark_right", "lc_rate_left_hz", "lc_rate_right_hz",
               "dnp09_left_hz", "dnp09_right_hz", "cmd_left", "cmd_right",
               "thorax_x_mm", "thorax_y_mm", "heading_deg", "dist_to_pillar_mm"]
    rows = []
    ema_l = ema_r = None

    print(f"\n[цикл] шагов мозга={inner_brain}, шагов физики={inner_phys}")
    t0 = time.perf_counter()
    with torch.no_grad():
        for cycle in range(args.cycles):
            ro = sim.get_ommatidia_readouts(fly.name)
            inten, dark = eye_darkness(ro, baseline)

            rate_l = LC_BASE_HZ + LC_SPAN_HZ * float(dark[0])
            rate_r = LC_BASE_HZ + LC_SPAN_HZ * float(dark[1])
            hz_l, hz_r = step_brain(rate_l, rate_r)

            if ema_l is None:
                ema_l, ema_r = hz_l, hz_r
            else:
                ema_l += alpha * (hz_l - ema_l)
                ema_r += alpha * (hz_r - ema_r)
            cmd_l = float(np.clip(ema_l / ref_l, 0.0, 1.0))
            cmd_r = float(np.clip(ema_r / ref_r, 0.0, 1.0))

            step_body(np.array([cmd_l, cmd_r]))

            pos = sim.get_body_positions(fly.name)[thorax_idx]
            m = sim.mj_data.xmat[thorax_body_id].reshape(3, 3)
            heading = float(np.degrees(np.arctan2(m[1, 0], m[0, 0])))
            dist = float(np.hypot(px - pos[0], py - pos[1]))

            rows.append([cycle, round(cycle * SYNC_MS / 1000.0, 4),
                         round(float(inten[0]), 4), round(float(inten[1]), 4),
                         round(float(dark[0]), 4), round(float(dark[1]), 4),
                         round(rate_l, 1), round(rate_r, 1),
                         round(hz_l, 1), round(hz_r, 1),
                         round(cmd_l, 4), round(cmd_r, 4),
                         round(float(pos[0]), 4), round(float(pos[1]), 4),
                         round(heading, 2), round(dist, 3)])

            if cycle % 10 == 0 or cycle == args.cycles - 1:
                print(f"  [{cycle:3d}/{args.cycles}] глаза {inten[0]:.3f}/{inten[1]:.3f} "
                      f"темнота {dark[0]:.3f}/{dark[1]:.3f}  LC {rate_l:.0f}/{rate_r:.0f}Гц  "
                      f"DNp09 {hz_l:.0f}/{hz_r:.0f}Гц  cmd {cmd_l:.2f}/{cmd_r:.2f}  "
                      f"x={pos[0]:.1f} y={pos[1]:.1f} курс={heading:+.0f}°")

    elapsed = time.perf_counter() - t0
    print(f"\nготово за {elapsed:.0f} с ({elapsed / args.cycles:.2f} с на цикл)")

    out_csv = out(f"closed_loop_vision_{args.tag}.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    print(f"лог: {out_csv}")

    if args.video:
        vp = out(f"closed_loop_vision_{args.tag}.mp4")
        sim.renderer.save_video(vp)
        print(f"видео: {vp}")

    df = pd.DataFrame(rows, columns=headers)
    print("\n" + "=" * 78)
    print(" ИТОГ")
    print("=" * 78)
    dx = df["thorax_x_mm"].iloc[-1] - df["thorax_x_mm"].iloc[0]
    dy = df["thorax_y_mm"].iloc[-1] - df["thorax_y_mm"].iloc[0]
    turn = df["heading_deg"].iloc[-1] - df["heading_deg"].iloc[0]
    turn = (turn + 180) % 360 - 180
    print(f"  смещение: dx={dx:+.2f} dy={dy:+.2f} мм, путь {np.hypot(dx, dy):.2f} мм")
    print(f"  поворот курса за прогон: {turn:+.1f}°")
    print(f"  темнота левого глаза:  среднее {df['dark_left'].mean():.4f}, "
          f"максимум {df['dark_left'].max():.4f}")
    print(f"  темнота правого глаза: среднее {df['dark_right'].mean():.4f}, "
          f"максимум {df['dark_right'].max():.4f}")
    print(f"  cmd_left  среднее {df['cmd_left'].mean():.3f}, разброс {df['cmd_left'].std():.3f}")
    print(f"  cmd_right среднее {df['cmd_right'].mean():.3f}, разброс {df['cmd_right'].std():.3f}")
    if not args.no_pillar:
        print(f"  дистанция до столба: старт {df['dist_to_pillar_mm'].iloc[0]:.1f}, "
              f"минимум {df['dist_to_pillar_mm'].min():.1f}, "
              f"финиш {df['dist_to_pillar_mm'].iloc[-1]:.1f} мм")
    asym = df["cmd_left"] - df["cmd_right"]
    print(f"  асимметрия команды (лево-право): среднее {asym.mean():+.4f}")


if __name__ == "__main__":
    main()
