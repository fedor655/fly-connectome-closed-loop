"""Зрительный замкнутый контур: глаза мухи -> LC9 -> DNp09 -> походка.

Почему именно так. Разбор входов DNp09 по аннотациям FlyWire показал, что
единственный крупный источник чистого возбуждения для него — зрительные
проекционные нейроны (+880 при 22.8% массы веса), а вход от ног через
восходящие даёт 1.8% и в сумме тормозит. Прямая проверка подтвердила:
стимуляция 1736 восходящих и 2656 механосенсорных нейронов не даёт DNp09
ни одного спайка, а LC9 зажигают его до 159 Гц, причём строго ипсилатерально —
левые LC9 действуют только на левый DNp09, перекрёстных наводок ноль.

Схема одного цикла 15 мс:

    левый глаз  -> темнота в поле зрения -> Пуассон на LC9 слева  (87 нейронов)
    правый глаз -> темнота в поле зрения -> Пуассон на LC9 справа (92 нейрона)
                        |
                мозг, 150 шагов по 0.1 мс
                        |
            частота DNp09 -> сглаживание -> команда [0..1]^2
                        |
            HybridTurningController -> CPG -> 150 шагов физики

Честная оговорка. Внутриглазной ретинотопии здесь НЕТ: весь глаз сводится к
одному числу, и все LC9 своей стороны получают одинаковую стимуляцию.
Это не упрощение ради простоты, а следствие измерения: положение LC9 не
предсказывает вес его связи с DNp09 (корреляция +0.20 слева и -0.10 справа),
поэтому различать участки поля зрения ЭТИМ выходом невозможно в принципе.
Подробности в tools/lc_retinotopy.py.

Поведение специально не закладывалось: ни избегание, ни приближение. Мы просто
соединили измеренный путь.
"""
from __future__ import annotations

import argparse
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
# MUJOCO_GL под платформу выставляет flypaths, импортированный выше

from benchmark import path_comp, path_con, path_wt  # noqa: E402
import run_pytorch as rp  # noqa: E402

from flyreplay import FAR_AWAY, PillarWorld, Recorder  # noqa: E402
from tools.visual_field_map import N_STRIPS, load_or_build  # noqa: E402
from flygym.simulation import Simulation  # noqa: E402
from flygym.utils.math import Rotation3D  # noqa: E402
from flygym_demo.complex_terrain import (  # noqa: E402
    HybridControllerObservation,
    HybridTurningController,
    PreprogrammedSteps,
    apply_locomotion_action,
    make_locomotion_fly,
)

# Стороны по аннотациям FlyWire; в benchmark.py подписи перепутаны местами.
DNP09_LEFT = 720575940635872101
DNP09_RIGHT = 720575940627652358

DT_BRAIN_MS = 0.1
SYNC_MS = 15.0

# Сглаживание команды. Обосновано измерением d-prime: при 15 мс окне DNp09
# выдаёт единицы спайков, и мгновенная оценка частоты квантуется шагом 66.7 Гц.
# См. tools/p9_noise_floor.py.
TAU_CMD_MS = 150.0

# Рабочая точка по измеренной кривой (tools/visual_to_dn.py): LC9 на 200 Гц
# дают DNp09 около 110 Гц. Ниже 100 Гц нельзя — там 0-1 спайк на окно.
LC_BASE_HZ = 100.0
LC_SPAN_HZ = 300.0

# Рабочая точка команды: при ПУСТОМ поле зрения муха должна идти нормально,
# объект лишь модулирует. Первая версия калибровалась по максимальной
# стимуляции, из-за чего при пустом поле команда выходила 0.04, муха не шла,
# до объекта не доходила и зрение ничего не видело — замкнутый круг.
TARGET_CMD_AT_BASE = 0.65

# Усиление сигнала темноты: объект меняет среднюю яркость глаза на единицы
# процентов, без усиления модуляция тонет.
DARK_GAIN = 6.0

# Низкочастотный фильтр яркости каждого глаза.
#
# При ходьбе тело качает по крену, яркость глаз колеблется с периодом 45 мс
# (22 Гц — вторая гармоника CPG). Измерено: в контроле без объекта корреляция
# глаз -0.915, то есть помеха ПРОТИВОФАЗНА. Бинокулярная разность, которая
# напрашивалась первой, её бы усилила, а не подавила: синфазная составляющая
# там в десять раз меньше разностной. Разделять надо по времени.
#
# Постоянная подобрана по данным (tools/analyze_selfmotion.py): отношение
# сигнала к помехе растёт с 28 при нуле до 97 при 100 мс и 108 при 150 мс,
# дальше падает. Берём 100 мс — почти максимум при меньшей задержке.
TAU_EYE_MS = 100.0

# Сцена и запись траектории живут в flyreplay.py: тем же кодом её пересобирает
# просмотрщик, иначе запись и воспроизведение разъедутся.


def _select_population(device, weights, n, flyid2i, ann, verbose):
    """Отобрать латерализованные нисходящие, реально отзывающиеся на зрение.

    Отбирать по анатомическим весам нельзя: уже дважды выяснялось, что вес
    связи не равен функциональному влиянию (Sugar GRNs до P9, DNg108 от
    восходящих). Поэтому стимулируем зрительные проекционные каждой стороны и
    берём тех нисходящих, кто загорается на своей стороне и молчит на чужой.

    Результат кэшируется: сам отбор занимает около минуты, а зависит только от
    коннектома, то есть не меняется от прогона к прогону.
    """
    cache = Path(out("dn_population_cache.npz"))
    if cache.exists():
        z = np.load(cache)
        if verbose:
            print(f"[мозг] популяция из кэша: слева {len(z['left'])}, "
                  f"справа {len(z['right'])}")
        return z["left"].tolist(), z["right"].tolist()

    if verbose:
        print("[мозг] отбираю популяцию нисходящих (кэша нет, около минуты)...")
    vis = ann[ann["super_class"] == "visual_projection"]
    vis_l = [flyid2i[i] for i in vis.loc[vis["side"] == "left", "root_id"]]
    vis_r = [flyid2i[i] for i in vis.loc[vis["side"] == "right", "root_id"]]

    desc = ann[(ann["super_class"] == "descending") & ann["side"].isin(["left", "right"])]
    dn_idx = [flyid2i[int(x)] for x in desc["root_id"]]
    dn_side = np.array(desc["side"].tolist())
    idx_t = torch.tensor(dn_idx, dtype=torch.long, device=device)

    def probe(stim):
        m = rp.TorchModel(1, n, DT_BRAIN_MS, rp.MODEL_PARAMS, weights,
                          exc_indices=list(stim), device=device)
        c, d, s, v, r = m.state_init()
        rates = torch.zeros(1, n, device=device)
        rates[:, stim] = 150.0
        g = torch.Generator(device=device)
        g.manual_seed(2024)
        acc = torch.zeros(len(dn_idx), device=device)
        n_tr, n_me = int(300.0 / DT_BRAIN_MS), int(2000.0 / DT_BRAIN_MS)
        with torch.no_grad():
            for step in range(n_tr + n_me):
                c, d, s, v, r = m(rates, c, d, s, v, r, generator=g)
                if step >= n_tr:
                    acc.add_(s[0, idx_t])
        return np.array(acc.tolist()) / 2.0

    hz_l, hz_r = probe(vis_l), probe(vis_r)
    ipsi = np.where(dn_side == "left", hz_l, hz_r)
    contra = np.where(dn_side == "left", hz_r, hz_l)
    tot = ipsi + contra
    lat = np.where(tot > 0, (ipsi - contra) / np.maximum(tot, 1e-9), 0.0)
    keep = (ipsi > 5.0) & (lat > 0.5)

    arr = np.array(dn_idx)
    sel_l = arr[keep & (dn_side == "left")]
    sel_r = arr[keep & (dn_side == "right")]
    np.savez(cache, left=sel_l, right=sel_r)
    if verbose:
        print(f"[мозг] отобрано: слева {len(sel_l)}, справа {len(sel_r)}")
    return sel_l.tolist(), sel_r.tolist()


def load_brain_assets(device, verbose=True, readout="population", stim="visual"):
    """Загрузить коннектом и найти нужные популяции. Делается ОДИН раз на серию:
    чтение parquet и весов занимает около минуты, и повторять его на каждый
    прогон бессмысленно."""
    if verbose:
        print("[мозг] читаю нейроны и аннотации...")
    comp = pd.read_csv(path_comp, index_col=0)
    flyid2i = {int(j): i for i, j in enumerate(comp.index)}
    n = len(flyid2i)

    ann = pd.read_csv(ANNOTATIONS, sep="\t", low_memory=False)
    ann["root_id"] = pd.to_numeric(ann["root_id"], errors="coerce")
    ann = ann.dropna(subset=["root_id"])
    ann["root_id"] = ann["root_id"].astype("int64")
    ann = ann[ann["root_id"].isin(flyid2i.keys())]

    if stim == "lc9":
        src = ann[ann["cell_type"] == "LC9"]
    else:
        src = ann[ann["super_class"] == "visual_projection"]
    lc_l = [flyid2i[i] for i in src.loc[src["side"] == "left", "root_id"]]
    lc_r = [flyid2i[i] for i in src.loc[src["side"] == "right", "root_id"]]

    if verbose:
        print(f"[мозг] нейронов {n}; вход «{stim}»: слева {len(lc_l)}, справа {len(lc_r)}")
        print("[мозг] загружаю веса...")
    weights = rp.get_weights(str(path_con), str(path_comp), str(path_wt), csr=True).to(device)

    if readout == "population":
        read_l, read_r = _select_population(device, weights, n, flyid2i, ann, verbose)
    else:
        read_l, read_r = [flyid2i[DNP09_LEFT]], [flyid2i[DNP09_RIGHT]]
        if verbose:
            print("[мозг] считывание: одиночный DNp09 с каждой стороны")

    return {"weights": weights, "n": n, "read_l": read_l, "read_r": read_r,
            "lc_l": lc_l, "lc_r": lc_r, "readout": readout, "stim": stim}


LOG_COLUMNS = ["cycle", "t_sec", "eye_left_int", "eye_right_int",
               "dark_left", "dark_right", "lc_rate_left_hz", "lc_rate_right_hz",
               "dnp09_left_hz", "dnp09_right_hz", "cmd_left", "cmd_right",
               "thorax_x_mm", "thorax_y_mm", "heading_deg", "dist_to_pillar_mm"]


def strip_intensity(raw, om_strip, n_strips):
    """Средняя яркость по полосам поля зрения: (2, 721) -> (2, n_strips).

    Поправка А: карта омматидиев `om_strip` тоже формы (2, 721) — глаза
    зеркальны по азимуту (левый flip=False, правый flip=True), поэтому у
    каждого глаза своя строка карты, и сравнивать нужно `raw[k]` со
    `om_strip[k]`, а не с общей картой на оба глаза.

    При n_strips=1 это ровно прежнее .mean(axis=1) — скалярный режим остаётся
    частным случаем пространственного, а не отдельной веткой кода, которая
    могла бы с ним разойтись.
    """
    return np.stack([[raw[k][om_strip[k] == s].mean() for s in range(n_strips)]
                     for k in (0, 1)])


def run_trial(assets, device, *, pillar_y=3.0, no_pillar=False, pillar_x=12.0,
              cycles=100, autocal=15, cal_brain_ms=3000.0, seed=0,
              tau_eye=TAU_EYE_MS, tau_cmd=TAU_CMD_MS,
              lc_base=None, lc_span=None, spatial=False,
              pillar_z=None, pillar_h=None,
              video_path=None, traj_path=None, traj_every=10, verbose=True):
    """Один прогон. Возвращает (DataFrame лога, словарь сводки)."""
    px = FAR_AWAY if no_pillar else pillar_x
    py = FAR_AWAY if no_pillar else pillar_y

    n = assets["n"]
    read_l, read_r = assets["read_l"], assets["read_r"]
    lc_l, lc_r = assets["lc_l"], assets["lc_r"]

    # Скалярный режим — одна полоса на глаз. Один и тот же код обслуживает оба
    # режима, поэтому контроль «скалярный вход на тех же сценах» не может
    # разойтись с основным по реализации (поправка В: число полос берётся из
    # карты, а не из максимума индекса).
    #
    # Поправка Б: в ОБОИХ режимах группы — это пересечение множества карты с
    # `lc_l`/`lc_r`, а не всё множество карты. Карта покрывает 3973 из 4008
    # левых проекционных и 4007 из 4030 правых (не хватает выхода оцеллярного
    # глаза и нейронов без синапсов от медуллы); если бы пространственный
    # режим стимулировал только покрытых картой, а скалярный — вообще всех,
    # то режимы отличались бы не только пространственностью, но и составом
    # нейронов, а скалярный режим — это ОБЯЗАТЕЛЬНЫЙ КОНТРОЛЬ ценности карты,
    # и такое расхождение состава его портит. Пересечение с `lc_l`/`lc_r`
    # нужно ещё и потому, что при `--stim lc9` этот набор — только LC9, и
    # карта не должна расширять стимуляцию за его пределы.
    # Названа vfm, а не m: в основном цикле ниже уже есть матрица поворота
    # тела (sim.mj_data.xmat) — короткое имя `m` для неё напрашивалось бы и
    # столкнулось с картой.
    vfm = load_or_build()
    if spatial:
        n_strips = N_STRIPS
        om_strip = vfm["ommatidia"]
        grp_l = [vfm["left_idx"][(vfm["left_strip"] == s) & np.isin(vfm["left_idx"], lc_l)]
                 for s in range(n_strips)]
        grp_r = [vfm["right_idx"][(vfm["right_strip"] == s) & np.isin(vfm["right_idx"], lc_r)]
                 for s in range(n_strips)]
    else:
        n_strips = 1
        om_strip = np.zeros((2, 721), dtype=int)
        grp_l = [vfm["left_idx"][np.isin(vfm["left_idx"], lc_l)]]
        grp_r = [vfm["right_idx"][np.isin(vfm["right_idx"], lc_r)]]

    # Широкий вход (все зрительные проекционные, около 4000 нейронов) требует
    # вчетверо меньших частот, чем узкий по LC9 (87 нейронов), иначе сеть
    # уходит в насыщение.
    if lc_base is None:
        lc_base = 60.0 if assets.get("stim") == "visual" else LC_BASE_HZ
    if lc_span is None:
        lc_span = 140.0 if assets.get("stim") == "visual" else LC_SPAN_HZ

    model = rp.TorchModel(1, n, DT_BRAIN_MS, rp.MODEL_PARAMS, assets["weights"],
                          exc_indices=sorted(set(lc_l) | set(lc_r)), device=device)
    cond, delay_buf, spikes, v, refrac = model.state_init()

    fly = make_locomotion_fly()
    fly.add_vision()
    cam = fly.add_tracking_camera(name="trackcam") if video_path else None
    world = (PillarWorld(px, py) if pillar_z is None
             else PillarWorld(px, py, z=pillar_z, h=pillar_h))
    world.add_fly(fly, spawn_position=[0.0, 0.0, 0.5],
                  spawn_rotation=Rotation3D("quat", [1, 0, 0, 0]),
                  add_ground_contact_sensors=True)
    sim = Simulation(world)
    sim.reset()
    if video_path:
        sim.set_renderer(cam, camera_res=(360, 480), playback_speed=0.2, output_fps=25)

    controller = HybridTurningController(
        timestep=sim.timestep, preprogrammed_steps=PreprogrammedSteps())
    controller.reset(seed=seed)
    sim.warmup(0.05)

    inner_brain = int(SYNC_MS / DT_BRAIN_MS)
    inner_phys = int(round(SYNC_MS / 1000.0 / sim.timestep))

    rates = torch.zeros(1, n, device=device)
    # Считываем СРЕДНЮЮ частоту популяции с каждой стороны. Одиночный нейрон
    # даёт полтора спайка за окно и около 60 процентов шума — именно это
    # разваливало поведение в предыдущей серии.
    idx_read_l = torch.tensor(read_l, dtype=torch.long, device=device)
    idx_read_r = torch.tensor(read_r, dtype=torch.long, device=device)
    n_read_l, n_read_r = max(len(read_l), 1), max(len(read_r), 1)
    dn_acc = torch.zeros(2, device=device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    alpha_cmd = 1.0 - float(np.exp(-SYNC_MS / tau_cmd)) if tau_cmd > 0 else 1.0
    alpha_eye = 1.0 - float(np.exp(-SYNC_MS / tau_eye)) if tau_eye > 0 else 1.0

    body_order = fly.get_bodysegs_order()
    bodyseg_cls = type(fly).BODY_SEGMENT_CLASS
    thorax_idx = body_order.index(bodyseg_cls("c_thorax"))
    thorax_body_id = sim._internal_bodyids_by_fly[fly.name][thorax_idx]

    rec = Recorder(sim, every=traj_every) if traj_path else None

    def step_body(cmd):
        for _ in range(inner_phys):
            obs = HybridControllerObservation.from_sim(sim, fly.name)
            apply_locomotion_action(sim, fly.name, controller.step(cmd, obs))
            sim.step()
            if rec:
                rec.grab()
            if video_path:
                sim.render_as_needed()

    def step_brain(rate_l, rate_r):
        """rate_l, rate_r — массивы длины n_strips: своя частота на свою полосу."""
        nonlocal cond, delay_buf, spikes, v, refrac
        rates.zero_()
        for s in range(n_strips):
            rates[:, grp_l[s]] = float(rate_l[s])
            rates[:, grp_r[s]] = float(rate_r[s])
        dn_acc.zero_()
        for _ in range(inner_brain):
            cond, delay_buf, spikes, v, refrac = model(
                rates, cond, delay_buf, spikes, v, refrac, generator=gen)
            # add_, а не +=: augmented assignment во вложенной функции сделал бы
            # dn_acc локальной и уронил бы zero_() выше
            dn_acc[0] += spikes[0, idx_read_l].sum()
            dn_acc[1] += spikes[0, idx_read_r].sum()
        a, b = dn_acc.tolist()
        t_win = SYNC_MS / 1000.0
        return a / n_read_l / t_win, b / n_read_r / t_win

    # --- Калибровка в два приёма ---
    # Яркость меряем на теле (нужен реальный обзор), а отклик мозга — отдельным
    # длинным прогоном без шагания. Первая версия мерила отклик по нескольким
    # окнам синхронизации, то есть по 4-8 спайкам: двукратная разница между
    # каналами оказывалась шумом счёта, нормировка выходила заниженной, правый
    # канал сидел в насыщении, и муха разворачивалась на 69 градусов даже без
    # объекта. Поймано контролем.
    base_int = []
    with torch.no_grad():
        for _ in range(autocal):
            step_body(np.array([0.85, 0.85]))
            base_int.append(strip_intensity(
                sim.get_ommatidia_readouts(fly.name).sum(axis=2), om_strip, n_strips))
    baseline = np.array(base_int).mean(axis=0)          # (2, n_strips)

    n_cal = int(cal_brain_ms / SYNC_MS)
    sp_l = sp_r = 0.0
    with torch.no_grad():
        flat = np.full(n_strips, lc_base)
        for w in range(n_cal):
            a, b = step_brain(flat, flat)
            if w >= n_cal // 4:                # четверть на переходный процесс
                sp_l += a * (SYNC_MS / 1000.0)
                sp_r += b * (SYNC_MS / 1000.0)
    t_cal = max(n_cal - n_cal // 4, 1) * SYNC_MS / 1000.0
    ref_l = max(sp_l / t_cal, 1.0) / TARGET_CMD_AT_BASE
    ref_r = max(sp_r / t_cal, 1.0) / TARGET_CMD_AT_BASE
    if verbose:
        # baseline теперь (2, n_strips): в печать идёт среднее по полосам —
        # тот же смысл, что раньше было одно число на глаз.
        print(f"  [калибровка] яркость {baseline[0].mean():.4f}/{baseline[1].mean():.4f}, "
              f"спайков {sp_l:.0f}/{sp_r:.0f}, норма {ref_l:.1f}/{ref_r:.1f}")

    # --- Основной цикл ---
    rows = []
    ema_l = ema_r = None
    eye_filt = None
    t0 = time.perf_counter()
    with torch.no_grad():
        for cycle in range(cycles):
            inten_raw = strip_intensity(
                sim.get_ommatidia_readouts(fly.name).sum(axis=2), om_strip, n_strips)
            if eye_filt is None:
                eye_filt = inten_raw.copy()
            else:
                eye_filt += alpha_eye * (inten_raw - eye_filt)
            rel = (baseline - eye_filt) / np.maximum(baseline, 1e-6)
            dark = np.clip(rel * DARK_GAIN, 0.0, 1.0)          # (2, n_strips)

            rate_l = lc_base + lc_span * dark[0]                # (n_strips,)
            rate_r = lc_base + lc_span * dark[1]
            hz_l, hz_r = step_brain(rate_l, rate_r)

            if ema_l is None:
                ema_l, ema_r = hz_l, hz_r
            else:
                ema_l += alpha_cmd * (hz_l - ema_l)
                ema_r += alpha_cmd * (hz_r - ema_r)
            cmd_l = float(np.clip(ema_l / ref_l, 0.0, 1.0))
            cmd_r = float(np.clip(ema_r / ref_r, 0.0, 1.0))

            step_body(np.array([cmd_l, cmd_r]))

            pos = sim.get_body_positions(fly.name)[thorax_idx]
            rot = sim.mj_data.xmat[thorax_body_id].reshape(3, 3)
            heading = float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0])))
            dist = float(np.hypot(px - pos[0], py - pos[1]))

            # Сводные колонки (среднее по полосам) остаются на прежних местах:
            # tools/replicate_vision.py и tools/analyze_replication.py читают
            # именно их, и переписывать разбор ради нового режима незачем.
            row = [cycle, cycle * SYNC_MS / 1000.0,
                   float(eye_filt[0].mean()), float(eye_filt[1].mean()),
                   float(dark[0].mean()), float(dark[1].mean()),
                   float(rate_l.mean()), float(rate_r.mean()),
                   hz_l, hz_r, cmd_l, cmd_r,
                   float(pos[0]), float(pos[1]), heading, dist]
            if n_strips > 1:
                row += [float(x) for x in dark[0]] + [float(x) for x in dark[1]]
                row += [float(x) for x in rate_l] + [float(x) for x in rate_r]
            rows.append(row)

    elapsed = time.perf_counter() - t0
    cols = list(LOG_COLUMNS)
    if n_strips > 1:
        cols += [f"dark_l{s}" for s in range(n_strips)]
        cols += [f"dark_r{s}" for s in range(n_strips)]
        cols += [f"rate_l{s}" for s in range(n_strips)]
        cols += [f"rate_r{s}" for s in range(n_strips)]
    df = pd.DataFrame(rows, columns=cols)

    if video_path:
        sim.renderer.save_video(video_path)
    if rec:
        rec.save(traj_path, (px, py))
    sim.close()

    turn = df["heading_deg"].iloc[-1] - df["heading_deg"].iloc[0]
    turn = (turn + 180) % 360 - 180

    # Связь зрения с командой ВНУТРИ прогона: сто точек вместо одной.
    # Сравнение сцен между собой даёт по одному числу на прогон, а здесь видно,
    # передаётся ли асимметрия поля зрения в асимметрию команды на каждом шаге.
    d_asym = (df["dark_left"] - df["dark_right"]).to_numpy()
    c_asym = (df["cmd_left"] - df["cmd_right"]).to_numpy()
    if d_asym.std() > 1e-9 and c_asym.std() > 1e-9:
        corr_now = float(np.corrcoef(d_asym, c_asym)[0, 1])
        # мозг отвечает не мгновенно: сдвигаем команду на цикл вперёд
        corr_lag = float(np.corrcoef(d_asym[:-1], c_asym[1:])[0, 1])
    else:
        corr_now = corr_lag = float("nan")
    dx = df["thorax_x_mm"].iloc[-1] - df["thorax_x_mm"].iloc[0]
    dy = df["thorax_y_mm"].iloc[-1] - df["thorax_y_mm"].iloc[0]
    summary = {
        "seed": seed, "no_pillar": bool(no_pillar), "pillar_y": None if no_pillar else pillar_y,
        "readout": assets.get("readout", "?"), "stim": assets.get("stim", "?"),
        "spatial": bool(spatial), "n_strips": n_strips,
        "n_read_left": len(read_l), "n_read_right": len(read_r),
        "turn_deg": turn, "path_mm": float(np.hypot(dx, dy)),
        "dx_mm": float(dx), "dy_mm": float(dy),
        "dark_left_mean": float(df["dark_left"].mean()),
        "dark_right_mean": float(df["dark_right"].mean()),
        "dark_left_max": float(df["dark_left"].max()),
        "dark_right_max": float(df["dark_right"].max()),
        "cmd_left_mean": float(df["cmd_left"].mean()),
        "cmd_right_mean": float(df["cmd_right"].mean()),
        "cmd_asym_mean": float((df["cmd_left"] - df["cmd_right"]).mean()),
        "corr_dark_cmd": corr_now,
        "corr_dark_cmd_lag1": corr_lag,
        "dist_min_mm": None if no_pillar else float(df["dist_to_pillar_mm"].min()),
        "dist_end_mm": None if no_pillar else float(df["dist_to_pillar_mm"].iloc[-1]),
        "elapsed_s": elapsed,
    }
    return df, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=100)
    ap.add_argument("--pillar-x", type=float, default=12.0)
    ap.add_argument("--pillar-y", type=float, default=3.0,
                    help="сторона столба: >0 слева, <0 справа")
    ap.add_argument("--no-pillar", action="store_true",
                    help="контроль: столб убран, поле зрения пустое")
    ap.add_argument("--tag", type=str, default="vision")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--record", action="store_true",
                    help="писать траекторию в npz: потом её смотрят "
                         "replay_view.py и рендерят replay_render.py")
    ap.add_argument("--record-every", type=int, default=10,
                    help="каждый N-й шаг физики (шаг 0.1 мс)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--autocal", type=int, default=15)
    ap.add_argument("--cal-brain-ms", type=float, default=3000.0)
    ap.add_argument("--tau-eye", type=float, default=TAU_EYE_MS)
    ap.add_argument("--tau-cmd", type=float, default=TAU_CMD_MS)
    ap.add_argument("--readout", choices=("population", "dnp09"), default="population",
                    help="считывать среднее по популяции латерализованных "
                         "нисходящих или один DNp09 на сторону")
    ap.add_argument("--stim", choices=("visual", "lc9"), default="visual",
                    help="куда подавать яркость: во все зрительные проекционные "
                         "или только в LC9")
    ap.add_argument("--lc-base", type=float, default=None)
    ap.add_argument("--lc-span", type=float, default=None)
    ap.add_argument("--spatial", action="store_true",
                    help="подавать яркость по полосам поля зрения, а не одним "
                         "числом на глаз")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 78)
    print(" ЗРИТЕЛЬНЫЙ ЗАМКНУТЫЙ КОНТУР: глаза -> зрительные -> нисходящие -> походка")
    print("=" * 78)
    print(f"device={device}, циклов={args.cycles}, seed={args.seed}, "
          f"столб {'убран' if args.no_pillar else f'y={args.pillar_y:+.1f}'}")
    print(f"вход «{args.stim}», считывание «{args.readout}», "
          f"подача {'пространственная' if args.spatial else 'скалярная'}")

    assets = load_brain_assets(device, readout=args.readout, stim=args.stim)
    video_path = out(f"closed_loop_vision_{args.tag}.mp4") if args.video else None
    traj_path = out(f"closed_loop_vision_{args.tag}_traj.npz") if args.record else None

    df, s = run_trial(assets, device, pillar_y=args.pillar_y, no_pillar=args.no_pillar,
                      pillar_x=args.pillar_x, cycles=args.cycles, autocal=args.autocal,
                      cal_brain_ms=args.cal_brain_ms, seed=args.seed,
                      tau_eye=args.tau_eye, tau_cmd=args.tau_cmd,
                      lc_base=args.lc_base, lc_span=args.lc_span,
                      spatial=args.spatial,
                      video_path=video_path, traj_path=traj_path,
                      traj_every=args.record_every)

    out_csv = out(f"closed_loop_vision_{args.tag}.csv")
    df.round(4).to_csv(out_csv, index=False)
    print(f"\nготово за {s['elapsed_s']:.0f} с, лог: {out_csv}")
    if video_path:
        print(f"видео: {video_path}")
    if traj_path:
        print(f"траектория: {traj_path}")

    print("\n" + "=" * 78)
    print(" ИТОГ")
    print("=" * 78)
    print(f"  смещение: dx={s['dx_mm']:+.2f} dy={s['dy_mm']:+.2f} мм, "
          f"путь {s['path_mm']:.2f} мм")
    print(f"  поворот курса: {s['turn_deg']:+.1f}°")
    print(f"  темнота левого:  среднее {s['dark_left_mean']:.4f}, "
          f"максимум {s['dark_left_max']:.4f}")
    print(f"  темнота правого: среднее {s['dark_right_mean']:.4f}, "
          f"максимум {s['dark_right_max']:.4f}")
    print(f"  cmd_left {s['cmd_left_mean']:.3f}, cmd_right {s['cmd_right_mean']:.3f}")
    print(f"  асимметрия команды: {s['cmd_asym_mean']:+.4f}")
    if not args.no_pillar:
        print(f"  дистанция до столба: минимум {s['dist_min_mm']:.1f}, "
              f"финиш {s['dist_end_mm']:.1f} мм")


if __name__ == "__main__":
    main()
