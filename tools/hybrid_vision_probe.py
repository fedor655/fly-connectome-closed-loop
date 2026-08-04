"""Различает ли муха УЧАСТКИ поля зрения, а не только общую яркость?

Главный вопрос всего блока. До сих пор каждый глаз сводился к одному числу, и
муха могла отличить «слева темнее» от «справа темнее» и больше ничего. Теперь
собран сквозной путь:

  сцена -> глаз flygym (721 омматидий) -> решётка flyvis (шаг 1.1)
        -> обученные оптические доли flyvis
        -> нейроны FlyWire по разложению колонок (шаг 1.2)
        -> спайковая LIF -> зрительные проекционные нейроны

и вопрос ставится строго: столб слева и столб справа обязаны зажигать РАЗНЫЕ
НАБОРЫ зрительных проекционных, а не просто давать разную суммарную яркость.

Как отличить одно от другого. Сравнение «слева против справа» само по себе
ничего не доказывает: наборы могут различаться просто потому, что сцены разной
яркости. Поэтому в опыте есть два разных положения столба С ОДНОЙ стороны.
Тогда:

  если муха видит ПРОСТРАНСТВО, два левых положения похожи между собой сильнее,
  чем левое на правое;
  если она видит только яркость, все три сравнения выйдут одинаковыми.

Отрицательный контроль — сцена без столба.

Свободный параметр здесь один и назван явно: перевод безразмерной активности
flyvis в герцы. Он берётся один раз по контрольной сцене и дальше не меняется,
иначе сравнение сцен потеряет смысл.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flypaths import ANNOTATIONS, add_fly_brain_to_path, out  # noqa: E402

add_fly_brain_to_path()
from benchmark import path_comp, path_con, path_wt  # noqa: E402
import run_pytorch as rp  # noqa: E402

import flyvis  # noqa: E402
from flyreplay import PillarWorld  # noqa: E402
from flygym.simulation import Simulation  # noqa: E402
from flygym.utils.math import Rotation3D  # noqa: E402
from flygym_demo.complex_terrain import (  # noqa: E402
    HybridControllerObservation,
    HybridTurningController,
    PreprogrammedSteps,
    apply_locomotion_action,
    make_locomotion_fly,
)

DT_BRAIN_MS = 0.1
# Первая версия брала окно 300 мс. При частоте LC около 5 Гц это полтора спайка
# на нейрон, и сценозависимая часть тонула в шуме счёта: доля общей для всех
# сцен дисперсии выходила 99.9 процента. Окно удлинено.
BRAIN_MS = 2000.0
TRANSIENT_MS = 200.0
FLYVIS_MODEL = "flow/0000/000"
FLYVIS_FRAMES = 40
FLYVIS_DT = 1 / 100.0
MAX_HZ = 200.0          # рабочая точка стыка из развёртки по шуму считывания
WARM_STEPS = 40
# Снимать ли глаза последовательностью на ходу вместо одного кадра. Нужно для
# T4 и T5: на неподвижной картинке они дают ровно ноль (проверено в
# tools/flyvis_smoke_check.py: 0.000 против 0.06-0.08 у движущейся полосы).
MOVING = "--static" not in sys.argv

# Первая версия ставила столб в 12-18 мм, как рабочий контур, но там муха ИДЁТ
# на него и он вырастает. Здесь муха стоит, и на таком расстоянии столб менял
# яркость глаза на один процент, а вход на стык не менялся вовсе: 108.6 Гц во
# всех шести сценах до четвёртой значащей цифры. Поэтому столб придвинут: при
# 5 мм и радиусе 1.2 он занимает около 27 градусов, то есть примерно пять
# колонок. Два положения с каждой стороны различаются по азимуту на 18
# градусов, а левое от правого — на 54.
SCENES = {
    "без столба": (0.0, 500.0),
    "слева 27°": (5.0, 2.5),
    "слева 45°": (4.0, 4.0),
    "справа 27°": (5.0, -2.5),
    "справа 45°": (4.0, -4.0),
    "впереди": (5.0, 0.0),
}


def eye_sequence(px, py, seed=0, n_frames=FLYVIS_FRAMES, walk=True):
    """Последовательность показаний глаз, снятая пока муха идёт вперёд.

    Нужна потому, что T4 и T5 — детекторы движения: в дымовой проверке
    неподвижная полоса давала им ровно 0.000, а движущаяся 0.06-0.08. При
    статичной картинке половина стыка молчит по определению, и пространственную
    разницу несут только устойчивые типы. Живая муха тоже видит объект прежде
    всего потому, что он смещается, пока она идёт.
    """
    fly = make_locomotion_fly()
    fly.add_vision()
    world = PillarWorld(px, py)
    world.add_fly(fly, spawn_position=[0.0, 0.0, 0.5],
                  spawn_rotation=Rotation3D("quat", [1, 0, 0, 0]),
                  add_ground_contact_sensors=True)
    sim = Simulation(world)
    sim.reset()
    ctl = HybridTurningController(timestep=sim.timestep,
                                  preprogrammed_steps=PreprogrammedSteps())
    ctl.reset(seed=seed)
    sim.warmup(0.05)
    cmd = np.array([0.6, 0.6]) if walk else np.array([0.0, 0.0])
    frames = []
    steps_per_frame = max(int(round(FLYVIS_DT / sim.timestep)), 1)
    for _ in range(WARM_STEPS):
        obs = HybridControllerObservation.from_sim(sim, fly.name)
        apply_locomotion_action(sim, fly.name, ctl.step(cmd, obs))
        sim.step()
    for _ in range(n_frames):
        for _ in range(steps_per_frame):
            obs = HybridControllerObservation.from_sim(sim, fly.name)
            apply_locomotion_action(sim, fly.name, ctl.step(cmd, obs))
            sim.step()
        frames.append(sim.get_ommatidia_readouts(fly.name).sum(axis=2))
    return np.stack(frames)                                  # (кадры, 2, 721)


def eye_readouts(px, py, seed=0):
    """Показания 721 омматидия каждого глаза в заданной сцене."""
    fly = make_locomotion_fly()
    fly.add_vision()
    world = PillarWorld(px, py)
    world.add_fly(fly, spawn_position=[0.0, 0.0, 0.5],
                  spawn_rotation=Rotation3D("quat", [1, 0, 0, 0]),
                  add_ground_contact_sensors=True)
    sim = Simulation(world)
    sim.reset()
    ctl = HybridTurningController(timestep=sim.timestep,
                                  preprogrammed_steps=PreprogrammedSteps())
    ctl.reset(seed=seed)
    sim.warmup(0.05)
    cmd = np.array([0.0, 0.0])
    for _ in range(WARM_STEPS):
        obs = HybridControllerObservation.from_sim(sim, fly.name)
        apply_locomotion_action(sim, fly.name, ctl.step(cmd, obs))
        sim.step()
    return sim.get_ommatidia_readouts(fly.name).sum(axis=2)      # (2, 721)


def main():
    print("=" * 78)
    print(" РАЗЛИЧАЕТ ЛИ МУХА УЧАСТКИ ПОЛЯ ЗРЕНИЯ")
    print("=" * 78)

    align = np.load(out("hex_lattice_align.npz"))
    omm2col = align["ommatidium_to_column"]
    print(f"\nсопоставление решёток загружено: {len(omm2col)} омматидиев")

    asg = pd.read_csv(out("column_assignment.csv"))
    print(f"разложение колонок: {len(asg)} нейронов, "
          f"{asg['cell_type'].nunique()} типов")

    nv = flyvis.NetworkView(FLYVIS_MODEL)
    net = nv.init_network()
    fv_types = np.array([t.decode() if isinstance(t, bytes) else str(t)
                         for t in net.connectome.nodes.type[:]])
    fv_u = np.asarray(net.connectome.nodes.u[:])
    fv_v = np.asarray(net.connectome.nodes.v[:])
    from flyvis.utils.hex_utils import get_hex_coords
    hu, hv = get_hex_coords(15)
    col_of_uv = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(hu, hv))}
    fv_col = np.array([col_of_uv.get((int(a), int(b)), -1)
                       for a, b in zip(fv_u, fv_v)])

    # индекс узла flyvis по (тип, колонка)
    node_of = {}
    for i, (t, c) in enumerate(zip(fv_types, fv_col)):
        if c >= 0:
            node_of[(t, int(c))] = i

    comp = pd.read_csv(path_comp, index_col=0)
    flyid2i = {int(j): i for i, j in enumerate(comp.index)}
    n = len(flyid2i)
    ann = pd.read_csv(ANNOTATIONS, sep="\t", low_memory=False)
    ann["root_id"] = pd.to_numeric(ann["root_id"], errors="coerce")
    ann = ann.dropna(subset=["root_id"])
    ann["root_id"] = ann["root_id"].astype("int64")
    ann = ann[ann["root_id"].isin(flyid2i.keys())]

    # какие нейроны FlyWire получают активность и от какого узла flyvis
    asg = asg[[(t, int(c)) in node_of for t, c in
               zip(asg["cell_type"], asg["column"])]]
    fw_idx = np.array([flyid2i[int(r)] for r in asg["root_id"]])
    fv_idx = np.array([node_of[(t, int(c))] for t, c in
                       zip(asg["cell_type"], asg["column"])])
    print(f"стык: {len(fw_idx)} нейронов FlyWire получают вход от flyvis")

    vp = ann["super_class"] == "visual_projection"
    lc_l = np.array([flyid2i[int(x)] for x in ann.loc[vp & (ann["side"] == "left"), "root_id"]])
    lc_r = np.array([flyid2i[int(x)] for x in ann.loc[vp & (ann["side"] == "right"), "root_id"]])
    print(f"зрительных проекционных: слева {len(lc_l)}, справа {len(lc_r)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\nзагружаю веса FlyWire...")
    weights = rp.get_weights(str(path_con), str(path_comp), str(path_wt),
                             csr=True).to(device)

    def flyvis_activity(omm):
        """Показания глаз -> активность узлов flyvis, отдельно на глаз.

        Принимает либо один кадр (2, 721), либо последовательность
        (кадры, 2, 721). Последовательность нужна для T4 и T5: они детекторы
        движения и на неподвижной картинке дают ровно ноль.
        """
        seq = omm[None] if omm.ndim == 2 else omm
        if seq.shape[0] == 1:
            seq = np.tile(seq, (FLYVIS_FRAMES, 1, 1))
        acts = []
        for eye in range(2):
            col_vals = np.zeros((seq.shape[0], 721), dtype=np.float32)
            col_vals[:, omm2col] = seq[:, eye]
            x = torch.tensor(col_vals[None, :, None, :], device=flyvis.device)
            with torch.no_grad():
                st = net.simulate(x, dt=FLYVIS_DT)
            # среднее по последней трети: установившийся отклик без транзиента
            acts.append(st[0, -max(seq.shape[0] // 3, 1):].mean(0).cpu().numpy())
        return acts

    def lc_response(rate_vec):
        model = rp.TorchModel(1, n, DT_BRAIN_MS, rp.MODEL_PARAMS, weights,
                              exc_indices=fw_idx.tolist(), device=device)
        c, d, s, v, r = model.state_init()
        rates = torch.zeros(1, n, device=device)
        rates[0, torch.tensor(fw_idx, device=device)] = torch.tensor(
            rate_vec, dtype=torch.float32, device=device)
        g = torch.Generator(device=device)
        g.manual_seed(4242)
        # Копим ДВЕ половины окна отдельно. Это даёт бесплатную оценку
        # воспроизводимости: две половины одного прогона — независимые оценки
        # одной и той же сцены, и корреляция между ними задаёт уровень шума, с
        # которым надо сравнивать разницу между сценами. Без такого порога
        # число вроде «+0.06 у двух левых» прочесть нельзя: оно одинаково
        # согласуется и с «различает», и с «утонуло в шуме счёта».
        halves = [torch.zeros(len(lc_l) + len(lc_r), device=device)
                  for _ in range(2)]
        tl = torch.tensor(lc_l, dtype=torch.long, device=device)
        tr = torch.tensor(lc_r, dtype=torch.long, device=device)
        idx_all = torch.cat([tl, tr])
        n_tr, n_me = int(TRANSIENT_MS / DT_BRAIN_MS), int(BRAIN_MS / DT_BRAIN_MS)
        with torch.no_grad():
            for step in range(n_tr + n_me):
                c, d, s, v, r = model(rates, c, d, s, v, r, generator=g)
                if step >= n_tr:
                    halves[0 if step - n_tr < n_me // 2 else 1].add_(s[0, idx_all])
        t = BRAIN_MS / 2000.0
        h0 = halves[0].cpu().numpy() / t
        h1 = halves[1].cpu().numpy() / t
        return h0, h1

    # ---------- калибровка перевода активности в герцы ----------
    print("\nконтрольная сцена: считаю нормировку активности в герцы...")
    t0 = time.perf_counter()
    grab = eye_sequence if MOVING else (lambda x, y: eye_readouts(x, y))
    print("  режим съёмки глаз: "
          + ("последовательность на ходу" if MOVING else "один кадр стоя"))
    omm0 = grab(*SCENES["без столба"])
    a0 = flyvis_activity(omm0)
    seam_vals = np.concatenate([a[fv_idx] for a in a0])
    rest = float(np.percentile(seam_vals, 10))
    span = float(np.percentile(seam_vals, 95) - rest)
    print(f"  покой {rest:.4f}, размах до 95-го процентиля {span:.4f} "
          f"[{time.perf_counter() - t0:.0f} с]")

    def to_hz(a_left, a_right, sides):
        v = np.where(sides == "left", a_left[fv_idx], a_right[fv_idx])
        return np.clip((v - rest) / max(span, 1e-9) * MAX_HZ, 0.0, MAX_HZ)

    sides = asg["side"].to_numpy()

    # ---------- сцены ----------
    results = {}
    for name, (px, py) in SCENES.items():
        t0 = time.perf_counter()
        omm = omm0 if name == "без столба" else grab(px, py)
        a = a0 if name == "без столба" else flyvis_activity(omm)
        hz = to_hz(a[0], a[1], sides)
        h0, h1 = lc_response(hz)
        lc = (h0 + h1) / 2
        bright = omm.reshape(-1, 2, 721).mean(axis=(0, 2))
        results[name] = {"hz": hz, "lc": lc, "h0": h0, "h1": h1, "eye": bright}
        print(f"  {name:<14s} яркость {bright[0]:.4f}/"
              f"{bright[1]:.4f}; вход {hz.mean():>6.1f} Гц; "
              f"LC активных {(lc > 0).sum():>5d}; "
              f"средняя {lc.mean():>6.2f} Гц "
              f"[{time.perf_counter() - t0:.0f} с]")

    # ---------- сравнение наборов ----------
    print("\n" + "=" * 78)
    print(" СРАВНЕНИЕ НАБОРОВ АКТИВНЫХ ЗРИТЕЛЬНЫХ ПРОЕКЦИОННЫХ")
    print("=" * 78)

    # Сырая корреляция векторов частот тут негодна и это выяснилось замером:
    # первая версия дала 0.999 во ВСЕХ сравнениях. Причина не в конвейере, а в
    # мере: почти вся дисперсия по 8038 нейронам — это «кто вообще способен
    # разряжаться», и она одна и та же во всех сценах. Сценозависимая часть
    # тонет. Поэтому из каждого вектора вычитается среднее ПО СЦЕНАМ, и
    # сравнивается то, что осталось.
    names = list(SCENES)
    stack = np.stack([results[k]["lc"] for k in names])
    common = stack.mean(0)
    for k in names:
        results[k]["dev"] = results[k]["lc"] - common
        results[k]["dev0"] = results[k]["h0"] - common
        results[k]["dev1"] = results[k]["h1"] - common
    print(f"\n  доля дисперсии, общая для всех сцен: "
          f"{common.var() / max(stack.var(), 1e-12):.1%}")

    print("\n  различается ли сам ВХОД между сценами:")
    print(f"  {'сцена':<14s} {'вход, Гц':>9s} {'отличие от контроля, Гц':>25s} "
          f"{'нейронов изменилось':>20s}")
    hz0 = results["без столба"]["hz"]
    for k in names:
        h = results[k]["hz"]
        print(f"  {k:<14s} {h.mean():>9.1f} {np.abs(h - hz0).mean():>25.2f} "
              f"{int((np.abs(h - hz0) > 1.0).sum()):>20d}")

    def jaccard(a, b):
        A, B = a > 0, b > 0
        u = (A | B).sum()
        return float((A & B).sum() / u) if u else float("nan")

    def corr(a, b):
        if a.std() < 1e-12 or b.std() < 1e-12:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    pairs = [("слева 27°", "слева 45°", "две левых"),
             ("справа 27°", "справа 45°", "две правых"),
             ("слева 27°", "справа 27°", "лево против право"),
             ("слева 45°", "справа 45°", "лево против право"),
             ("слева 27°", "впереди", "лево против перед"),
             ("справа 27°", "впереди", "право против перед")]
    print(f"\n  {'сравнение':<22s} {'сцены':<32s} {'совпадение':>11s} "
          f"{'сырая корр.':>12s} {'по отклонению':>14s}")
    rows = []
    for a, b, lab in pairs:
        j = jaccard(results[a]["lc"], results[b]["lc"])
        c = corr(results[a]["lc"], results[b]["lc"])
        cd = corr(results[a]["dev"], results[b]["dev"])
        print(f"  {lab:<22s} {a + ' / ' + b:<32s} {j:>11.3f} {c:>12.3f} "
              f"{cd:>14.3f}")
        rows.append({"label": lab, "scene_a": a, "scene_b": b,
                     "jaccard": j, "corr": c, "corr_dev": cd})

    df = pd.DataFrame(rows)
    df.to_csv(out("hybrid_vision_probe.csv"), index=False)
    np.savez(out("hybrid_vision_lc.npz"),
             **{k: v["lc"] for k, v in results.items()})
    print(f"\nсохранено: {out('hybrid_vision_probe.csv')}")

    print("\n" + "=" * 78)
    print(" КРИТЕРИЙ ПРИЁМКИ")
    print("=" * 78)
    rel = float(np.mean([corr(results[k]["dev0"], results[k]["dev1"])
                         for k in names]))
    same = df[df["label"].isin(["две левых", "две правых"])]["corr_dev"].mean()
    diff = df[df["label"] == "лево против право"]["corr_dev"].mean()
    print(f"  ВОСПРОИЗВОДИМОСТЬ сцены самой собой (половины окна): {rel:.3f}")
    print(f"  корреляция внутри одной стороны: {same:.3f}")
    print(f"  корреляция между сторонами:      {diff:.3f}")
    print(f"  разница: {same - diff:+.3f}")
    ok = same - diff > 0.05
    print(f"\n  сторона объекта закодирована: {'да' if ok else 'НЕТ'}")
    print("\n  Про положение ВНУТРИ стороны читать так. Если воспроизводимость")
    print("  много выше корреляции двух сцен одной стороны, значит эти две сцены")
    print("  и правда разные для мухи — то есть различается не сторона, а место.")
    print("  Если воспроизводимость сама около нуля, мерить нечем: всё, что")
    print("  ниже неё, неотличимо от шума счёта спайков.")
    within = "различает место внутри стороны" if (rel > 0.3 and same < rel - 0.2) \
        else ("шум, мерить нечем" if rel < 0.3 else "не различает")
    print(f"\n  вывод про место внутри стороны: {within}")


if __name__ == "__main__":
    main()
