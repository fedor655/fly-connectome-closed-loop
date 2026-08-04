"""Стало ли деталей больше: два теста, которые старая схема проваливает по построению.

Что уже доказано (шаг 1.3): сторона объекта кодируется, и две сцены одной
стороны дают разные картины. Чего не доказано: что муха получила именно
ПРОСТРАНСТВЕННОЕ разрешение, а не просто более чувствительную яркость.

Здесь два опыта, устроенных так, что старая схема — одно число на глаз — не
может пройти их в принципе, а не «плохо проходит».

  Тест «один толстый против двух тонких». Два тонких столба по бокам и один
  толстый впереди подбираются так, чтобы СУММАРНОЕ затемнение глаз совпало.
  Для одного числа на глаз это буквально одна и та же цифра. Для
  пространственного зрения — совершенно разные картины.

  Тест «выше или ниже горизонта». Компактный объект на одном азимуте, но над
  линией взгляда и под ней. Суммарная яркость та же, глаз тот же, отличается
  только высота. Одно число на глаз тут беспомощно вдвойне: у него нет ни
  азимута, ни высоты.

Контроль обязателен и он самый важный в этой работе: КАЖДАЯ сцена прогоняется
дважды, гибридом и старой схемой. У старой схемы все нейроны стыка одного глаза
получают ОДИНАКОВУЮ частоту, равную средней по гибриду для этого глаза. То есть
суммарный вход в мозг тот же, а пространственная структура снята. Всё, что
различается между двумя прогонами, — это заслуга пространства и ничего больше.

Потолок надёжности меряется как и раньше: окно делится пополам, две половины
дают независимые оценки одной сцены, поправка Спирмена-Брауна приводит к длине
целого окна. Выше потолка не бывает ничего.
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

DT_BRAIN_MS = 0.1
BRAIN_MS = 2000.0
TRANSIENT_MS = 200.0
FLYVIS_MODEL = "flow/0000/000"
FLYVIS_FRAMES = 40
FLYVIS_DT = 1 / 100.0
MAX_HZ = 200.0
WARM_STEPS = 40
# Независимых прогонов в одном проходе; половина против половины даёт потолок.
# Шесть один раз уронили счёт с CUDA error: unknown error на четвёртой сцене.
# Видеокарта в этой машине обслуживает и рабочий стол Windows — из 8 ГБ ей
# самой остаётся около полутора, и шесть прогонов ложатся впритык.
BATCH = 4
DARK = [0.05, 0.05, 0.05, 1.0]


class ObjectWorld(FlatGroundWorld):
    """Плоский грунт и произвольный набор тёмных тел."""

    def __init__(self, objects):
        super().__init__(name="object_world", half_size=300)
        for i, o in enumerate(objects):
            if o["kind"] == "cylinder":
                self.mjcf_root.worldbody.add_geom(
                    type=GEOM_TYPES["cylinder"], name=f"obj{i}",
                    size=[o["r"], o["h"] / 2, 0.0],
                    pos=[o["x"], o["y"], o["h"] / 2],
                    rgba=DARK, contype=0, conaffinity=0)
            else:
                self.mjcf_root.worldbody.add_geom(
                    type=GEOM_TYPES["sphere"], name=f"obj{i}",
                    size=[o["r"], 0.0, 0.0],
                    pos=[o["x"], o["y"], o["z"]],
                    rgba=DARK, contype=0, conaffinity=0)


def cyl(x, y, r=1.2, h=8.0):
    return {"kind": "cylinder", "x": x, "y": y, "r": r, "h": h}


def ball(x, y, z, r=1.0):
    return {"kind": "sphere", "x": x, "y": y, "z": z, "r": r}


def eyes(objects, seed=0):
    """Показания 721 омматидия каждого глаза."""
    fly = make_locomotion_fly()
    fly.add_vision()
    world = ObjectWorld(objects)
    world.add_fly(fly, spawn_position=[0.0, 0.0, 0.5],
                  spawn_rotation=Rotation3D("quat", [1, 0, 0, 0]),
                  add_ground_contact_sensors=True)
    sim = Simulation(world)
    sim.reset()
    ctl = HybridTurningController(timestep=sim.timestep,
                                  preprogrammed_steps=PreprogrammedSteps())
    ctl.reset(seed=seed)
    sim.warmup(0.05)
    for _ in range(WARM_STEPS):
        obs = HybridControllerObservation.from_sim(sim, fly.name)
        apply_locomotion_action(sim, fly.name, ctl.step(np.zeros(2), obs))
        sim.step()
    return sim.get_ommatidia_readouts(fly.name).sum(axis=2)


def cca_corr(a, b):
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    print("=" * 78)
    print(" СТАЛО ЛИ ДЕТАЛЕЙ БОЛЬШЕ: ДВА РЕШАЮЩИХ ТЕСТА")
    print("=" * 78)

    align = np.load(out("hex_lattice_align.npz"))
    omm2col = align["ommatidium_to_column"]
    asg = pd.read_csv(out("column_assignment.csv"))

    # ---------- 1. подбор сцен по равенству суммарного затемнения ----------
    print("\n----- подбираю толстый столб под два тонких -----")
    two_thin = [cyl(4.0, 3.4, r=0.7), cyl(4.0, -3.4, r=0.7)]
    e_two = eyes(two_thin)
    target = e_two.mean()
    print(f"  два тонких по ±40°: яркость {e_two.mean(axis=1)[0]:.4f}/"
          f"{e_two.mean(axis=1)[1]:.4f}, среднее {target:.4f}")
    # Первый заход брал сетку от 0.8 и уравнял яркость лишь до 4 процентов.
    # Этого хватило старой схеме: её корреляция -0.36 превысила её же потолок
    # 0.30, то есть она различила сцены по одной яркости, и тест стал
    # недействительным. Сетка сдвинута вниз и сгущена.
    best = None
    for r in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80):
        e = eyes([cyl(5.0, 0.0, r=r)])
        d = abs(e.mean() - target)
        print(f"  толстый впереди r={r:.1f}: яркость {e.mean(axis=1)[0]:.4f}/"
              f"{e.mean(axis=1)[1]:.4f}, среднее {e.mean():.4f}, "
              f"расхождение {d:.4f}")
        if best is None or d < best[0]:
            best = (d, r, e)
    print(f"  выбран r={best[1]:.1f}, расхождение по яркости {best[0]:.4f} "
          f"({best[0] / target:.1%})")
    e_one = best[2]

    print("\n----- объект выше и ниже линии взгляда -----")
    e_up = eyes([ball(5.0, 0.0, 4.0, r=1.0)])
    e_dn = eyes([ball(5.0, 0.0, 0.6, r=1.0)])
    print(f"  выше: яркость {e_up.mean(axis=1)[0]:.4f}/{e_up.mean(axis=1)[1]:.4f}, "
          f"среднее {e_up.mean():.4f}")
    print(f"  ниже: яркость {e_dn.mean(axis=1)[0]:.4f}/{e_dn.mean(axis=1)[1]:.4f}, "
          f"среднее {e_dn.mean():.4f}")
    print(f"  расхождение по яркости {abs(e_up.mean() - e_dn.mean()) / e_up.mean():.1%}")

    e_ctl = eyes([cyl(0.0, 500.0)])
    scenes = {"пусто": e_ctl, "два тонких": e_two, "один толстый": e_one,
              "выше горизонта": e_up, "ниже горизонта": e_dn}

    # ---------- 2. flyvis на все сцены разом, потом освободить ----------
    print("\n----- считаю оптические доли для всех сцен -----")
    nv = flyvis.NetworkView(FLYVIS_MODEL)
    net = nv.init_network()
    fv_types = np.array([t.decode() if isinstance(t, bytes) else str(t)
                         for t in net.connectome.nodes.type[:]])
    from flyvis.utils.hex_utils import get_hex_coords
    hu, hv = get_hex_coords(15)
    col_of_uv = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(hu, hv))}
    fv_col = np.array([col_of_uv.get((int(a), int(b)), -1)
                       for a, b in zip(np.asarray(net.connectome.nodes.u[:]),
                                       np.asarray(net.connectome.nodes.v[:]))])
    node_of = {(t, int(c)): i for i, (t, c) in enumerate(zip(fv_types, fv_col))
               if c >= 0}
    asg = asg[[(t, int(c)) in node_of for t, c in
               zip(asg["cell_type"], asg["column"])]]
    fv_idx = np.array([node_of[(t, int(c))] for t, c in
                       zip(asg["cell_type"], asg["column"])])
    sides = asg["side"].to_numpy()

    def activity(omm):
        acts = []
        for eye in range(2):
            cv = np.zeros(721, dtype=np.float32)
            cv[omm2col] = omm[eye]
            x = torch.tensor(np.tile(cv, (FLYVIS_FRAMES, 1))[None, :, None, :],
                             device=flyvis.device)
            with torch.no_grad():
                st = net.simulate(x, dt=FLYVIS_DT)
            acts.append(st[0, -1].cpu().numpy())
        return acts

    acts = {k: activity(v) for k, v in scenes.items()}
    seam0 = np.concatenate([a[fv_idx] for a in acts["пусто"]])
    rest = float(np.percentile(seam0, 10))
    span = float(np.percentile(seam0, 95) - rest)
    hz = {}
    for k, a in acts.items():
        v = np.where(sides == "left", a[0][fv_idx], a[1][fv_idx])
        hz[k] = np.clip((v - rest) / max(span, 1e-9) * MAX_HZ, 0.0, MAX_HZ)
    print(f"  покой {rest:.4f}, размах {span:.4f}")

    # старая схема: та же суммарная подача, но без пространственной структуры
    hz_flat = {}
    for k, h in hz.items():
        f = np.empty_like(h)
        for s in ("left", "right"):
            m = sides == s
            f[m] = h[m].mean()
        hz_flat[k] = f

    del net, nv
    torch.cuda.empty_cache()
    print("  flyvis выгружен из памяти видеокарты")

    # ---------- 3. мозг ----------
    comp = pd.read_csv(path_comp, index_col=0)
    flyid2i = {int(j): i for i, j in enumerate(comp.index)}
    n = len(flyid2i)
    ann = pd.read_csv(ANNOTATIONS, sep="\t", low_memory=False)
    ann["root_id"] = pd.to_numeric(ann["root_id"], errors="coerce")
    ann = ann.dropna(subset=["root_id"])
    ann["root_id"] = ann["root_id"].astype("int64")
    ann = ann[ann["root_id"].isin(flyid2i.keys())]
    fw_idx = np.array([flyid2i[int(r)] for r in asg["root_id"]])
    vp = ann["super_class"] == "visual_projection"
    lc = np.array([flyid2i[int(x)] for x in ann.loc[vp, "root_id"]])
    print(f"\nстык {len(fw_idx)} нейронов, зрительных проекционных {len(lc)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("загружаю веса FlyWire...")
    weights = rp.get_weights(str(path_con), str(path_comp), str(path_wt),
                             csr=True).to(device)
    idx_lc = torch.tensor(lc, dtype=torch.long, device=device)
    idx_fw = torch.tensor(fw_idx, dtype=torch.long, device=device)

    def brain(rate_vec):
        """Пачка независимых прогонов разом.

        Первая версия делила ОДНО окно пополам и считала потолок надёжности по
        половинам. Вышло -0.068 у гибрида и -3.15 у старой схемы, а поправка
        Спирмена-Брауна на отрицательной надёжности выдала -4.5e8. Две беды
        сразу: сигнал в этих сценах и правда ниже шума, и в самой оценке была
        утечка — общее среднее считалось по обеим половинам, и половины через
        него связывались отрицательно.

        Здесь идут BATCH независимых прогонов в одном проходе: пуассоновский
        вход даёт свой шум каждому элементу пачки. Шум падает как корень из
        размера пачки, а потолок меряется двумя НЕЗАВИСИМЫМИ половинами пачки,
        каждая со своим общим средним.
        """
        model = rp.TorchModel(BATCH, n, DT_BRAIN_MS, rp.MODEL_PARAMS, weights,
                              exc_indices=fw_idx.tolist(), device=device)
        c, d, s, v, r = model.state_init()
        rates = torch.zeros(BATCH, n, device=device)
        rates[:, idx_fw] = torch.tensor(rate_vec, dtype=torch.float32,
                                        device=device)
        g = torch.Generator(device=device)
        g.manual_seed(4242)
        acc = torch.zeros(BATCH, len(lc), device=device)
        n_tr, n_me = int(TRANSIENT_MS / DT_BRAIN_MS), int(BRAIN_MS / DT_BRAIN_MS)
        with torch.no_grad():
            for step in range(n_tr + n_me):
                c, d, s, v, r = model(rates, c, d, s, v, r, generator=g)
                if step >= n_tr:
                    acc.add_(s[:, idx_lc])
        a = acc.cpu().numpy() / (BRAIN_MS / 1000.0)
        half = BATCH // 2
        return a[:half].mean(0), a[half:].mean(0)

    res = {}
    for scheme, table in (("гибрид", hz), ("старая схема", hz_flat)):
        print(f"\n----- {scheme} -----")
        for k in scenes:
            t0 = time.perf_counter()
            a, b = brain(table[k])
            res[(scheme, k)] = (a, b)
            print(f"  {k:<16s} вход {table[k].mean():>6.1f} Гц; "
                  f"LC средняя {((a + b) / 2).mean():>6.2f} Гц "
                  f"[{time.perf_counter() - t0:.0f} с]")

    # ---------- 4. разбор ----------
    print("\n" + "=" * 78)
    print(" РЕЗУЛЬТАТ")
    print("=" * 78)
    rows = []
    for scheme in ("гибрид", "старая схема"):
        names = list(scenes)
        # Каждая половина пачки обрабатывается отдельно, со своим общим средним:
        # так две оценки одной сцены остаются независимыми и потолок не уезжает
        # в минус из-за общего вычитаемого.
        dev = {}
        for h in (0, 1):
            com = np.mean([res[(scheme, k)][h] for k in names], axis=0)
            for k in names:
                dev[(k, h)] = res[(scheme, k)][h] - com
        rh = float(np.mean([cca_corr(dev[(k, 0)], dev[(k, 1)]) for k in names]))
        # надёжность половины пачки -> надёжность целой, поправка Спирмена-Брауна
        ceil = 2 * rh / (1 + rh) if rh > 0 else rh
        print(f"\n  {scheme}: надёжность половины {rh:.3f}, потолок целой {ceil:.3f}")
        if ceil <= 0.05:
            print("    ПОТОЛОК ОКОЛО НУЛЯ: сцены неразличимы даже сами с собой,")
            print("    мерить нечем. Всё ниже — не результат, а шум.")
        for a, b, lab in (("два тонких", "один толстый", "толстый против двух тонких"),
                          ("выше горизонта", "ниже горизонта", "выше против ниже")):
            c = 0.5 * (cca_corr(dev[(a, 0)], dev[(b, 1)]) +
                       cca_corr(dev[(a, 1)], dev[(b, 0)]))
            rel = c / ceil if ceil > 0.05 else float("nan")
            print(f"    {lab:<28s} корреляция {c:>7.3f}   доля потолка "
                  f"{rel:>6.2f}")
            rows.append({"scheme": scheme, "test": lab, "corr": c,
                         "ceiling": ceil, "corr_corrected": rel})

    df = pd.DataFrame(rows)
    df.to_csv(out("hybrid_vision_tests.csv"), index=False)
    print(f"\nсохранено: {out('hybrid_vision_tests.csv')}")

    print("\n" + "=" * 78)
    print(" КАК ЧИТАТЬ")
    print("=" * 78)
    print("  Доля потолка около 1.0 означает «сцены неотличимы»: они дали то же,")
    print("  что сцена даёт сама себе. Заметно ниже 1.0 — различимы. У старой")
    print("  схемы обе пары ОБЯЗАНЫ выйти около 1.0, иначе подача уравнена")
    print("  неверно и весь опыт недействителен.")
    print("  Если потолок около нуля, читать нечего: измерение не разрешает")
    print("  даже сцену от неё самой, и любые числа ниже — шум.")
    for lab in ("толстый против двух тонких", "выше против ниже"):
        h = df[(df["scheme"] == "гибрид") & (df["test"] == lab)]
        o = df[(df["scheme"] == "старая схема") & (df["test"] == lab)]
        if not len(h) or not len(o):
            continue
        hc, oc = float(h["corr"].iloc[0]), float(o["corr"].iloc[0])
        hp, op = float(h["ceiling"].iloc[0]), float(o["ceiling"].iloc[0])
        # Старая схема считается «увидевшей» сцены, только если её корреляция
        # выходит за её собственный потолок по модулю: иначе это её шум, а не
        # различение. Именно так и провалился первый заход теста про толстый
        # столб — там |-0.36| превысило потолок 0.30.
        old_sees = abs(oc) > op
        hyb_sees = hc < 0.5 * hp and hp > 0.5
        if hyb_sees and not old_sees:
            verdict = "ГИБРИД РАЗЛИЧАЕТ, СТАРАЯ НЕТ"
        elif old_sees:
            verdict = ("сцены не уравнены по яркости: старая схема реагирует "
                       "сильнее собственного потолка — тест недействителен")
        elif not hyb_sees:
            verdict = "не различает ни одна"
        else:
            verdict = "вывод неоднозначен"
        print(f"\n  {lab}: гибрид {hc:+.3f} при потолке {hp:.3f}, "
              f"старая {oc:+.3f} при потолке {op:.3f}")
        print(f"    -> {verdict}")


if __name__ == "__main__":
    main()
