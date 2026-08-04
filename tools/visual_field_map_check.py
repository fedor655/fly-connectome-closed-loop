"""Карта поля зрения по связям двумерна или всё-таки одномерна?

Зачем. Согласие карты с листом координат сомы дало канонические корреляции
0.985 и 0.425 при 0.073 и 0.018 на перемешанных связях. Первая ось поля зрения
восстановлена, со второй непонятно, и трактовок ровно две:

  A. Карта вырождена, она почти одномерная, и второй оси в ней нет.
  B. Карта двумерна, а негоден ЭТАЛОН: у листа сомы мы сами измерили дрожание
     около половины шага решётки, и его вторая ось может быть шумом.

Разница решающая. При A гибрид даст мухе только одну ось поля зрения, при B —
обе, и низкая вторая корреляция окажется свойством проверки, а не карты.

Две проверки, ни одна не опирается на координаты сомы:

  1. Собственная двумерность карты. Доли дисперсии самой карты по её главным
     осям. Если вторая доля мала, верна трактовка A и спорить не о чем.

  2. Согласие между типами, тайлящими ОДНИ И ТЕ ЖЕ колонки. Mi1, Tm1, T4a и
     T4c сидят в одних колонках медуллы, значит их облака в карте обязаны
     совпасть, а ближайшие соседи из разных типов — оказаться одной колонкой.
     Меряем расстояние до ближайшего соседа другого типа в единицах среднего
     расстояния между соседями внутри типа. Контроль — перемешать позиции
     одного из типов: тогда отношение обязано вырасти до случайного.

Для сравнения печатается и двумерность самого листа сомы: если у эталона
вторая доля мала, это прямо подтверждает трактовку B.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flypaths import ANNOTATIONS, add_fly_brain_to_path, out  # noqa: E402

add_fly_brain_to_path()
from benchmark import path_comp  # noqa: E402

VOXEL_NM = np.array([4.0, 4.0, 40.0])
TYPES = ["L1", "L2", "L3", "Mi1", "Mi4", "Mi9", "Tm1", "Tm2", "Tm9",
         "T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]
PAIRS = [("Mi1", "Tm1"), ("Mi1", "T4c"), ("T4a", "T4c"), ("T5a", "T5c"),
         ("Mi1", "L1"), ("Tm9", "T5c")]


def var_fracs(xy):
    c = xy - xy.mean(0)
    s = np.linalg.svd(c, compute_uv=False)
    v = s ** 2 / (s ** 2).sum()
    return float(v[0]), float(v[1])


def nn_ratio(A, B):
    """Медиана расстояния от точек A до ближайшей точки B, в единицах
    медианного расстояния между соседями внутри A."""
    dA = np.sqrt(((A[:, None, :] - A[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(dA, np.inf)
    inner = np.median(dA.min(1))
    dAB = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))
    return float(np.median(dAB.min(1)) / max(inner, 1e-9))


def main():
    print("=" * 78)
    print(" КАРТА ПОЛЯ ЗРЕНИЯ: двумерна ли она на самом деле")
    print("=" * 78)

    vf = pd.read_csv(out("visual_field_map.csv"))
    comp = pd.read_csv(path_comp, index_col=0)
    our = set(int(x) for x in comp.index)
    ann = pd.read_csv(ANNOTATIONS, sep="\t", low_memory=False)
    ann["root_id"] = pd.to_numeric(ann["root_id"], errors="coerce")
    ann = ann.dropna(subset=["root_id"])
    ann["root_id"] = ann["root_id"].astype("int64")
    ann = ann[ann["root_id"].isin(our)]
    ann = ann[ann[["pos_x", "pos_y", "pos_z"]].notna().all(axis=1)]
    m = ann.merge(vf[["root_id", "azimuth_um", "elevation_um"]], on="root_id")
    ct = m["cell_type"].fillna("")
    print(f"нейронов с картой и координатами: {len(m)}")

    # ---------- 1. собственная двумерность ----------
    print("\n----- доли дисперсии: карта против листа сомы -----")
    print(f"  {'тип':<6s} {'сторона':>7s} {'n':>5s} "
          f"{'карта: 1-я / 2-я':>20s} {'лист сомы: 1-я / 2-я':>24s}")
    rows = []
    for t in TYPES:
        for side in ("left", "right"):
            sub = m[(ct == t) & (m["side"] == side)]
            if len(sub) < 50:
                continue
            xy = sub[["azimuth_um", "elevation_um"]].to_numpy(float)
            v1, v2 = var_fracs(xy)
            p = sub[["pos_x", "pos_y", "pos_z"]].to_numpy(float) * VOXEL_NM
            c = p - p.mean(0)
            s = np.linalg.svd(c, compute_uv=False) ** 2
            s = s / s.sum()
            print(f"  {t:<6s} {side:>7s} {len(sub):>5d} "
                  f"{v1:>10.3f} / {v2:<7.3f} {s[0]:>14.3f} / {s[1]:<7.3f}")
            rows.append({"cell_type": t, "side": side, "n": len(sub),
                         "map_var1": v1, "map_var2": v2,
                         "soma_var1": float(s[0]), "soma_var2": float(s[1])})
    df = pd.DataFrame(rows)

    # ---------- 2. согласие между типами одних колонок ----------
    print("\n----- совпадают ли типы, сидящие в одних колонках -----")
    print("  чем ближе к 1.0, тем точнее чужой тип попадает в ту же колонку;")
    print("  контроль — позиции второго типа перемешаны между его же нейронами")
    print(f"\n  {'пара':<14s} {'сторона':>7s} {'отношение':>11s} {'перемешано':>12s}")
    rng = np.random.default_rng(11)
    prows = []
    for a, b in PAIRS:
        for side in ("left", "right"):
            A = m[(ct == a) & (m["side"] == side)][["azimuth_um", "elevation_um"]].to_numpy(float)
            B = m[(ct == b) & (m["side"] == side)][["azimuth_um", "elevation_um"]].to_numpy(float)
            if len(A) < 50 or len(B) < 50:
                continue
            real = nn_ratio(A, B)
            # контроль: координаты перемешиваются НЕЗАВИСИМО по осям, поэтому
            # облако сохраняет размер и форму по каждой оси, но теряет то, где
            # именно сидит конкретный нейрон
            Bs = np.stack([rng.permutation(B[:, 0]), rng.permutation(B[:, 1])], 1)
            shuf = nn_ratio(A, Bs)
            print(f"  {a + '/' + b:<14s} {side:>7s} {real:>11.2f} {shuf:>12.2f}")
            prows.append({"pair": f"{a}/{b}", "side": side,
                          "nn_ratio": real, "nn_ratio_shuffled": shuf})

    df.to_csv(out("visual_field_map_check.csv"), index=False)
    pd.DataFrame(prows).to_csv(out("visual_field_map_pairs.csv"), index=False)

    print("\n" + "=" * 78)
    print(" ВЫВОД")
    print("=" * 78)
    if not df.empty:
        print(f"  вторая доля дисперсии карты, среднее:      {df['map_var2'].mean():.3f}")
        print(f"  вторая доля дисперсии листа сомы, среднее: {df['soma_var2'].mean():.3f}")
    if prows:
        pr = pd.DataFrame(prows)
        print(f"  попадание в чужую колонку:  {pr['nn_ratio'].mean():.2f}, "
              f"перемешано {pr['nn_ratio_shuffled'].mean():.2f}")
    print()
    print("  Трактовка A (карта одномерна) верна, если вторая доля дисперсии карты")
    print("  заметно меньше второй доли листа сомы. Трактовка B (негоден эталон)")
    print("  верна, если наоборот, и при этом типы одних колонок ложатся друг на")
    print("  друга заметно точнее, чем перемешанный контроль.")


if __name__ == "__main__":
    main()
