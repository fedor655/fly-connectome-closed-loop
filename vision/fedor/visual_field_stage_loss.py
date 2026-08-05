"""На каком шаге пути теряется вторая ось поля зрения?

Что уже известно. Карта надёжна (половины связей сходятся на 0.993 и 0.965
слева), эталон цел (типы одних колонок дают совпадающие листы сомы, 0.999 и
0.998), широкопольные нейроны ни при чём (исключение верхушки по числу
партнёров ничего не даёт, проверено до порога 200). При этом сверка карты с
полными тремя координатами сомы даёт по ступеням:

  ламина L1 / L2      слева 0.741 / 0.847   справа 0.485 / 0.605
  медулла Mi1 / Tm1   слева 0.538 / 0.560   справа 0.911 / 0.904
  детекторы T4a / T4c слева 0.177 / 0.119   справа 0.569 / 0.591

Стороны ведут себя ПРОТИВОПОЛОЖНО: слева вглубь хуже, справа вглубь лучше. Это
подозрительно само по себе и намекает, что беда не в карте и не в проводке, а в
том, как соотносятся положение сомы и положение колонки у разных типов.

Проверка без эталона в самой сверке. Типы, сидящие в одних колонках, обязаны
получать одинаковые позиции в карте. Пары нейронов составляются по близости
сомы — это законно, потому что у типов одной колонки сомы соседние, и это
измерено (0.999 и 0.998). Но СРАВНИВАЮТСЯ потом карты, а не координаты. Если
карта Mi1 и карта T4c согласны, значит вторая ось шаг Mi1 → T4 переживает, и
низкая сверка T4 с сомой означает, что негодна сома T4, а не карта. Если не
согласны — ось теряется именно там.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from flypaths import ANNOTATIONS, add_fly_brain_to_path, out  # noqa: E402

add_fly_brain_to_path()
from benchmark import path_comp  # noqa: E402

VOXEL_NM = np.array([4.0, 4.0, 40.0])
CHAIN = [("L1", "Mi1"), ("Mi1", "Mi4"), ("Mi1", "Tm1"), ("Mi1", "T4a"),
         ("Mi1", "T4c"), ("T4a", "T4c"), ("Tm1", "T5c"), ("T4c", "T5c"),
         ("L1", "L2"), ("L2", "Mi1")]


def cca(A, B):
    def white(X):
        X = X - X.mean(0)
        q, r_ = np.linalg.qr(X)
        return q[:, :np.linalg.matrix_rank(r_)]
    qa, qb = white(A), white(B)
    if qa.shape[1] < 2 or qb.shape[1] < 2:
        return float("nan"), float("nan")
    s = np.clip(np.linalg.svd(qa.T @ qb, compute_uv=False), 0, 1)
    return float(s[0]), float(s[1])


def main():
    print("=" * 78)
    print(" ГДЕ ТЕРЯЕТСЯ ВТОРАЯ ОСЬ: СРАВНЕНИЕ КАРТЫ С КАРТОЙ")
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

    def grab(t, side):
        s = m[(ct == t) & (m["side"] == side)]
        if len(s) < 50:
            return None
        return (s[["pos_x", "pos_y", "pos_z"]].to_numpy(float) * VOXEL_NM,
                s[["azimuth_um", "elevation_um"]].to_numpy(float))

    print("\n  Пары составлены по близости сомы, сравниваются КАРТЫ.")
    print(f"\n  {'пара':<12s} {'сторона':>7s} {'карта против карты':>21s} "
          f"{'разлёт сомы, мкм':>18s}")
    rows = []
    for a, b in CHAIN:
        for side in ("left", "right"):
            ga, gb = grab(a, side), grab(b, side)
            if ga is None or gb is None:
                continue
            PA, MA = ga
            PB, MB = gb
            d = ((PA[:, None, :] - PB[None, :, :]) ** 2).sum(-1)
            j = d.argmin(1)
            gap = np.sqrt(d[np.arange(len(PA)), j]) / 1000.0
            c1, c2 = cca(MA, MB[j])
            print(f"  {a + '/' + b:<12s} {side:>7s} {c1:>12.3f} {c2:>8.3f} "
                  f"{np.median(gap):>18.1f}")
            rows.append({"pair": f"{a}/{b}", "side": side,
                         "map_cc1": c1, "map_cc2": c2,
                         "soma_gap_um": float(np.median(gap))})

    df = pd.DataFrame(rows)
    df.to_csv(out("visual_field_stage_loss.csv"), index=False)
    print(f"\nсохранено: {out('visual_field_stage_loss.csv')}")

    print("\n" + "=" * 78)
    print(" ЧТЕНИЕ")
    print("=" * 78)
    for side in ("left", "right"):
        d = df[df["side"] == side]
        print(f"  {side}: карта с картой в среднем {d['map_cc1'].mean():.3f} и "
              f"{d['map_cc2'].mean():.3f}")
    print()
    print("  Если карта с картой согласуется хорошо всюду, а с сомой — только у")
    print("  части типов, то теряется не ось, а годность СОМЫ как эталона для")
    print("  этих типов. Тогда чинить надо проверку, а не карту.")
    print()
    print("  Столбец разлёта сомы — медиана расстояния до сопоставленного нейрона")
    print("  другого типа. Если он много больше шага колонки (около 5 мкм),")
    print("  сопоставление по соме ненадёжно и всю строку читать нельзя.")


if __name__ == "__main__":
    main()
