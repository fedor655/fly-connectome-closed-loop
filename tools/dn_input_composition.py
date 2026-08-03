"""Что на самом деле слушает DNp09: состав его входов по типам клеток.

Зачем. Стимуляция всей популяции восходящих нейронов не вызывает у DNp09
отклика (tools/ascending_to_dn.py). Вместо того чтобы гадать, какую популяцию
пробовать следующей, посмотрим прямо: кто входит в DNp09 и к каким классам эти
нейроны принадлежат по аннотациям FlyWire.

Это переворачивает задачу. Раньше мы искали «чем бы постимулировать, чтобы
дошло», перебирая кандидатов. Теперь просто спрашиваем у данных, из чего
DNp09 собирает свой вход, и работаем с этим.

Считается по 755/607 прямым входам DNp09 с учётом весов, плюс второй уровень.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flypaths import ANNOTATIONS, add_fly_brain_to_path, out  # noqa: E402

add_fly_brain_to_path()
from benchmark import path_comp, path_con  # noqa: E402

# Стороны по аннотациям FlyWire
DNP09_LEFT = 720575940635872101
DNP09_RIGHT = 720575940627652358

TOP_SHOW = 12


def describe(ann_by_idx, indices, weights, title):
    """Разложить набор входов по классам аннотаций, взвешивая по силе связи."""
    print(f"\n----- {title} -----")
    rows = []
    for i, w in zip(indices, weights):
        meta = ann_by_idx.get(int(i))
        rows.append({
            "super_class": (meta or {}).get("super_class") or "(нет аннотации)",
            "cell_class": (meta or {}).get("cell_class") or "(нет)",
            "cell_type": (meta or {}).get("cell_type") or "(нет)",
            "weight": abs(float(w)),
            "signed": float(w),
        })
    df = pd.DataFrame(rows)
    total = df["weight"].sum()
    print(f"  входов: {len(df)}, суммарный |вес|: {total:.0f}")

    g = df.groupby("super_class").agg(
        n=("weight", "size"), w=("weight", "sum"), signed=("signed", "sum"))
    g["доля_%"] = 100 * g["w"] / max(total, 1)
    g = g.sort_values("w", ascending=False)
    print(f"  {'super_class':<22s} {'шт':>5s} {'|вес|':>9s} {'доля %':>8s} {'алгебр.':>9s}")
    for name, r in g.iterrows():
        print(f"  {name:<22s} {int(r['n']):>5d} {r['w']:>9.0f} "
              f"{r['доля_%']:>8.1f} {r['signed']:>+9.0f}")

    top = df.groupby("cell_type").agg(n=("weight", "size"), w=("weight", "sum"))
    top = top.sort_values("w", ascending=False).head(TOP_SHOW)
    print(f"\n  частые типы клеток на входе:")
    for name, r in top.iterrows():
        print(f"    {name:<24s} {int(r['n']):>4d} шт, |вес| {r['w']:>7.0f}")
    return df


def main():
    print("=" * 78)
    print(" СОСТАВ ВХОДОВ DNp09 ПО ТИПАМ КЛЕТОК")
    print("=" * 78)

    comp = pd.read_csv(path_comp, index_col=0)
    flyid2i = {int(j): i for i, j in enumerate(comp.index)}
    i2flyid = {i: j for j, i in flyid2i.items()}
    n = len(flyid2i)

    ann = pd.read_csv(ANNOTATIONS, sep="\t", low_memory=False)
    ann["root_id"] = pd.to_numeric(ann["root_id"], errors="coerce")
    ann = ann.dropna(subset=["root_id"])
    ann["root_id"] = ann["root_id"].astype("int64")
    ann_by_id = ann.set_index("root_id")[
        ["super_class", "cell_class", "cell_type", "side"]].to_dict("index")
    ann_by_idx = {flyid2i[k]: v for k, v in ann_by_id.items() if k in flyid2i}
    print(f"нейронов: {n}, из них с аннотацией: {len(ann_by_idx)}")

    conn = pd.read_parquet(path_con, columns=[
        "Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
    W = sp.coo_matrix(
        (conn["Excitatory x Connectivity"].to_numpy().astype(np.float32),
         (conn["Postsynaptic_Index"].to_numpy(), conn["Presynaptic_Index"].to_numpy())),
        shape=(n, n)).tocsr()
    del conn

    all_frames = []
    for label, nid in (("DNp09 left", DNP09_LEFT), ("DNp09 right", DNP09_RIGHT)):
        idx = flyid2i[nid]
        row = W.getrow(idx).tocoo()
        df = describe(ann_by_idx, row.col, row.data, f"{label}: все прямые входы")
        df["target"] = label
        all_frames.append(df)

        pos = row.data > 0
        describe(ann_by_idx, row.col[pos], row.data[pos],
                 f"{label}: только возбуждающие входы (те, что мы стимулируем)")

    # второй уровень: кто питает возбуждающих партнёров
    print("\n" + "=" * 78)
    print(" ВТОРОЙ УРОВЕНЬ")
    print("=" * 78)
    for label, nid in (("DNp09 left", DNP09_LEFT), ("DNp09 right", DNP09_RIGHT)):
        idx = flyid2i[nid]
        row = W.getrow(idx).tocoo()
        drv = row.col[row.data > 0]
        scores = {}
        for t in drv:
            r2 = W.getrow(int(t)).tocoo()
            for c, v in zip(r2.col, r2.data):
                if v > 0:
                    scores[int(c)] = scores.get(int(c), 0.0) + float(v)
        if scores:
            cols = np.array(list(scores.keys()))
            vals = np.array(list(scores.values()))
            describe(ann_by_idx, cols, vals, f"{label}: входы второго уровня")

    out_csv = out("dn_input_composition.csv")
    pd.concat(all_frames).to_csv(out_csv, index=False)
    print(f"\nсохранено: {out_csv}")


if __name__ == "__main__":
    main()
