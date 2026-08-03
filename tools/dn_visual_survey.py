"""Какие из 1299 нисходящих нейронов управляются зрением?

Зачем. Сейчас выход контура — одна пара DNp09. Аннотации показали, что в мозге
1299 нисходящих нейронов, и выход можно радикально расширить. Но брать их
наугад бессмысленно: у DNp09 сработало именно потому, что он получает крупный
чистый возбуждающий вход от зрительных проекционных нейронов (+880 при 22.8%
массы веса), тогда как восходящие от ног дают 1.8% и в сумме тормозят.

Поэтому считаем для КАЖДОГО нисходящего нейрона состав его входов по классам
и ранжируем по чистому возбуждению от зрения. Получаем обоснованный список
кандидатов вместо гадания.

Дополнительно проверяем латеральность: приходит ли к левому DN зрительный вход
преимущественно слева. Без этого пара каналов для поворота не построится.
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

DNP09_LEFT = 720575940635872101
DNP09_RIGHT = 720575940627652358
TOP_SHOW = 20


def main():
    print("=" * 78)
    print(" ОБЗОР НИСХОДЯЩИХ НЕЙРОНОВ: КТО ИЗ НИХ УПРАВЛЯЕТСЯ ЗРЕНИЕМ")
    print("=" * 78)

    comp = pd.read_csv(path_comp, index_col=0)
    flyid2i = {int(j): i for i, j in enumerate(comp.index)}
    i2flyid = {i: j for j, i in flyid2i.items()}
    n = len(flyid2i)

    ann = pd.read_csv(ANNOTATIONS, sep="\t", low_memory=False)
    ann["root_id"] = pd.to_numeric(ann["root_id"], errors="coerce")
    ann = ann.dropna(subset=["root_id"])
    ann["root_id"] = ann["root_id"].astype("int64")
    ann = ann[ann["root_id"].isin(flyid2i.keys())]

    # классы и стороны по индексу нейрона
    sc = np.array(["(нет)"] * n, dtype=object)
    side_arr = np.array(["(нет)"] * n, dtype=object)
    for rid, s, sd in zip(ann["root_id"], ann["super_class"].fillna("(нет)"),
                          ann["side"].fillna("(нет)")):
        i = flyid2i[int(rid)]
        sc[i] = s
        side_arr[i] = sd

    conn = pd.read_parquet(path_con, columns=[
        "Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
    W = sp.coo_matrix(
        (conn["Excitatory x Connectivity"].to_numpy().astype(np.float32),
         (conn["Postsynaptic_Index"].to_numpy(), conn["Presynaptic_Index"].to_numpy())),
        shape=(n, n)).tocsr()
    del conn

    desc = ann[ann["super_class"] == "descending"]
    print(f"нисходящих нейронов: {len(desc)}")

    is_vis = np.isin(sc, ["visual_projection", "visual_centrifugal"])
    is_asc = sc == "ascending"

    rows = []
    for rid, ctype, dside in zip(desc["root_id"],
                                 desc["cell_type"].fillna("(без типа)"),
                                 desc["side"].fillna("(нет)")):
        idx = flyid2i[int(rid)]
        row = W.getrow(idx).tocoo()
        if row.nnz == 0:
            continue
        cols, vals = row.col, row.data
        total_abs = np.abs(vals).sum()

        vmask = is_vis[cols]
        amask = is_asc[cols]
        vis_net = float(vals[vmask].sum())
        vis_abs = float(np.abs(vals[vmask]).sum())
        asc_net = float(vals[amask].sum())

        # латеральность зрительного входа: доля веса, пришедшая со стороны DN
        vis_cols = cols[vmask]
        vis_vals = vals[vmask]
        pos = vis_vals > 0
        if pos.sum() and dside in ("left", "right"):
            same = side_arr[vis_cols[pos]] == dside
            lat = float(vis_vals[pos][same].sum() / max(vis_vals[pos].sum(), 1e-9))
        else:
            lat = float("nan")

        rows.append({
            "root_id": int(rid), "cell_type": ctype, "side": dside,
            "n_inputs": int(row.nnz), "total_abs_weight": float(total_abs),
            "visual_net": vis_net, "visual_abs": vis_abs,
            "visual_frac": vis_abs / max(total_abs, 1e-9),
            "ascending_net": asc_net,
            "visual_ipsi_frac": lat,
        })

    df = pd.DataFrame(rows)
    print(f"с ненулевым входом: {len(df)}")

    ref = df[df["root_id"].isin([DNP09_LEFT, DNP09_RIGHT])]
    print("\n----- для сравнения, наш рабочий DNp09 -----")
    print(f"  {'сторона':>8s} {'вход зрения, чистый':>20s} {'доля массы':>12s} "
          f"{'ипсилатеральность':>19s}")
    for _, r in ref.iterrows():
        print(f"  {r['side']:>8s} {r['visual_net']:>20.0f} {r['visual_frac']:>12.3f} "
              f"{r['visual_ipsi_frac']:>19.3f}")

    print(f"\n----- топ-{TOP_SHOW} нисходящих по чистому возбуждению от зрения -----")
    top = df.sort_values("visual_net", ascending=False).head(TOP_SHOW)
    print(f"  {'cell_type':<16s} {'сторона':>7s} {'зрение':>9s} {'доля':>7s} "
          f"{'ипси':>6s} {'восходящие':>11s}")
    for _, r in top.iterrows():
        lat = "—" if np.isnan(r["visual_ipsi_frac"]) else f"{r['visual_ipsi_frac']:.2f}"
        print(f"  {str(r['cell_type'])[:16]:<16s} {r['side']:>7s} "
              f"{r['visual_net']:>9.0f} {r['visual_frac']:>7.3f} {lat:>6s} "
              f"{r['ascending_net']:>11.0f}")

    # какие типы встречаются парами лево-право с сильным зрением
    print(f"\n----- типы, представленные с ОБЕИХ сторон (годятся для поворота) -----")
    strong = df[df["visual_net"] > 100]
    pairs = []
    for ctype, g in strong.groupby("cell_type"):
        sides = set(g["side"])
        if {"left", "right"} <= sides:
            pairs.append({
                "cell_type": ctype, "n": len(g),
                "visual_net_mean": g["visual_net"].mean(),
                "visual_frac_mean": g["visual_frac"].mean(),
                "ipsi_mean": g["visual_ipsi_frac"].mean(),
            })
    pdf = pd.DataFrame(pairs).sort_values("visual_net_mean", ascending=False)
    if len(pdf):
        print(f"  {'cell_type':<18s} {'шт':>4s} {'зрениеср.':>11s} {'доля':>7s} {'ипси':>7s}")
        for _, r in pdf.head(15).iterrows():
            print(f"  {str(r['cell_type'])[:18]:<18s} {int(r['n']):>4d} "
                  f"{r['visual_net_mean']:>11.0f} {r['visual_frac_mean']:>7.3f} "
                  f"{r['ipsi_mean']:>7.2f}")
    else:
        print("  таких не нашлось")

    print("\n----- сколько нисходящих вообще получают заметный вход от ног -----")
    asc_strong = df[df["ascending_net"] > 100]
    print(f"  с чистым возбуждением от восходящих > 100: {len(asc_strong)} из {len(df)}")
    if len(asc_strong):
        t = asc_strong.sort_values("ascending_net", ascending=False).head(10)
        print(f"  {'cell_type':<18s} {'сторона':>8s} {'восходящие':>11s} {'зрение':>9s}")
        for _, r in t.iterrows():
            print(f"  {str(r['cell_type'])[:18]:<18s} {r['side']:>8s} "
                  f"{r['ascending_net']:>11.0f} {r['visual_net']:>9.0f}")

    p = out("dn_visual_survey.csv")
    df.to_csv(p, index=False)
    print(f"\nсохранено: {p}")


if __name__ == "__main__":
    main()
