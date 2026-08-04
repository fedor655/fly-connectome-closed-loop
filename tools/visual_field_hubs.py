"""Широкопольные нейроны как разрушители ретинотопии: диагноз и лечение.

Что известно. Карта поля зрения по связям надёжна: две независимые половины
связей дают одну и ту же карту, канонические корреляции между половинами 0.993
и 0.965 слева, 0.992 и 0.932 справа, а на перемешанных связях 0.08 и 0.02.
Эталон тоже цел: листы координат сомы у типов из одних колонок совпадают на
0.999 и 0.998 с обеих сторон.

При этом карта и эталон расходятся по второй оси, и расхождение растёт вглубь
пути, причём только слева:

  ламина L1 / L2      слева 0.741 / 0.847   справа 0.485 / 0.605
  медулла Mi1 / Tm1   слева 0.538 / 0.560   справа 0.911 / 0.904
  детекторы T4a / T4c слева 0.177 / 0.119   справа 0.569 / 0.591

(сверка с полными тремя координатами сомы; сплющивание эталона в плоскость двух
главных осей само по себе занижало вторую корреляцию, слева с 0.481 до 0.326)

Вторая ось есть на входе и теряется по дороге. По половинам связей карта T4
слева воспроизводится на 0.975, то есть теряется систематически, а не в шум.

Гипотеза. Виноваты широкопольные нейроны. CT1 — одна клетка на полушарие,
касающаяся всех колонок сразу. Её положение по построению равно центру глаза,
и любой, кто получает от неё заметную долю входа, тянется к центру. Такая
клетка не несёт положения вовсе, но в диффузии участвует наравне с колончатыми.
Амакринные Am, Lawf и прочие тангенциальные — того же рода.

Проверка и лечение сразу: считаем, кто в подграфе крупный узел, и разворачиваем
порог исключения. Если гипотеза верна, вторая корреляция слева вырастет при
исключении верхушки и перестанет расти дальше. Если не вырастет — гипотеза
неверна, и это надо записать так же прямо.
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

VOXEL_NM = np.array([4.0, 4.0, 40.0])
N_ITER = 12
PROBE_TYPES = ["L1", "L2", "Mi1", "Mi4", "Tm1", "Tm9", "T4a", "T4c", "T5a", "T5c"]
# Порог задаётся числом партнёров, а не процентилем: у колончатой клетки
# партнёров десятки, у широкопольной тысячи, и граница между ними физическая.
OUT_DEGREE_CAPS = [None, 3000, 1000, 500, 200]


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


def retina_frame(p_nm):
    c = p_nm - p_nm.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    nrm = vt[2]
    up = np.array([0.0, -1.0, 0.0])
    e_el = up - (up @ nrm) * nrm
    e_el /= np.linalg.norm(e_el)
    e_az = np.cross(nrm, e_el)
    e_az /= np.linalg.norm(e_az)
    if e_az @ np.array([0.0, 0.0, -1.0]) < 0:
        e_az = -e_az
    return p_nm.mean(0), np.stack([e_az, e_el])


def diffuse(A, seed_loc, seed_xy, n_nodes, n_iter=N_ITER):
    D = np.zeros((n_nodes, 2))
    fixed = np.zeros(n_nodes, bool)
    D[seed_loc] = seed_xy
    fixed[seed_loc] = True
    has = fixed.copy()
    for _ in range(n_iter):
        num = A @ (D * has[:, None])
        den = np.asarray(A @ has.astype(float)).ravel()
        ok = den > 0
        new = D.copy()
        new[ok] = num[ok] / den[ok, None]
        new[fixed] = D[fixed]
        D = new
        has = has | ok
    return D, has


def main():
    print("=" * 78)
    print(" ШИРОКОПОЛЬНЫЕ НЕЙРОНЫ И ВТОРАЯ ОСЬ РЕТИНОТОПИИ")
    print("=" * 78)

    comp = pd.read_csv(path_comp, index_col=0)
    flyid2i = {int(j): i for i, j in enumerate(comp.index)}

    ann = pd.read_csv(ANNOTATIONS, sep="\t", low_memory=False)
    ann["root_id"] = pd.to_numeric(ann["root_id"], errors="coerce")
    ann = ann.dropna(subset=["root_id"])
    ann["root_id"] = ann["root_id"].astype("int64")
    ann = ann[ann["root_id"].isin(flyid2i.keys())]
    ann["idx"] = [flyid2i[int(x)] for x in ann["root_id"]]
    ct = ann["cell_type"].fillna("")
    sc = ann["super_class"].fillna("")
    cc = ann["cell_class"].fillna("")
    type_of = dict(zip(ann["idx"], ct))

    print("\nчитаю связи...")
    conn = pd.read_parquet(path_con, columns=[
        "Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
    pre = conn["Presynaptic_Index"].to_numpy()
    post = conn["Postsynaptic_Index"].to_numpy()
    w = np.abs(conn["Excitatory x Connectivity"].to_numpy().astype(np.float64))
    del conn

    rows = []
    for side in ("left", "right"):
        m_side = ann["side"] == side
        photo = ann[m_side & (sc == "sensory") & (cc == "visual")]
        photo = photo[photo[["pos_x", "pos_y", "pos_z"]].notna().all(axis=1)]
        p = photo[["pos_x", "pos_y", "pos_z"]].to_numpy(float) * VOXEL_NM
        origin, axes = retina_frame(p)
        seed_xy = (p - origin) @ axes.T / 1000.0

        target = ann[m_side & (sc.isin(["optic", "visual_projection"]) |
                               ct.isin(PROBE_TYPES))]
        all_idx = np.unique(np.concatenate([photo["idx"].to_numpy(),
                                            target["idx"].to_numpy()]))
        keep = np.isin(pre, all_idx) & np.isin(post, all_idx)
        ipr = np.searchsorted(all_idx, pre[keep])
        ipo = np.searchsorted(all_idx, post[keep])
        ww = w[keep]
        n_nodes = len(all_idx)
        seed_loc = np.searchsorted(all_idx, photo["idx"].to_numpy())

        out_deg = np.bincount(ipr, minlength=n_nodes)
        print("\n" + "-" * 78)
        print(f" глаз {side}: узлов {n_nodes}, рёбер {len(ipr)}")
        print("-" * 78)
        top = np.argsort(-out_deg)[:12]
        print(f"  {'тип':<14s} {'партнёров на выходе':>20s}")
        for j in top:
            print(f"  {str(type_of.get(all_idx[j], '(нет)'))[:14]:<14s} "
                  f"{int(out_deg[j]):>20d}")
        print(f"  нейронов с более чем 500 партнёрами: "
              f"{int((out_deg > 500).sum())}; более 1000: {int((out_deg > 1000).sum())}")

        print(f"\n  {'порог':>7s} {'убрано':>7s}" +
              "".join(f"{t:>14s}" for t in ("Mi1", "T4a", "T4c", "T5c")))
        for cap in OUT_DEGREE_CAPS:
            if cap is None:
                sel = np.ones(len(ipr), bool)
                n_drop = 0
            else:
                hub = out_deg > cap
                hub[seed_loc] = False          # семена не трогаем
                sel = ~hub[ipr]
                n_drop = int(hub.sum())
            A = sp.coo_matrix((ww[sel], (ipo[sel], ipr[sel])),
                              shape=(n_nodes, n_nodes)).tocsr()
            D, has = diffuse(A, seed_loc, seed_xy, n_nodes)
            cells = []
            for t in ("Mi1", "T4a", "T4c", "T5c"):
                sub = ann[(ct == t) & m_side]
                sub = sub[sub[["pos_x", "pos_y", "pos_z"]].notna().all(axis=1)]
                loc = np.searchsorted(all_idx, sub["idx"].to_numpy())
                ok = (loc < n_nodes) & (all_idx[np.clip(loc, 0, n_nodes - 1)] ==
                                        sub["idx"].to_numpy())
                loc, sub = loc[ok], sub[ok]
                g = has[loc]
                if g.sum() < 50:
                    cells.append(float("nan"))
                    continue
                P3 = sub[["pos_x", "pos_y", "pos_z"]].to_numpy(float)[g] * VOXEL_NM
                _, c2 = cca(D[loc][g], P3)
                cells.append(c2)
                rows.append({"side": side, "cap": -1 if cap is None else cap,
                             "cell_type": t, "cc2_vs_soma3d": c2,
                             "n_hubs_removed": n_drop})
            lab = "нет" if cap is None else str(cap)
            print(f"  {lab:>7s} {n_drop:>7d}" +
                  "".join(f"{v:>14.3f}" for v in cells))

    df = pd.DataFrame(rows)
    df.to_csv(out("visual_field_hubs.csv"), index=False)
    print(f"\nсохранено: {out('visual_field_hubs.csv')}")

    print("\n" + "=" * 78)
    print(" ЧТЕНИЕ")
    print("=" * 78)
    print("  Числа — вторая каноническая корреляция карты с ПОЛНЫМИ тремя")
    print("  координатами сомы. Первая ось везде около 0.98 и здесь не печатается.")
    print()
    print("  Гипотеза подтверждается, если слева при исключении верхушки вторая")
    print("  корреляция заметно растёт и выходит на полку. Опровергается, если")
    print("  не меняется или падает — тогда широкопольные ни при чём.")


if __name__ == "__main__":
    main()
