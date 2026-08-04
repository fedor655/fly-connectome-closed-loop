"""Разовая проверка: теряется ли вторая ось при сплющивании эталона в плоскость."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flypaths import ANNOTATIONS, add_fly_brain_to_path, out  # noqa: E402

add_fly_brain_to_path()
from benchmark import path_comp  # noqa: E402

VOX = np.array([4.0, 4.0, 40.0])


def cca(A, B):
    def w(X):
        X = X - X.mean(0)
        q, r = np.linalg.qr(X)
        return q[:, :np.linalg.matrix_rank(r)]
    qa, qb = w(A), w(B)
    if qa.shape[1] < 2 or qb.shape[1] < 2:
        return float("nan"), float("nan")
    s = np.clip(np.linalg.svd(qa.T @ qb, compute_uv=False), 0, 1)
    return float(s[0]), float(s[1])


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

print(f"{'тип':<6s} {'сторона':>7s} {'сплющено в плоскость':>22s} "
      f"{'все три координаты':>22s}")
rows = []
for t in ["Mi1", "Mi4", "Tm1", "Tm9", "T4a", "T4c", "T5c", "L1", "L2"]:
    for side in ("left", "right"):
        s = m[(ct == t) & (m["side"] == side)]
        if len(s) < 50:
            continue
        M = s[["azimuth_um", "elevation_um"]].to_numpy(float)
        P = s[["pos_x", "pos_y", "pos_z"]].to_numpy(float) * VOX
        c = P - P.mean(0)
        _, _, vt = np.linalg.svd(c, full_matrices=False)
        a1, a2 = cca(M, c @ vt[:2].T)
        b1, b2 = cca(M, P)
        print(f"{t:<6s} {side:>7s} {a1:>12.3f} {a2:>9.3f} {b1:>12.3f} {b2:>9.3f}")
        rows.append((side, a1, a2, b1, b2))

r = np.array([[x[1], x[2], x[3], x[4]] for x in rows])
sd = [x[0] for x in rows]
for side in ("left", "right"):
    k = [i for i, s in enumerate(sd) if s == side]
    print(f"\n{side}: сплющено {r[k, 0].mean():.3f} / {r[k, 1].mean():.3f}   "
          f"все три координаты {r[k, 2].mean():.3f} / {r[k, 3].mean():.3f}")
