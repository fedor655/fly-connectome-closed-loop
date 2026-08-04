"""Почему у левого глаза не восстанавливается вторая ось поля зрения?

Что известно. Карта поля зрения по связям даёт канонические корреляции с листом
координат сомы 0.986 и 0.678 у правого глаза против 0.985 и 0.164 у левого.
Первая ось восстановлена у обоих, вторая только у правого. Число итераций
сглаживания ни при чём: справа вторая корреляция растёт с итерациями
0.262 → 0.678, слева стоит на 0.16 и не двигается. Затравка двумерна у обоих:
доли 0.740 / 0.260 слева и 0.716 / 0.284 справа. Подграфы почти равны: слева
48 536 нейронов и 3 962 224 ребра, справа 47 811 и 4 292 875.

Версий три, и они требуют РАЗНЫХ действий, поэтому различить их надо до того,
как что-то чинить:

  A. Плоха карта. Вторая ось слева — шум, и в проводке левой доли её просто нет.
  B. Плох эталон. Карта слева не хуже, но лист координат сомы слева негоден
     по второй оси, и корреляция меряет его беду, а не нашу.
  C. Плоха затравка. Среди «сенсорных зрительных» слева есть примесь — например
     фоторецепторы глазков, они сидят посреди головы и тянут карту к себе.

Три проверки, по одной на версию.

  Против A — надёжность по половинам. Связи делятся случайно пополам, на каждой
  половине строится своя карта, и они сравниваются друг с другом. Координаты
  сомы тут не участвуют вовсе. Если вторая ось — сигнал, половины согласятся;
  если шум, разойдутся. Это прямая проверка версии A, и она не зависит от
  качества эталона.

  Против B — согласие эталона с самим собой. Mi1 и Tm1 сидят в одних колонках
  медуллы, значит их листы координат сомы обязаны совпадать по обеим осям.
  Если слева совпадают только по первой, эталон слева и правда одномерен.

  Против C — состав и форма затравки: разбор по типам, поиск выбросов,
  устойчивый размах по процентилям вместо минимума и максимума.
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
PROBE_TYPES = ["Mi1", "Mi4", "Tm1", "Tm9", "T4a", "T4c", "T5c", "L1", "L2"]
CO_COLUMNAR = [("Mi1", "Tm1"), ("Mi1", "Mi4"), ("T4a", "T4c"), ("Mi1", "T4c")]


def canon_corr(A, B):
    def white(X):
        X = X - X.mean(0)
        q, r_ = np.linalg.qr(X)
        return q[:, :np.linalg.matrix_rank(r_)]
    qa, qb = white(A), white(B)
    if qa.shape[1] < 2 or qb.shape[1] < 2:
        return float("nan"), float("nan")
    s = np.clip(np.linalg.svd(qa.T @ qb, compute_uv=False), 0, 1)
    return float(s[0]), float(s[1])


def sheet_coords(p_nm):
    c = p_nm - p_nm.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    return c @ vt[:2].T


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
    return p_nm.mean(0), np.stack([e_az, e_el]), nrm


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
    print(" ДИАГНОЗ ЛЕВОГО ГЛАЗА")
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

    # ================= версия C: состав затравки =================
    print("\n" + "=" * 78)
    print(" C. СОСТАВ И ФОРМА ЗАТРАВКИ")
    print("=" * 78)
    for side in ("left", "right"):
        m = (sc == "sensory") & (cc == "visual") & (ann["side"] == side)
        sub = ann[m & ann[["pos_x", "pos_y", "pos_z"]].notna().all(axis=1)]
        p = sub[["pos_x", "pos_y", "pos_z"]].to_numpy(float) * VOXEL_NM
        types = sub["cell_type"].fillna("(без типа)").value_counts()
        print(f"\n  {side}: всего {len(sub)}")
        print("    типы:", ", ".join(f"{k} {v}" for k, v in types.items()))
        origin, axes, nrm = retina_frame(p)
        xy = (p - origin) @ axes.T / 1000.0
        rng_full = xy.max(0) - xy.min(0)
        lo, hi = np.percentile(xy, [1, 99], axis=0)
        print(f"    размах полный   {np.round(rng_full, 0)} мкм")
        print(f"    размах 1-99 проц {np.round(hi - lo, 0)} мкм  "
              f"(доля выбросов по краям видна из разницы)")
        d = np.linalg.norm(xy - np.median(xy, 0), axis=1)
        mad = np.median(np.abs(d - np.median(d)))
        n_out = int((d > np.median(d) + 5 * max(mad, 1e-9)).sum())
        print(f"    выбросов дальше 5 MAD от центра: {n_out}")
        # сколько «сенсорных зрительных» сидят далеко от основного облака
        for t in ("R1-6", "R7", "R8"):
            mt = sub["cell_type"].fillna("") == t
            if mt.sum() < 20:
                continue
            q = xy[mt.to_numpy()]
            print(f"      {t}: n={int(mt.sum())}, размах "
                  f"{np.round(q.max(0) - q.min(0), 0)} мкм")

    # ================= версия B: качество эталона =================
    print("\n" + "=" * 78)
    print(" B. СОГЛАСИЕ ЭТАЛОНА С САМИМ СОБОЙ")
    print("=" * 78)
    print("  Типы одних колонок обязаны давать совпадающие листы координат сомы.")
    print("  Сравниваем канонической корреляцией после сопоставления ближайших.")
    print(f"\n  {'пара':<14s} {'сторона':>7s} {'канон. 1':>10s} {'канон. 2':>10s}")
    ref_rows = []
    for a, b in CO_COLUMNAR:
        for side in ("left", "right"):
            sa = ann[(ct == a) & (ann["side"] == side)]
            sb = ann[(ct == b) & (ann["side"] == side)]
            sa = sa[sa[["pos_x", "pos_y", "pos_z"]].notna().all(axis=1)]
            sb = sb[sb[["pos_x", "pos_y", "pos_z"]].notna().all(axis=1)]
            if len(sa) < 50 or len(sb) < 50:
                continue
            XA = sheet_coords(sa[["pos_x", "pos_y", "pos_z"]].to_numpy(float) * VOXEL_NM)
            XB = sheet_coords(sb[["pos_x", "pos_y", "pos_z"]].to_numpy(float) * VOXEL_NM)
            # каждому нейрону a — ближайший по листу нейрон b
            dd = ((XA[:, None, :] - XB[None, :, :]) ** 2).sum(-1)
            j = dd.argmin(1)
            c1, c2 = canon_corr(XA, XB[j])
            print(f"  {a + '/' + b:<14s} {side:>7s} {c1:>10.3f} {c2:>10.3f}")
            ref_rows.append({"pair": f"{a}/{b}", "side": side,
                             "ref_cc1": c1, "ref_cc2": c2})

    # ================= версия A: надёжность по половинам =================
    print("\n" + "=" * 78)
    print(" A. НАДЁЖНОСТЬ КАРТЫ ПО ПОЛОВИНАМ СВЯЗЕЙ")
    print("=" * 78)
    print("  читаю связи...")
    conn = pd.read_parquet(path_con, columns=[
        "Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
    pre = conn["Presynaptic_Index"].to_numpy()
    post = conn["Postsynaptic_Index"].to_numpy()
    w = np.abs(conn["Excitatory x Connectivity"].to_numpy().astype(np.float64))
    del conn

    rng = np.random.default_rng(20260804)
    rows = []
    for side in ("left", "right"):
        m_side = ann["side"] == side
        photo = ann[m_side & (sc == "sensory") & (cc == "visual")]
        photo = photo[photo[["pos_x", "pos_y", "pos_z"]].notna().all(axis=1)]
        p = photo[["pos_x", "pos_y", "pos_z"]].to_numpy(float) * VOXEL_NM
        origin, axes, nrm = retina_frame(p)
        seed_xy = (p - origin) @ axes.T / 1000.0

        target = ann[m_side & (sc.isin(["optic", "visual_projection"]) |
                               ct.isin(PROBE_TYPES))]
        all_idx = np.unique(np.concatenate([photo["idx"].to_numpy(),
                                            target["idx"].to_numpy()]))
        keep = np.isin(pre, all_idx) & np.isin(post, all_idx)
        pr, po, ww = pre[keep], post[keep], w[keep]
        ipr, ipo = np.searchsorted(all_idx, pr), np.searchsorted(all_idx, po)
        seed_loc = np.searchsorted(all_idx, photo["idx"].to_numpy())
        n_nodes = len(all_idx)

        half = rng.random(len(ipr)) < 0.5
        maps = []
        for sel in (half, ~half):
            A = sp.coo_matrix((ww[sel], (ipo[sel], ipr[sel])),
                              shape=(n_nodes, n_nodes)).tocsr()
            maps.append(diffuse(A, seed_loc, seed_xy, n_nodes))
        # контроль: перемешать пресинаптические концы в одной из половин
        A_sh = sp.coo_matrix((ww[half], (ipo[half], rng.permutation(ipr[half]))),
                             shape=(n_nodes, n_nodes)).tocsr()
        map_sh = diffuse(A_sh, seed_loc, seed_xy, n_nodes)

        print(f"\n  глаз {side}: узлов {n_nodes}, рёбер {len(ipr)}, "
              f"в половине {int(half.sum())}")
        print(f"  {'тип':<6s} {'n':>5s} {'половины: 1-я':>14s} {'2-я':>7s} "
              f"{'перемешано: 1-я':>17s} {'2-я':>7s}")
        for t in PROBE_TYPES:
            sub = ann[(ct == t) & m_side]
            loc = np.searchsorted(all_idx, sub["idx"].to_numpy())
            ok = (loc < n_nodes) & (all_idx[np.clip(loc, 0, n_nodes - 1)] ==
                                    sub["idx"].to_numpy())
            loc = loc[ok]
            if len(loc) < 50:
                continue
            g = maps[0][1][loc] & maps[1][1][loc]
            if g.sum() < 50:
                continue
            c1, c2 = canon_corr(maps[0][0][loc][g], maps[1][0][loc][g])
            gs = maps[0][1][loc] & map_sh[1][loc]
            s1, s2 = canon_corr(maps[0][0][loc][gs], map_sh[0][loc][gs])
            print(f"  {t:<6s} {len(loc):>5d} {c1:>14.3f} {c2:>7.3f} "
                  f"{s1:>17.3f} {s2:>7.3f}")
            rows.append({"side": side, "cell_type": t, "n": int(len(loc)),
                         "split_cc1": c1, "split_cc2": c2,
                         "shuffled_cc1": s1, "shuffled_cc2": s2})

    df = pd.DataFrame(rows)
    df.to_csv(out("left_eye_diagnosis.csv"), index=False)
    pd.DataFrame(ref_rows).to_csv(out("left_eye_reference.csv"), index=False)

    print("\n" + "=" * 78)
    print(" ЧТЕНИЕ")
    print("=" * 78)
    if not df.empty:
        for side in ("left", "right"):
            d = df[df["side"] == side]
            if d.empty:
                continue
            print(f"  {side}: половины сходятся на {d['split_cc1'].mean():.3f} и "
                  f"{d['split_cc2'].mean():.3f}; перемешано "
                  f"{d['shuffled_cc1'].mean():.3f} и {d['shuffled_cc2'].mean():.3f}")
    print()
    print("  Версия A (плоха карта) верна, если слева вторая корреляция между")
    print("  половинами низкая. Тогда второй оси в проводке левой доли нет.")
    print("  Версия B (плох эталон) верна, если половины слева сходятся хорошо,")
    print("  а не сходится только сверка с листом сомы — значит негоден лист.")


if __name__ == "__main__":
    main()
