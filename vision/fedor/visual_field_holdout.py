"""Настоящая проверка карты: спрятать часть фоторецепторов и угадать их место.

Зачем именно так. Все прежние сверки шли против координат сомы, а этот эталон
оказался негодным. Видно по его же числам: сопоставление нейронов разных
нейропилей по близости сомы даёт разлёт 12-24 мкм при шаге колонки около 5, и
на таких парах вторая корреляция падает до 0.027 там, где на надёжных парах
(разлёт 3-9 мкм) она 0.94-0.97. То есть рушится сопоставление, а не карта.
Отсюда и мнимая разница между глазами, из-за которой я объявил левый глаз
одномерным.

Здесь эталон настоящий. У фоторецепторов истинное положение в сетчатке
известно — они и служат затравкой. Прячем часть из них, строим карту по
остальным, а потом предсказываем место спрятанного как средневзвешенное по
связям место тех, кому он сам шлёт сигнал. Ошибка меряется в микронах, по
каждой оси отдельно, и это уже не корреляция, а расстояние.

Два контроля обязательны:
  1. Предсказание центром облака — сколько ошибётся тот, кто не знает ничего.
  2. Перемешанные связи — сколько останется, если проводку разрушить.
Если наша ошибка не отличается от первого контроля, карта бесполезна.

Отдельно печатается ошибка по второй оси: именно она под вопросом. Для меры
даётся шаг между соседними омматидиями, посчитанный по самим фоторецепторам.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from flypaths import ANNOTATIONS, add_fly_brain_to_path, out  # noqa: E402

add_fly_brain_to_path()
from benchmark import path_comp, path_con  # noqa: E402

VOXEL_NM = np.array([4.0, 4.0, 40.0])
N_ITER = 12
HOLDOUT = 0.20
N_REPEAT = 3
PROBE_TYPES = ["L1", "L2", "Mi1", "Mi4", "Tm1", "Tm9", "T4a", "T4c", "T5a", "T5c"]


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
    print(" ПРОВЕРКА НА СПРЯТАННЫХ ФОТОРЕЦЕПТОРАХ")
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
        xy_true = (p - origin) @ axes.T / 1000.0

        # шаг между соседними фоторецепторами разных омматидиев: мера, в которой
        # ошибку осмысленно читать
        d = np.sqrt(((xy_true[:, None, :] - xy_true[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        step = float(np.median(np.sort(d, axis=1)[:, 6]))

        target = ann[m_side & (sc.isin(["optic", "visual_projection"]) |
                               ct.isin(PROBE_TYPES))]
        all_idx = np.unique(np.concatenate([photo["idx"].to_numpy(),
                                            target["idx"].to_numpy()]))
        keep = np.isin(pre, all_idx) & np.isin(post, all_idx)
        ipr = np.searchsorted(all_idx, pre[keep])
        ipo = np.searchsorted(all_idx, post[keep])
        ww = w[keep]
        n_nodes = len(all_idx)
        ph_loc = np.searchsorted(all_idx, photo["idx"].to_numpy())

        # матрица «от кого кому»: для предсказания спрятанного нужен его ВЫХОД
        A_fwd = sp.coo_matrix((ww, (ipo, ipr)), shape=(n_nodes, n_nodes)).tocsr()
        A_out = sp.coo_matrix((ww, (ipr, ipo)), shape=(n_nodes, n_nodes)).tocsr()

        print("\n" + "-" * 78)
        print(f" глаз {side}: фоторецепторов {len(photo)}, шаг между омматидиями "
              f"{step:.1f} мкм")
        print("-" * 78)

        for rep in range(N_REPEAT):
            rng = np.random.default_rng(1000 + rep)
            hide = rng.random(len(ph_loc)) < HOLDOUT
            seed_loc, seed_xy = ph_loc[~hide], xy_true[~hide]
            D, has = diffuse(A_fwd, seed_loc, seed_xy, n_nodes)

            def predict(mat):
                num = mat[ph_loc[hide]] @ (D * has[:, None])
                den = np.asarray(mat[ph_loc[hide]] @ has.astype(float)).ravel()
                ok = den > 0
                pr_ = np.full((int(hide.sum()), 2), np.nan)
                pr_[ok] = num[ok] / den[ok, None]
                return pr_, ok

            pred, ok = predict(A_out)
            true = xy_true[hide]
            err = np.abs(pred[ok] - true[ok])
            dist = np.linalg.norm(pred[ok] - true[ok], axis=1)

            # контроль 1: центр облака
            base = np.linalg.norm(true[ok] - seed_xy.mean(0), axis=1)
            # контроль 2: перемешанные связи
            A_sh = sp.coo_matrix((ww, (ipo, rng.permutation(ipr))),
                                 shape=(n_nodes, n_nodes)).tocsr()
            Dsh, hsh = diffuse(A_sh, seed_loc, seed_xy, n_nodes)
            num = A_out[ph_loc[hide]] @ (Dsh * hsh[:, None])
            den = np.asarray(A_out[ph_loc[hide]] @ hsh.astype(float)).ravel()
            ok2 = den > 0
            psh = num[ok2] / den[ok2, None]
            dsh = np.linalg.norm(psh - true[ok2], axis=1)

            print(f"  повтор {rep + 1}: спрятано {int(hide.sum())}, "
                  f"предсказано {int(ok.sum())}")
            print(f"    ошибка медиана {np.median(dist):>6.1f} мкм "
                  f"= {np.median(dist) / step:>4.1f} шага; "
                  f"по 1-й оси {np.median(err[:, 0]):>5.1f}, "
                  f"по 2-й оси {np.median(err[:, 1]):>5.1f} мкм")
            print(f"    контроль «центр облака» {np.median(base):>6.1f} мкм; "
                  f"контроль «связи перемешаны» {np.median(dsh):>6.1f} мкм")
            rows.append({"side": side, "repeat": rep, "step_um": step,
                         "n_hidden": int(hide.sum()), "n_predicted": int(ok.sum()),
                         "err_median_um": float(np.median(dist)),
                         "err_axis1_um": float(np.median(err[:, 0])),
                         "err_axis2_um": float(np.median(err[:, 1])),
                         "err_center_um": float(np.median(base)),
                         "err_shuffled_um": float(np.median(dsh))})

    df = pd.DataFrame(rows)
    df.to_csv(out("visual_field_holdout.csv"), index=False)
    print(f"\nсохранено: {out('visual_field_holdout.csv')}")

    print("\n" + "=" * 78)
    print(" ИТОГ")
    print("=" * 78)
    for side in ("left", "right"):
        d = df[df["side"] == side]
        if d.empty:
            continue
        print(f"  {side}: ошибка {d['err_median_um'].mean():.1f} мкм "
              f"({d['err_median_um'].mean() / d['step_um'].mean():.1f} шага), "
              f"по осям {d['err_axis1_um'].mean():.1f} и {d['err_axis2_um'].mean():.1f}; "
              f"контроли {d['err_center_um'].mean():.1f} и "
              f"{d['err_shuffled_um'].mean():.1f}")
    print()
    print("  Карта годится, если ошибка много меньше обоих контролей И если")
    print("  ошибка по второй оси сопоставима с ошибкой по первой. Если вторая")
    print("  ось не работает, ошибка по ней сравняется с разбросом самого облака.")


if __name__ == "__main__":
    main()
