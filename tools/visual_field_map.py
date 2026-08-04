"""Где в поле зрения сидит каждый нейрон оптических долей — по связям, а не по координатам.

Зачем. Гибриду нужно знать, какому нейрону FlyWire отдать активность какой
колонки flyvis. Напрашивающийся способ — взять координаты и разложить их по
шестиугольной решётке — измерен и не работает: у Mi1 слева CV шага между
соседями 0.50, отношение шестого соседа к первому 2.62, а у случайного облака
той же плотности 0.52 и 2.83. То есть по соседям координаты сомы неотличимы от
случайных: точка в аннотациях стоит у тела клетки в корковом слое, а колонка —
это её отросток. Ровно на этом провалилась и ретинотопия по LC.

Что делаем вместо. Позицию в поле зрения берём не из геометрии, а из связей.
Фоторецепторы сидят в самой сетчатке и упорядочены ретинотопически по
построению, их координаты проверены: доли дисперсии 0.709 / 0.249 / 0.041.
Значит им можно назначить направление взгляда честно, из геометрии глаза. А
дальше направление разносится по коннектому: позиция нейрона — это средневзвешенное
по весам связей направление его пресинаптических партнёров. Сигнал идёт
сетчатка → ламина → медулла → лобула, и позиция едет вместе с ним.

Оси FAFB установлены замером по известным структурам, а не приняты на веру:
  x, охват 815 мкм — лево-право (левое полушарие 324 мкм, правое 722);
  y, охват 392 мкм — дорсо-вентраль (вкусовые GRN подглоточной зоны 349, LC9 179);
  z, охват 278 мкм — передне-задняя (обоняние 33, лобула 239).
Ход зрительного пути виден в тех же числах: фоторецепторы x=192, ламина 229,
медулла 259, лобула 325 — слои идут к средней линии по x, а ретинотопический
лист натянут на y и z.

Две проверки, без которых верить результату нельзя:
  1. Согласие с независимым источником. Координаты сомы для одной колонки
     негодны, но КРУПНЫЙ порядок в них есть: лист двумерный. Если позиция по
     связям и позиция по соме коррелируют, значит связи разнесли направление
     туда же, куда указывает анатомия. Это независимая проверка, потому что в
     диффузию координаты нейронов не входят вовсе — только веса связей.
  2. Контроль с перемешиванием. Если перемешать пресинаптические концы рёбер,
     согласие обязано развалиться. Если не развалится — значит его давала не
     проводка, а что-то ещё.
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
REPORT_TYPES = ["L1", "L2", "L3", "L5", "Mi1", "Mi4", "Mi9", "Tm1", "Tm2", "Tm9",
                "T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]


def fit_sphere(p):
    """Глаз — выпуклая поверхность. |p - c|^2 = r^2 решается линейно."""
    A = np.hstack([2 * p, np.ones((len(p), 1))])
    b = (p ** 2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:3]
    r = np.sqrt(max(sol[3] + (c ** 2).sum(), 0.0))
    return c, r, np.abs(np.linalg.norm(p - c, axis=1) - r)


def retina_frame(p_nm):
    """Система координат самой сетчатки: две касательные оси её листа.

    Первый заход считал направление взгляда через подгонку сферы и дал
    неправдоподобные углы: левый глаз покрывал 84 градуса азимута, правый 29,
    хотя у мухи оба около 150 и симметрично. Причина в обусловленности: сетчатка
    — пологая шапка, центр сферы по ней определяется плохо (радиус вышел 182 мкм
    при невязке 13), и угловой размах сжимается тем сильнее, чем дальше уехал
    центр. Порядок нейронов при этом оставался верным, ломался только масштаб.

    Поэтому работаем в плоскости листа, где такой болезни нет.

    Второй заход брал главные оси листа каждого глаза по отдельности и выбирал
    из них ту, что ближе к анатомическому «вверх». Вышла несогласованность между
    глазами: у левого высота шла -15..98, у правого -77..78 — то есть оси у двух
    листов сопоставились по-разному.

    Третий заход брал анатомические оси напрямую, пару (-z, -y). Это оказалось
    неверно: -z не касательная к сетчатке, а частично ось глубины. Видно по тому
    же ходу пути, где z растёт от 159 у фоторецепторов до 239 у лобулы вместе с
    x. Азимут сжался до 33 мкм размаха, согласие упало с 0.869 до 0.750.

    Сейчас рамка строится из анатомии ВНУТРИ касательной плоскости, а не
    выбирается среди её осей: «вверх» проецируется в плоскость и даёт ось
    высоты, азимут берётся перпендикулярно ей в той же плоскости. Форма
    построения одна для обоих глаз, поэтому и рамки выходят согласованными.
    Проекции на анатомические оси печатаются как диагностика, чтобы разметку
    можно было проверить, а не принимать на слово.
    """
    c = p_nm - p_nm.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    nrm = vt[2]                                   # нормаль к листу сетчатки
    up = np.array([0.0, -1.0, 0.0])
    e_el = up - (up @ nrm) * nrm
    e_el /= np.linalg.norm(e_el)
    e_az = np.cross(nrm, e_el)
    e_az /= np.linalg.norm(e_az)
    fwd = np.array([0.0, 0.0, -1.0])
    if e_az @ fwd < 0:                            # азимут растёт вперёд
        e_az = -e_az
    return p_nm.mean(0), np.stack([e_az, e_el]), nrm


def sheet_coords(p_nm):
    """Две главные оси листа — независимый от связей источник крупного порядка."""
    c = p_nm - p_nm.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    return c @ vt[:2].T


def canon_corr(A, B):
    """Канонические корреляции двух двумерных облаков.

    Мера обязана быть инвариантной к повороту. Первая версия брала максимум
    модуля корреляции каждой оси карты с каждой осью листа и усредняла — и это
    была ошибка: диффузия во всех заходах давала ОДНУ И ТУ ЖЕ карту, менялась
    только рамка её считывания, то есть поворот. Мера при этом прыгала с 0.869
    до 0.750 и обратно, хотя содержание карты не менялось вовсе. Каноническая
    корреляция от поворота не зависит и потому годится, а та мера не годилась.
    """
    def white(X):
        X = X - X.mean(0)
        q, r_ = np.linalg.qr(X)
        return q[:, :np.linalg.matrix_rank(r_)]
    qa, qb = white(A), white(B)
    if qa.shape[1] == 0 or qb.shape[1] == 0:
        return np.nan, np.nan
    s = np.linalg.svd(qa.T @ qb, compute_uv=False)
    s = np.clip(s, 0, 1)
    return (float(s[0]), float(s[1]) if len(s) > 1 else float("nan"))


def main():
    print("=" * 78)
    print(" КАРТА ПОЛЯ ЗРЕНИЯ ПО СВЯЗЯМ")
    print("=" * 78)

    comp = pd.read_csv(path_comp, index_col=0)
    flyid2i = {int(j): i for i, j in enumerate(comp.index)}
    n = len(flyid2i)

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
    print(f"  рёбер: {len(pre)}")

    def run_side(side, shuffled=False, rng=None):
        m_side = ann["side"] == side
        photo = ann[m_side & (sc == "sensory") & (cc == "visual")]
        photo = photo[photo[["pos_x", "pos_y", "pos_z"]].notna().all(axis=1)]
        p = photo[["pos_x", "pos_y", "pos_z"]].to_numpy(float) * VOXEL_NM
        c, r, resid = fit_sphere(p)          # только как диагностика формы глаза
        origin, axes, nrm = retina_frame(p)
        d = (p - origin) @ axes.T / 1000.0   # мкм в плоскости сетчатки

        # приёмники позиции: всё зрительное этой стороны
        target = ann[m_side & (sc.isin(["optic", "visual_projection"]) |
                               ct.isin(REPORT_TYPES))]
        seed_idx = photo["idx"].to_numpy()
        all_idx = np.unique(np.concatenate([seed_idx, target["idx"].to_numpy()]))

        # подграф; индексы переводим searchsorted, а не словарём в цикле —
        # рёбер полтора миллиона, питоновский цикл тут стоит минуты
        keep = np.isin(pre, all_idx) & np.isin(post, all_idx)
        pr, po, ww = pre[keep], post[keep], w[keep]
        if shuffled:
            pr = rng.permutation(pr)
        pos_of = {v: i for i, v in enumerate(all_idx)}
        A = sp.coo_matrix((ww, (np.searchsorted(all_idx, po),
                                np.searchsorted(all_idx, pr))),
                          shape=(len(all_idx), len(all_idx))).tocsr()

        # диффузия координат сетчатки вверх по пути
        D = np.zeros((len(all_idx), 2))
        fixed = np.zeros(len(all_idx), bool)
        loc_seed = np.searchsorted(all_idx, seed_idx)
        D[loc_seed] = d
        fixed[loc_seed] = True
        has = fixed.copy()
        for _ in range(N_ITER):
            num = A @ (D * has[:, None])
            den = np.asarray(A @ has.astype(float)).ravel()
            ok = den > 0
            new = D.copy()
            new[ok] = num[ok] / den[ok, None]
            new[fixed] = D[fixed]
            D = new
            has = has | ok
        return all_idx, pos_of, D, has, fixed, len(pr), c, r, resid, axes, nrm

    rng = np.random.default_rng(4242)
    results = {}
    for side in ("left", "right"):
        print("\n" + "-" * 78)
        print(f" глаз: {side}")
        print("-" * 78)
        all_idx, pos_of, D, has, fixed, n_edge, c, r, resid, axes, nrm = run_side(side)
        print(f"  фоторецепторов-семян: {int(fixed.sum())}; нейронов в подграфе: "
              f"{len(all_idx)}; рёбер: {n_edge}")
        print(f"  сфера глаза: радиус {r / 1000:.1f} мкм, невязка медиана "
              f"{np.median(resid) / 1000:.1f} мкм")
        print(f"  нормаль листа сетчатки (x,y,z): {np.round(nrm, 2)}")
        print(f"  ось азимута:  {np.round(axes[0], 2)}")
        print(f"  ось высоты:   {np.round(axes[1], 2)}  (проекция на анатом. «вверх» "
              f"{axes[1] @ np.array([0, -1, 0]):+.2f})")
        print(f"  получили позицию: {int(has.sum())} из {len(all_idx)} "
              f"({has.mean():.1%})")

        az, el = D[:, 0], D[:, 1]
        _, _, Ds, hs, *_ = run_side(side, shuffled=True, rng=rng)
        az_s, el_s = Ds[:, 0], Ds[:, 1]

        print(f"\n  {'тип':<6s} {'n':>5s} {'с позицией':>11s} {'азимут, мкм':>16s} "
              f"{'высота, мкм':>16s} {'кан.корр 1':>8s} {'2':>7s} "
              f"{'перем.1':>9s} {'2':>8s}")
        rows = []
        for t in REPORT_TYPES:
            sub = ann[(ct == t) & (ann["side"] == side)]
            sub = sub[sub[["pos_x", "pos_y", "pos_z"]].notna().all(axis=1)]
            if len(sub) < 50:
                continue
            loc = np.array([pos_of[i] for i in sub["idx"] if i in pos_of])
            if len(loc) < 50:
                continue
            good = has[loc]
            a, e = az[loc][good], el[loc][good]
            xy = sheet_coords(sub[["pos_x", "pos_y", "pos_z"]].to_numpy(float) * VOXEL_NM)[good]

            cc1, cc2 = canon_corr(np.stack([a, e], 1), xy)
            gs = hs[loc]
            a2, e2 = az_s[loc][gs], el_s[loc][gs]
            xy2 = sheet_coords(sub[["pos_x", "pos_y", "pos_z"]].to_numpy(float) * VOXEL_NM)[gs]
            sc1, sc2 = (canon_corr(np.stack([a2, e2], 1), xy2) if gs.sum() > 10
                        else (float("nan"), float("nan")))

            print(f"  {t:<6s} {len(sub):>5d} {good.mean():>10.0%} "
                  f"{a.min():>7.0f}..{a.max():<8.0f} {e.min():>7.0f}..{e.max():<8.0f} "
                  f"{cc1:>8.3f} {cc2:>7.3f} {sc1:>9.3f} {sc2:>8.3f}")
            rows.append({"side": side, "cell_type": t, "n": len(sub),
                         "frac_with_position": float(good.mean()),
                         "az_min_um": float(a.min()), "az_max_um": float(a.max()),
                         "el_min_um": float(e.min()), "el_max_um": float(e.max()),
                         "cc1_vs_soma_sheet": cc1, "cc2_vs_soma_sheet": cc2,
                         "cc1_shuffled": sc1, "cc2_shuffled": sc2})
        results[side] = (all_idx, az, el, has, pd.DataFrame(rows))

    df = pd.concat([results[s][4] for s in results], ignore_index=True)
    df.to_csv(out("visual_field_map_stats.csv"), index=False)

    keep_rows = []
    for side, (all_idx, az, el, has, _) in results.items():
        rid = {v: k for k, v in flyid2i.items()}
        for j, gi in enumerate(all_idx):
            if has[j]:
                keep_rows.append({"root_id": rid[gi], "idx": int(gi), "side": side,
                                  "azimuth_um": float(az[j]), "elevation_um": float(el[j])})
    pd.DataFrame(keep_rows).to_csv(out("visual_field_map.csv"), index=False)
    print(f"\nсохранено: {out('visual_field_map.csv')} ({len(keep_rows)} нейронов)")
    print(f"сохранено: {out('visual_field_map_stats.csv')}")

    print("\n" + "=" * 78)
    print(" КРИТЕРИЙ ПРИЁМКИ")
    print("=" * 78)
    print("  Карта годится, если ОБЕ канонические корреляции с листом сомы заметные")
    print("  (около 0.5 и выше), а на перемешанных связях они рушатся. Двумерность")
    print("  тут существенна: одной большой корреляции мало, она означала бы, что")
    print("  восстановлена одна ось поля зрения из двух.")
    if not df.empty:
        print(f"\n  канонические корреляции с листом сомы: "
              f"{df['cc1_vs_soma_sheet'].mean():.3f} и {df['cc2_vs_soma_sheet'].mean():.3f}")
        print(f"  они же на перемешанных связях:         "
              f"{df['cc1_shuffled'].mean():.3f} и {df['cc2_shuffled'].mean():.3f}")


if __name__ == "__main__":
    main()
