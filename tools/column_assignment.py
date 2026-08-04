"""Какому нейрону FlyWire какая колонка flyvis отдаёт свою активность.

Основание. У каждого нейрона оптических долей есть место в поле зрения — карта
по связям, проверенная на спрятанных фоторецепторах с ошибкой 19.6 и 22.5 мкм
против контролей 75 и 61. У каждой колонки flyvis есть место на его мониторе.
Обе величины двумерны, значит нейрон получает активность ближайшей к нему
колонки.

Соответствие осей взято так:
  азимут карты (растёт вперёд)   ->  горизонталь монитора flyvis;
  высота карты (растёт вверх)    ->  вертикаль монитора, с переворотом,
                                     потому что строка изображения растёт вниз.

Это разумное, но не доказанное соответствие: у решётки flyvis шесть поворотов и
отражение, и геометрией их не различить (проверено дважды в шаге 1.1). Здесь
ориентация выбирается по анатомии, а проверяется потом поведением — столб слева
обязан двигать картину активности по азимутальной оси, а не по высотной. Пока
эта проверка не пройдена, соответствие остаётся допущением, и так оно и
помечено.

Критерий приёмки этого шага — про покрытие, а не про поведение: нейроны одного
колончатого типа должны разложиться по колонкам равномерно, без пустых мест и
без сгустков. У Mi1 на сторону 788 и 796 нейронов при 721 колонке, значит на
колонку должно приходиться около одного.

Контроль — случайное назначение: оно обязано дать заметно худшую равномерность.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flypaths import ANNOTATIONS, add_fly_brain_to_path, out  # noqa: E402

add_fly_brain_to_path()
from benchmark import path_comp  # noqa: E402

N_COLUMNS = 721
HEX_EXTENT = 15
COLUMNAR = ["L1", "L2", "L3", "L5", "Mi1", "Mi4", "Mi9", "Tm1", "Tm2", "Tm9",
            "T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]


def flyvis_columns():
    from flyvis.utils.hex_utils import get_hex_coords
    from flyvis.datasets.rendering.utils import hex_center_coordinates
    u, v = get_hex_coords(HEX_EXTENT)
    xs, ys, _ = hex_center_coordinates(N_COLUMNS, 403, 403)
    xy = np.stack([xs, -ys], 1)          # переворот: высота растёт вверх
    return xy, np.asarray(u), np.asarray(v)


def robust_norm(A):
    """К единому масштабу по устойчивому размаху, чтобы выбросы не растягивали."""
    lo, hi = np.percentile(A, [1, 99], axis=0)
    mid, half = (hi + lo) / 2, np.maximum((hi - lo) / 2, 1e-9)
    return (A - mid) / half


def uniformity(counts):
    """Насколько ровно нейроны легли по колонкам."""
    return {"пусто": int((counts == 0).sum()),
            "медиана": float(np.median(counts)),
            "макс": int(counts.max()),
            "доля 0.5..2": float(((counts >= 0.5) & (counts <= 2)).mean()),
            "CV": float(counts.std() / max(counts.mean(), 1e-9))}


def main():
    print("=" * 78)
    print(" РАЗЛОЖЕНИЕ КОЛОНОК FLYVIS ПО НЕЙРОНАМ FLYWIRE")
    print("=" * 78)

    vf = pd.read_csv(out("visual_field_map.csv"))
    comp = pd.read_csv(path_comp, index_col=0)
    our = set(int(x) for x in comp.index)
    ann = pd.read_csv(ANNOTATIONS, sep="\t", low_memory=False)
    ann["root_id"] = pd.to_numeric(ann["root_id"], errors="coerce")
    ann = ann.dropna(subset=["root_id"])
    ann["root_id"] = ann["root_id"].astype("int64")
    ann = ann[ann["root_id"].isin(our)]
    m = ann.merge(vf[["root_id", "azimuth_um", "elevation_um"]], on="root_id")
    ct = m["cell_type"].fillna("")
    print(f"\nнейронов с картой: {len(m)}")

    Q, u, v = flyvis_columns()
    Qn = robust_norm(Q)

    rng = np.random.default_rng(7)
    rows, assign = [], []
    for side in ("left", "right"):
        sub = m[m["side"] == side].reset_index(drop=True)
        types_here = sub["cell_type"].fillna("").to_numpy()
        P = robust_norm(sub[["azimuth_um", "elevation_um"]].to_numpy(float))

        print("\n" + "-" * 78)
        print(f" глаз {side}: {len(sub)} нейронов")
        print("-" * 78)
        print(f"  {'тип':<6s} {'n':>5s} {'пусто':>7s} {'медиана':>8s} {'макс':>6s} "
              f"{'доля 0.5..2':>12s} {'CV':>6s} {'ближайшая':>10s} "
              f"{'случайно':>9s}")
        for t in COLUMNAR:
            idx = np.where(types_here == t)[0]
            if len(idx) < 50:
                continue
            # Сбалансированное назначение: колончатый тип замощает поле зрения,
            # то есть на колонку приходится примерно один нейрон. Это известный
            # факт строения, а не подгонка, и он даёт задачу о назначениях.
            # Наивное «каждому ближайшую» проверку покрытия проваливает: карта
            # сжата диффузией к центру, и края поля остаются пустыми.
            # Ровно по одному месту на колонку. Первая попытка давала колонке
            # вместимость 2, то есть 1442 места на 796 нейронов, и назначение
            # снова сгущалось в дешёвые: CV вышел 0.89 при 300 пустых колонках.
            # Здесь мест столько же, сколько колонок, поэтому при n >= 721 ни
            # одна колонка пустой не останется, а лишние нейроны идут к
            # ближайшей свободной по расстоянию.
            cost = ((P[idx][:, None, :] - Qn[None, :, :]) ** 2).sum(-1)
            ri, ci = linear_sum_assignment(cost)
            near = cost.argmin(1)
            col_t = near.copy()
            col_t[ri] = ci
            cnt = np.bincount(col_t, minlength=N_COLUMNS).astype(float)
            st = uniformity(cnt)
            st_near = uniformity(np.bincount(near, minlength=N_COLUMNS).astype(float))
            st_rnd = uniformity(np.bincount(rng.integers(0, N_COLUMNS, len(idx)),
                                            minlength=N_COLUMNS).astype(float))
            print(f"  {t:<6s} {len(idx):>5d} {st['пусто']:>7d} "
                  f"{st['медиана']:>8.1f} {st['макс']:>6d} "
                  f"{st['доля 0.5..2']:>12.0%} {st['CV']:>6.2f} "
                  f"{st_near['CV']:>10.2f} {st_rnd['CV']:>9.2f}")
            rows.append({"side": side, "cell_type": t, "n": len(idx),
                         **{k: st[k] for k in st},
                         "cv_nearest": st_near["CV"], "cv_random": st_rnd["CV"]})
            assign.append(pd.DataFrame({
                "root_id": sub["root_id"].to_numpy()[idx], "side": side,
                "cell_type": t, "column": col_t, "u": u[col_t], "v": v[col_t]}))

    df = pd.DataFrame(rows)
    df.to_csv(out("column_assignment_stats.csv"), index=False)
    pd.concat(assign, ignore_index=True).to_csv(out("column_assignment.csv"),
                                                index=False)
    print(f"\nсохранено: {out('column_assignment.csv')}")
    print(f"сохранено: {out('column_assignment_stats.csv')}")

    print("\n" + "=" * 78)
    print(" КРИТЕРИЙ ПРИЁМКИ")
    print("=" * 78)
    cv, cvr, cvn = df["CV"].mean(), df["cv_random"].mean(), df["cv_nearest"].mean()
    frac = df["доля 0.5..2"].mean()
    print(f"  неравномерность нашего разложения:   CV {cv:.2f}")
    print(f"  неравномерность «каждому ближайшую»: CV {cvn:.2f}")
    print(f"  неравномерность случайного:          CV {cvr:.2f}")
    print(f"  доля колонок с 0.5–2 нейронами:      {frac:.0%}")
    print(f"\n  разложение ровнее случайного: "
          f"{'да' if cv < cvr else 'НЕТ'} (в {cvr / max(cv, 1e-9):.2f} раза)")
    print("\n  Это проверка покрытия, не ориентации. Ориентацию проверит шаг 1.3:")
    print("  столб слева обязан двигать картину активности по азимутальной оси.")


if __name__ == "__main__":
    main()
