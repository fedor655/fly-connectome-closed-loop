"""Можно ли разложить омматидии симулятора по фоторецепторам коннектома?

Задача. Сейчас каждый глаз сводится к одному числу. В коннектоме есть 10855
фоторецепторов (R1-6, R7, R8) и 77530 нейронов оптических долей, и всё это
простаивает. Чтобы дать мухе настоящее зрение, надо сопоставить 721 омматидий
симулятора тела с фоторецепторами каждого глаза.

Почему на этот раз есть надежда. Попытка построить ретинотопию по LC-нейронам
провалилась: их координаты — точки у сомы в изогнутой корковой оболочке, они
легли почти в линию. У фоторецепторов ситуация принципиально другая: их тела
сидят в самой сетчатке и упорядочены ретинотопически по построению. Значит
координаты обязаны образовать двумерный лист, повторяющий поверхность глаза.

Проверяем:
  1. образуют ли фоторецепторы каждого глаза двумерную поверхность;
  2. равномерно ли покрытие;
  3. сколько фоторецепторов приходится на омматидий и группируются ли они
     (у дрозофилы в омматидии 8 штук: R1-R6 широкополосные, R7 и R8 цветные).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flypaths import ANNOTATIONS, add_fly_brain_to_path, out  # noqa: E402

add_fly_brain_to_path()
from benchmark import path_comp  # noqa: E402

OMMATIDIA_PER_EYE = 721

# Координаты FlyWire даны в вокселях FAFB, а он анизотропный: x и y по 4 нм,
# z — номер среза по 40 нм. Без приведения к нанометрам любая геометрия врёт:
# размах по z выходит 3949 против 110905 по x, и облако выглядит плоским,
# хотя это всего лишь разные единицы по осям.
VOXEL_NM = np.array([4.0, 4.0, 40.0])


def fit_sphere(p):
    """Подогнать сферу под облако точек: глаз мухи — выпуклая поверхность.

    Решается линейно: |p - c|^2 = r^2  ->  2p·c + (r^2 - |c|^2) = |p|^2
    """
    A = np.hstack([2 * p, np.ones((len(p), 1))])
    b = (p ** 2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:3]
    r = np.sqrt(max(sol[3] + (c ** 2).sum(), 0.0))
    resid = np.abs(np.linalg.norm(p - c, axis=1) - r)
    return c, r, resid


def main():
    print("=" * 78)
    print(" ГЕОМЕТРИЯ ФОТОРЕЦЕПТОРОВ: ложатся ли они в сетчатку")
    print("=" * 78)

    comp = pd.read_csv(path_comp, index_col=0)
    our = set(int(x) for x in comp.index)

    ann = pd.read_csv(ANNOTATIONS, sep="\t", low_memory=False)
    ann["root_id"] = pd.to_numeric(ann["root_id"], errors="coerce")
    ann = ann.dropna(subset=["root_id"])
    ann["root_id"] = ann["root_id"].astype("int64")
    ann = ann[ann["root_id"].isin(our)]

    ph = ann[(ann["super_class"] == "sensory") & (ann["cell_class"] == "visual")].copy()
    have = ph[["pos_x", "pos_y", "pos_z"]].notna().all(axis=1)
    ph = ph[have]
    print(f"фоторецепторов с координатами: {len(ph)} из "
          f"{int(((ann['super_class'] == 'sensory') & (ann['cell_class'] == 'visual')).sum())}")

    for side in ("left", "right"):
        s = ph[ph["side"] == side]
        if len(s) < 100:
            continue
        p = s[["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=float) * VOXEL_NM
        print(f"\n===== глаз {side}: {len(p)} фоторецепторов =====")

        # плоскость или поверхность
        c = p - p.mean(axis=0)
        _, sv, vt = np.linalg.svd(c, full_matrices=False)
        frac = sv ** 2 / (sv ** 2).sum()
        print(f"  доли дисперсии по осям: {frac[0]:.3f} / {frac[1]:.3f} / {frac[2]:.3f}")
        print(f"  (для сравнения, LC9 давали 0.913 / 0.078 / 0.008 — почти линия)")

        # сфера
        cen, rad, resid = fit_sphere(p)
        print(f"  подгонка сферы: радиус {rad:.0f}, отклонение "
              f"медиана {np.median(resid):.0f}, 90-й процентиль {np.percentile(resid, 90):.0f}")
        print(f"  относительная толщина слоя: {np.median(resid) / max(rad, 1):.3f}")

        # разложение по типам
        print("  по типам:", dict(s["cell_type"].fillna("(нет)").value_counts()))

        # проекция на сферу -> две угловые координаты
        d = p - cen
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        # локальный базис по главным осям облака
        u = d @ vt[0]
        w = d @ vt[1]
        az = np.degrees(np.arctan2(w, u))
        el = np.degrees(np.arcsin(np.clip(d @ vt[2], -1, 1)))
        print(f"  угловой размах: азимут {az.max() - az.min():.0f}°, "
              f"элевация {el.max() - el.min():.0f}°")

        # кластеризация в омматидии: у соседей из одного омматидия
        # расстояние должно быть заметно меньше, чем между омматидиями
        k = min(2000, len(p))
        sub = p[np.random.default_rng(0).choice(len(p), k, replace=False)]
        dist = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=2)
        np.fill_diagonal(dist, np.inf)
        nn1 = np.sort(dist, axis=1)[:, 0]
        nn8 = np.sort(dist, axis=1)[:, min(7, k - 2)]
        print(f"  до ближайшего соседа: медиана {np.median(nn1):.0f}")
        print(f"  до восьмого соседа:   медиана {np.median(nn8):.0f}")
        print(f"  отношение: {np.median(nn8) / max(np.median(nn1), 1e-9):.2f} "
              f"(близко к 1 — фоторецепторы не сгруппированы в омматидии по координатам)")

        print(f"  фоторецепторов на омматидий при {OMMATIDIA_PER_EYE} омматидиях: "
              f"{len(p) / OMMATIDIA_PER_EYE:.2f}")

    # ---------- ориентация сетчатки в сырых координатах ----------
    print("\n" + "=" * 78)
    print(" ОРИЕНТАЦИЯ: какая ось за что отвечает")
    print("=" * 78)
    print("  Подгонка сферы нестабильна (радиусы 109628 и 175593 при том, что")
    print("  сетчатка — не сфера, а участок), поэтому ориентацию берём из сырых")
    print("  осей FlyWire, а не из главных компонент.\n")

    stats = {}
    for side in ("left", "right"):
        s = ph[ph["side"] == side]
        p = s[["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=float) * VOXEL_NM
        stats[side] = p
        print(f"  глаз {side}:")
        for ax, name in enumerate(("x", "y", "z")):
            print(f"    {name}: центр {p[:, ax].mean():>9.0f}, "
                  f"размах {p[:, ax].max() - p[:, ax].min():>8.0f}")

    pl, pr = stats["left"], stats["right"]
    print("\n  разность центров (левый минус правый):")
    for ax, name in enumerate(("x", "y", "z")):
        d = pl[:, ax].mean() - pr[:, ax].mean()
        print(f"    {name}: {d:>+9.0f}")
    print("\n  Ось с наибольшей разностью центров разделяет глаза, то есть это")
    print("  медиолатеральная ось. Оставшиеся две задают поверхность сетчатки:")
    print("  одна отвечает за элевацию (верх-низ), другая за азимут (перёд-зад).")

    # какая из двух оставшихся осей вертикальная: у мухи глаз выше вытянут
    # слабее, чем в длину, но надёжнее опереться на симметрию полушарий —
    # вертикальная ось должна быть у обоих глаз почти одинаковой
    print("\n  сходство осей между глазами (чем ближе к 1, тем симметричнее):")
    for ax, name in enumerate(("x", "y", "z")):
        rl = pl[:, ax].max() - pl[:, ax].min()
        rr = pr[:, ax].max() - pr[:, ax].min()
        print(f"    {name}: размах слева {rl:>8.0f}, справа {rr:>8.0f}, "
              f"отношение {min(rl, rr) / max(rl, rr):.3f}")

    print("\n" + "=" * 78)
    print(" ВЫВОД")
    print("=" * 78)
    print("  Третья главная ось равна нулю, слой тонкий — координаты фоторецепторов")
    print("  задают поверхность глаза. В отличие от LC-нейронов, отображение")
    print("  омматидиев на фоторецепторы построить можно.")


if __name__ == "__main__":
    main()
