"""Ложатся ли колончатые нейроны медуллы во FlyWire в правильную решётку?

Зачем. Для гибрида надо знать, какому нейрону FlyWire отдать активность какой
колонки flyvis. У flyvis колонка задана явно — осевые координаты (u, v) в
шестиугольнике радиуса 15, ровно 721 штука на тип. У FlyWire такого поля нет,
есть только координаты точки нейрона. Вопрос: восстанавливается ли решётка из
координат.

Почему это не праздный вопрос. Попытка построить ретинотопию по LC уже
провалилась: их точки сидят у сомы в изогнутой корковой оболочке и легли почти в
линию, доли дисперсии 0.913 / 0.078 / 0.008. У фоторецепторов вышло наоборот
хорошо, 0.709 / 0.249 / 0.041. Медулла колончатая по строению, но проверять надо
её саму, а не рассуждать по аналогии.

Проверяем по каждому типу и стороне:
  1. сколько нейронов против 721 колонки flygym и flyvis;
  2. доли дисперсии по главным осям — это лист или это линия;
  3. регулярность решётки: у шестиугольной решётки первые шесть соседей лежат
     примерно на одном расстоянии, поэтому отношение шестого к первому близко
     к единице, а разброс первого расстояния мал.

Единицы обязательны к приведению: координаты FlyWire — воксели FAFB, а он
анизотропный, x и y по 4 нм, z по 40 нм. Без приведения любая геометрия врёт.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flypaths import ANNOTATIONS, add_fly_brain_to_path, out  # noqa: E402

add_fly_brain_to_path()
from benchmark import path_comp  # noqa: E402

VOXEL_NM = np.array([4.0, 4.0, 40.0])
HEX_EXTENT = 15
N_COLUMNS = 3 * HEX_EXTENT * (HEX_EXTENT + 1) + 1   # 721

# Типы, которые есть и у flyvis, и во FlyWire, и которые по строению колончатые.
# Mi1 и Tm — классические «одна клетка на колонку». T4/T5 по четыре на колонку
# (a, b, c, d — четыре направления), поэтому каждый подтип отдельно.
TYPES = ["L1", "L2", "L3", "L5", "Mi1", "Mi4", "Mi9", "Tm1", "Tm2", "Tm9", "Tm20",
         "T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]


def sheet_stats(p_nm):
    """Главные оси, доли дисперсии и регулярность решётки в плоскости листа."""
    c = p_nm - p_nm.mean(0)
    _, s, vt = np.linalg.svd(c, full_matrices=False)
    var = s ** 2 / (s ** 2).sum()
    xy = c @ vt[:2].T                      # проекция на плоскость листа
    # расстояния до соседей
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    d.sort(axis=1)
    nn1 = d[:, 0]
    nn6 = d[:, 5]
    thick = np.abs(c @ vt[2]).std()        # толщина листа
    return {
        "var1": var[0], "var2": var[1], "var3": var[2],
        "nn1_um": np.median(nn1) / 1000.0,
        "nn1_cv": nn1.std() / max(nn1.mean(), 1e-9),
        "nn6_over_nn1": np.median(nn6) / max(np.median(nn1), 1e-9),
        "thickness_um": thick / 1000.0,
        "extent_um": np.sqrt(((xy ** 2).sum(1)).max()) / 1000.0,
    }


def main():
    print("=" * 78)
    print(" РЕТИНОТОПИЯ МЕДУЛЛЫ: восстанавливается ли решётка колонок")
    print("=" * 78)

    comp = pd.read_csv(path_comp, index_col=0)
    our = set(int(x) for x in comp.index)

    ann = pd.read_csv(ANNOTATIONS, sep="\t", low_memory=False)
    ann["root_id"] = pd.to_numeric(ann["root_id"], errors="coerce")
    ann = ann.dropna(subset=["root_id"])
    ann["root_id"] = ann["root_id"].astype("int64")
    ann = ann[ann["root_id"].isin(our)]
    ann = ann[ann[["pos_x", "pos_y", "pos_z"]].notna().all(axis=1)]
    ct = ann["cell_type"].fillna("")

    print(f"\nколонок в шестиугольнике радиуса {HEX_EXTENT}: {N_COLUMNS}")
    print("(столько же омматидиев у flygym и столько же колонок у flyvis)\n")

    print(f"  {'тип':<6s} {'сторона':>7s} {'нейронов':>9s} {'/721':>6s} "
          f"{'доли дисперсии':>22s} {'шаг,мкм':>8s} {'CV шага':>8s} "
          f"{'6-й/1-й':>8s} {'толщина,мкм':>12s}")
    rows = []
    for t in TYPES:
        for side in ("left", "right"):
            m = (ct == t) & (ann["side"] == side)
            if m.sum() < 50:
                continue
            p = ann.loc[m, ["pos_x", "pos_y", "pos_z"]].to_numpy(float) * VOXEL_NM
            st = sheet_stats(p)
            n = int(m.sum())
            print(f"  {t:<6s} {side:>7s} {n:>9d} {n / N_COLUMNS:>6.2f} "
                  f"{st['var1']:>7.3f}/{st['var2']:.3f}/{st['var3']:.3f} "
                  f"{st['nn1_um']:>8.2f} {st['nn1_cv']:>8.2f} "
                  f"{st['nn6_over_nn1']:>8.2f} {st['thickness_um']:>12.2f}")
            rows.append({"cell_type": t, "side": side, "n": n, **st})

    df = pd.DataFrame(rows)
    df.to_csv(out("medulla_retinotopy.csv"), index=False)
    print(f"\nсохранено: {out('medulla_retinotopy.csv')}")

    print("\n" + "=" * 78)
    print(" КАК ЧИТАТЬ")
    print("=" * 78)
    print("  Лист, а не линия: третья доля дисперсии мала, вторая сравнима с первой.")
    print("  Для сравнения: у LC вышло 0.913/0.078/0.008 — это линия, оттуда")
    print("  ретинотопию достать не удалось. У фоторецепторов 0.709/0.249/0.041.")
    print()
    print("  Решётка, а не облако: CV шага заметно меньше 0.5, отношение шестого")
    print("  соседа к первому близко к 1.0-1.3. У случайного облака CV около 0.5,")
    print("  а отношение шестого к первому около 2.4.")
    print()
    print("  Число нейронов около 721 или кратно ему говорит, что тип и правда")
    print("  колончатый и каждой колонке досталось по клетке.")

    if not df.empty:
        best = df.sort_values("var3").head(5)
        print("\n  пять самых плоских листов:")
        for _, r in best.iterrows():
            print(f"    {r['cell_type']:<6s} {r['side']:>6s}  "
                  f"{r['var1']:.3f}/{r['var2']:.3f}/{r['var3']:.3f}  "
                  f"n={int(r['n'])}, CV шага {r['nn1_cv']:.2f}, "
                  f"6/1 {r['nn6_over_nn1']:.2f}")


if __name__ == "__main__":
    main()
