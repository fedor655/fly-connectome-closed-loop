"""Разбор аннотаций типов клеток FlyWire.

Зачем. Структурный поиск сенсорных популяций провалился (tools/find_sensory_inputs.py):
признак «нулевая входная степень» не опознал ни одного из 21 известного Sugar GRN.
Единственный способ двинуться дальше — настоящие аннотации.

Источник: Schlegel et al., Nature 2024, «Whole-brain annotation and multi-connectome
cell typing of Drosophila», репозиторий flyconnectome/flywire_annotations.

Что проверяем:
  1. Совпадают ли ID аннотаций с нашим списком нейронов (валидация версии данных).
  2. Действительно ли наши P9 помечены как нисходящие. Если нет — вся архитектура
     стояла на неверном допущении, и это надо знать.
  3. Сколько в мозге нисходящих нейронов помимо P9 (кандидаты в более широкий выход).
  4. Есть ли ВОСХОДЯЩИЕ нейроны. Это принципиально: именно они несут в мозг
     информацию от ног из вентрального тяжа, которого в датасете нет. Если они
     аннотированы, то обратную связь от лапок биологически правильно подавать
     именно в них, а не в подобранных вручную партнёров P9.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flypaths import ANNOTATIONS, add_fly_brain_to_path, out  # noqa: E402

add_fly_brain_to_path()
from benchmark import EXPERIMENTS, path_comp  # noqa: E402

P9_LEFT = 720575940627652358
P9_RIGHT = 720575940635872101

pd.set_option("display.width", 200)


def main():
    print("=" * 78)
    print(" АННОТАЦИИ ТИПОВ КЛЕТОК FLYWIRE")
    print("=" * 78)

    if not ANNOTATIONS.exists():
        raise FileNotFoundError(
            f"не найдено: {ANNOTATIONS}\n"
            "Скачать: raw.githubusercontent.com/flyconnectome/flywire_annotations/"
            "main/supplemental_files/Supplemental_file1_neuron_annotations.tsv"
        )

    ann = pd.read_csv(ANNOTATIONS, sep="\t", low_memory=False)
    print(f"строк в аннотациях: {len(ann)}")

    comp = pd.read_csv(path_comp, index_col=0)
    our_ids = set(int(x) for x in comp.index)
    print(f"нейронов в нашей модели: {len(our_ids)}")

    ann["root_id"] = pd.to_numeric(ann["root_id"], errors="coerce")
    ann = ann.dropna(subset=["root_id"])
    ann["root_id"] = ann["root_id"].astype("int64")

    matched = ann[ann["root_id"].isin(our_ids)]
    print(f"совпало по root_id: {len(matched)} "
          f"({100 * len(matched) / len(our_ids):.1f}% нашей модели)")
    if len(matched) / len(our_ids) < 0.9:
        print("  ВНИМАНИЕ: совпадение низкое, вероятно разные версии данных")

    # ---------- 1. распределение по классам ----------
    print("\n===== super_class (только нейроны нашей модели) =====")
    sc = matched["super_class"].fillna("(нет)").value_counts()
    for name, cnt in sc.items():
        print(f"  {name:<24s} {cnt:>7d}")

    print("\n===== flow =====")
    for name, cnt in matched["flow"].fillna("(нет)").value_counts().items():
        print(f"  {name:<24s} {cnt:>7d}")

    # ---------- 2. проверка P9 ----------
    print("\n===== ПРОВЕРКА: чем на самом деле являются наши P9 =====")
    cols = ["root_id", "flow", "super_class", "cell_class", "cell_type",
            "hemibrain_type", "side", "top_nt"]
    for label, nid in (("P9 left", P9_LEFT), ("P9 right", P9_RIGHT)):
        row = ann[ann["root_id"] == nid]
        if row.empty:
            print(f"  {label}: в аннотациях НЕ НАЙДЕН")
            continue
        r = row.iloc[0]
        print(f"  {label} ({nid}):")
        for c in cols[1:]:
            print(f"      {c:<16s} {r.get(c)}")

    # ---------- 3. нисходящие нейроны ----------
    desc = matched[matched["super_class"] == "descending"]
    print(f"\n===== нисходящие нейроны (выход из мозга): {len(desc)} =====")
    if len(desc):
        print("  по стороне:", dict(desc["side"].fillna("(нет)").value_counts()))
        top = desc["cell_type"].fillna("(без типа)").value_counts().head(15)
        print("  частые типы:")
        for name, cnt in top.items():
            print(f"    {name:<22s} {cnt:>5d}")

    # ---------- 4. восходящие нейроны ----------
    asc = matched[matched["super_class"] == "ascending"]
    print(f"\n===== ВОСХОДЯЩИЕ нейроны (вход из вентрального тяжа): {len(asc)} =====")
    if len(asc):
        print("  по стороне:", dict(asc["side"].fillna("(нет)").value_counts()))
        print("  по нерву:", dict(asc["nerve"].fillna("(нет)").value_counts().head(8)))
        top = asc["cell_type"].fillna("(без типа)").value_counts().head(15)
        print("  частые типы:")
        for name, cnt in top.items():
            print(f"    {name:<22s} {cnt:>5d}")
        print("\n  Это биологически правильная точка входа для обратной связи от ног:")
        print("  механосенсоры ног сидят в вентральном тяже, а в мозг их сигнал")
        print("  приходит именно через восходящие нейроны.")

    # ---------- 5. сенсорные ----------
    sens = matched[matched["super_class"] == "sensory"]
    print(f"\n===== сенсорные нейроны: {len(sens)} =====")
    if len(sens):
        print("  по классам:")
        for name, cnt in sens["cell_class"].fillna("(нет)").value_counts().head(12).items():
            print(f"    {name:<24s} {cnt:>6d}")

    # проверка признака на известных Sugar GRNs
    sugar_ids = EXPERIMENTS["sugar"]["neu_exc"]
    sug = ann[ann["root_id"].isin(sugar_ids)]
    print(f"\n  контроль на 21 известном Sugar GRN: найдено {len(sug)}")
    if len(sug):
        print(f"    super_class: {dict(sug['super_class'].fillna('(нет)').value_counts())}")
        print(f"    cell_class:  {dict(sug['cell_class'].fillna('(нет)').value_counts())}")
        print(f"    cell_type:   {dict(sug['cell_type'].fillna('(нет)').value_counts())}")

    # ---------- сохранение ----------
    keep = ["root_id", "flow", "super_class", "cell_class", "cell_sub_class",
            "cell_type", "hemibrain_type", "side", "nerve", "top_nt"]
    keep = [c for c in keep if c in matched.columns]
    out_csv = out("annotations_matched.csv")
    matched[keep].to_csv(out_csv, index=False)
    print(f"\nсохранено (только нейроны нашей модели): {out_csv}")

    for name, subset in (("descending", desc), ("ascending", asc)):
        if len(subset):
            p = out(f"neurons_{name}.csv")
            subset[keep].to_csv(p, index=False)
            print(f"сохранено: {p}  ({len(subset)} нейронов)")


if __name__ == "__main__":
    main()
