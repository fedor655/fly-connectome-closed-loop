"""Что из зрительного пути мухи есть в коннектоме и что мы из этого используем.

Вопрос простой: может ли эта муха видеть полноценно. Сейчас каждый глаз
сводится к ОДНОМУ числу — средней яркости. Надо понять, чем мы при этом
пренебрегаем: отсутствует ли остальное в данных или просто не подключено.

Считаем состав зрительной части коннектома по классам аннотаций и сверяем
с тем, что даёт симулятор тела (721 омматидий на глаз).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flypaths import ANNOTATIONS, add_fly_brain_to_path  # noqa: E402

add_fly_brain_to_path()
from benchmark import path_comp  # noqa: E402

OMMATIDIA_PER_EYE = 721        # из flygym, измерено в tools/vision_smoke_check.py


def main():
    print("=" * 78)
    print(" АУДИТ ЗРИТЕЛЬНОГО ПУТИ")
    print("=" * 78)

    comp = pd.read_csv(path_comp, index_col=0)
    our = set(int(x) for x in comp.index)

    ann = pd.read_csv(ANNOTATIONS, sep="\t", low_memory=False)
    ann["root_id"] = pd.to_numeric(ann["root_id"], errors="coerce")
    ann = ann.dropna(subset=["root_id"])
    ann["root_id"] = ann["root_id"].astype("int64")
    ann = ann[ann["root_id"].isin(our)]
    print(f"нейронов модели с аннотацией: {len(ann)}")

    print("\n----- сенсорные нейроны по классам -----")
    sens = ann[ann["super_class"] == "sensory"]
    for cls, cnt in sens["cell_class"].fillna("(нет)").value_counts().items():
        print(f"  {cls:<22s} {cnt:>7d}")

    vis_sens = sens[sens["cell_class"] == "visual"]
    print(f"\n----- зрительные сенсорные (фоторецепторы): {len(vis_sens)} -----")
    print("  по стороне:", dict(vis_sens["side"].fillna("(нет)").value_counts()))
    print("  по типам:")
    for t, cnt in vis_sens["cell_type"].fillna("(без типа)").value_counts().head(12).items():
        print(f"    {t:<18s} {cnt:>7d}")

    per_eye = len(vis_sens) / 2
    print(f"\n  фоторецепторов на глаз: {per_eye:.0f}")
    print(f"  омматидиев в симуляторе тела: {OMMATIDIA_PER_EYE}")
    print(f"  фоторецепторов на омматидий: {per_eye / OMMATIDIA_PER_EYE:.2f}")
    print("  (у дрозофилы в омматидии 8 фоторецепторов R1-R8)")

    print("\n----- прочие зрительные классы -----")
    for sc in ("optic", "visual_projection", "visual_centrifugal"):
        g = ann[ann["super_class"] == sc]
        if len(g):
            print(f"  {sc:<22s} {len(g):>7d}")
    # внутренние нейроны оптических долей часто помечены как central с
    # соответствующими типами, поэтому смотрим ещё и по названиям
    optic_like = ann[ann["cell_type"].fillna("").str.match(r"^(L[1-5]|Mi\d|Tm\d|T4|T5|C[23]|Dm\d)")]
    print(f"  нейроны оптических долей по типам (L, Mi, Tm, T4, T5, Dm): {len(optic_like)}")

    print("\n" + "=" * 78)
    print(" ЧТО ИСПОЛЬЗУЕТСЯ СЕЙЧАС")
    print("=" * 78)
    vp = ann[ann["super_class"] == "visual_projection"]
    print(f"  вход подаётся в: зрительные проекционные, {len(vp)} нейронов")
    print(f"  на каждую сторону подаётся ОДНО число (средняя яркость глаза)")
    print(f"  фоторецепторы ({len(vis_sens)}) не используются вовсе")
    print(f"  вся обработка оптических долей обходится стороной")
    print(f"\n  размерность входа сейчас:      2 числа")
    print(f"  доступно в симуляторе тела:    {OMMATIDIA_PER_EYE * 2} омматидиев x 2 канала")
    print(f"  доступно в коннектоме:         {len(vis_sens)} фоторецепторов")


if __name__ == "__main__":
    main()
