"""Можно ли построить ретинотопию LC9 по координатам из аннотаций?

Сейчас весь глаз сводится к одному числу, и все LC9 своей стороны получают
одинаковую стимуляцию. Настоящие LC ретинотопичны: каждый смотрит в свой
участок поля зрения. В аннотациях FlyWire есть координаты pos_x/pos_y/pos_z.

Проверяем по порядку:
  1. Образуют ли LC9 одной стороны двумерный лист. Если третья главная
     компонента сравнима с первыми двумя, это облако, а не лист, и
     ретинотопию по этим координатам строить нельзя.
  2. Насколько равномерно они покрывают этот лист.
  3. Строим отображение омматидиев на LC и смотрим, сколько омматидиев
     приходится на нейрон.

Честная оговорка, которую надо держать в голове: pos_x/y/z — это
представительная точка на скелете нейрона, а НЕ центр его рецептивного поля.
Ориентация листа относительно поля зрения (где верх, где перёд) из наших
данных не выводится. Поэтому получить можно только топографию «соседние
омматидии идут на соседние LC», а не привязку к настоящим направлениям взгляда.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flypaths import ANNOTATIONS, OUTPUT_DIR, add_fly_brain_to_path  # noqa: E402

add_fly_brain_to_path()
from benchmark import path_comp  # noqa: E402

CELL_TYPES = ["LC9", "LC31a", "LC31b", "LCe04"]


def analyze_sheet(pos, label):
    """Насколько облако точек похоже на двумерный лист."""
    c = pos - pos.mean(axis=0)
    u, s, vt = np.linalg.svd(c, full_matrices=False)
    var = s ** 2 / len(c)
    frac = var / var.sum()
    print(f"\n  --- {label}: {len(pos)} нейронов ---")
    print(f"    доли дисперсии по главным осям: "
          f"{frac[0]:.3f} / {frac[1]:.3f} / {frac[2]:.3f}")
    flatness = frac[2] / max(frac[1], 1e-9)
    print(f"    сплюснутость (3-я ось к 2-й): {flatness:.3f} "
          f"({'лист' if flatness < 0.35 else 'облако'})")

    xy = c @ vt[:2].T
    span_x = xy[:, 0].max() - xy[:, 0].min()
    span_y = xy[:, 1].max() - xy[:, 1].min()
    print(f"    размах листа: {span_x:.0f} x {span_y:.0f} (нм координат FlyWire)")

    # равномерность покрытия: расстояние до ближайшего соседа
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    print(f"    до ближайшего соседа: медиана {np.median(nn):.0f}, "
          f"разброс {nn.std():.0f}, максимум {nn.max():.0f}")
    print(f"    неравномерность (максимум к медиане): "
          f"{nn.max() / max(np.median(nn), 1e-9):.1f}")
    return xy, flatness


def main():
    print("=" * 78)
    print(" РЕТИНОТОПИЯ LC ПО КООРДИНАТАМ АННОТАЦИЙ")
    print("=" * 78)

    comp = pd.read_csv(path_comp, index_col=0)
    our_ids = set(int(x) for x in comp.index)

    ann = pd.read_csv(ANNOTATIONS, sep="\t", low_memory=False)
    ann["root_id"] = pd.to_numeric(ann["root_id"], errors="coerce")
    ann = ann.dropna(subset=["root_id"])
    ann["root_id"] = ann["root_id"].astype("int64")
    ann = ann[ann["root_id"].isin(our_ids)]

    have_pos = ann[["pos_x", "pos_y", "pos_z"]].notna().all(axis=1)
    print(f"нейронов с координатами: {int(have_pos.sum())} из {len(ann)}")

    results = []
    for ct in CELL_TYPES:
        sub = ann[(ann["cell_type"] == ct) & have_pos]
        if len(sub) < 8:
            print(f"\n  {ct}: слишком мало нейронов с координатами ({len(sub)})")
            continue
        for side in ("left", "right"):
            s = sub[sub["side"] == side]
            if len(s) < 8:
                continue
            pos = s[["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=float)
            xy, flat = analyze_sheet(pos, f"{ct} {side}")
            results.append({"cell_type": ct, "side": side, "n": len(s),
                            "flatness": flat})

    print("\n" + "=" * 78)
    print(" ИТОГ")
    print("=" * 78)
    if results:
        df = pd.DataFrame(results)
        print(df.to_string(index=False))
        df.to_csv(OUTPUT_DIR / "lc_retinotopy.csv", index=False)
        print(f"\nсохранено: {OUTPUT_DIR / 'lc_retinotopy.csv'}")

        ok = df[df["flatness"] < 0.35]
        print(f"\n  популяций, похожих на плоский лист: {len(ok)} из {len(df)}")
        print("\n  Читать так: если сплюснутость мала, координаты задают")
        print("  двумерную карту, и омматидии можно раскладывать по ней.")
        print("  Ориентация карты относительно поля зрения из этих данных не")
        print("  выводится, поэтому речь только о топографии соседства.")

    weight_vs_position(ann)


def weight_vs_position(ann):
    """Есть ли вообще смысл в позиции LC для нашего выхода?

    Даже если карту построить, она повлияет на поведение только тогда, когда
    вес связи LC -> DNp09 зависит от положения нейрона на карте. Если все LC9
    давят на DNp09 одинаково, то различать участки поля зрения этим выходом
    невозможно в принципе, и ретинотопия ничего не даст.

    Меряем прямо: корреляцию между положением LC9 вдоль главной оси и его
    весом на ипсилатеральный DNp09.
    """
    import scipy.sparse as sp
    from benchmark import path_con

    print("\n" + "=" * 78)
    print(" ЗАВИСИТ ЛИ ВЕС LC9 -> DNp09 ОТ ПОЛОЖЕНИЯ НЕЙРОНА")
    print("=" * 78)

    DNP09 = {"left": 720575940635872101, "right": 720575940627652358}

    comp = pd.read_csv(path_comp, index_col=0)
    flyid2i = {int(j): i for i, j in enumerate(comp.index)}
    n = len(flyid2i)

    conn = pd.read_parquet(path_con, columns=[
        "Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
    W = sp.coo_matrix(
        (conn["Excitatory x Connectivity"].to_numpy().astype(np.float32),
         (conn["Postsynaptic_Index"].to_numpy(), conn["Presynaptic_Index"].to_numpy())),
        shape=(n, n)).tocsr()
    del conn

    for side, dn_id in DNP09.items():
        sub = ann[(ann["cell_type"] == "LC9") & (ann["side"] == side)]
        sub = sub[sub[["pos_x", "pos_y", "pos_z"]].notna().all(axis=1)]
        if len(sub) < 8:
            continue
        pos = sub[["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=float)
        c = pos - pos.mean(axis=0)
        _, _, vt = np.linalg.svd(c, full_matrices=False)
        axis1 = c @ vt[0]

        dn_idx = flyid2i[dn_id]
        row = W.getrow(dn_idx)
        weights = np.array([float(row[0, flyid2i[int(r)]]) for r in sub["root_id"]])

        connected = weights != 0
        print(f"\n  --- LC9 {side} -> DNp09 {side} ---")
        print(f"    нейронов: {len(sub)}, из них связаны с DNp09: {int(connected.sum())}")
        if connected.sum() < 5:
            print("    связей слишком мало для вывода")
            continue
        w_c = weights[connected]
        a_c = axis1[connected]
        print(f"    веса: минимум {w_c.min():.0f}, максимум {w_c.max():.0f}, "
              f"среднее {w_c.mean():.1f}, разброс {w_c.std():.1f}")
        r = float(np.corrcoef(a_c, w_c)[0, 1])
        print(f"    корреляция положения с весом: {r:+.3f}")
        # разброс весов сам по себе: если он мал, различать нечего
        cv = w_c.std() / max(abs(w_c.mean()), 1e-9)
        print(f"    относительный разброс весов: {cv:.2f}")

    print("\n  Читать так: заметная корреляция означала бы, что положение LC")
    print("  задаёт силу его влияния на DNp09, и ретинотопия давала бы")
    print("  позиционную избирательность. Корреляция около нуля означает, что")
    print("  этим выходом участки поля зрения не различить, сколько карту ни строй.")


if __name__ == "__main__":
    main()
