"""Почему сигнал обрывается сразу за ламиной?

Измерено: стимуляция фоторецепторов даёт ламине 4.71 Гц, а медулле 0.01 Гц.
Это не плавное затухание, а обрыв. Причину надо установить, а не сочинить.

Три версии, каждая проверяется по данным:

  1. Связей ламина -> медулла в датасете мало или они слабые.
  2. Связи есть, но тормозные: тогда стимуляция не зажигает, а гасит.
  3. Связи есть и возбуждающие, но нейроны оптических долей в живой мухе
     не спайкуют, а работают градуальным потенциалом. Модель LIF передаёт
     только спайки, поэтому такой сигнал в ней не распространяется в принципе.

Третья версия для проекта самая важная: если верна она, то дело не в настройке,
а в границах применимости модели, и никакая проводка этого не исправит.
Различить версии можно по медиатору: гистамин у фоторецепторов и глутамат у
части нейронов ламины и медуллы — это тормозные, градуальные пути.
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


def main():
    print("=" * 78)
    print(" ДИАГНОЗ: где и почему обрывается зрительный путь")
    print("=" * 78)

    comp = pd.read_csv(path_comp, index_col=0)
    flyid2i = {int(j): i for i, j in enumerate(comp.index)}
    n = len(flyid2i)

    ann = pd.read_csv(ANNOTATIONS, sep="\t", low_memory=False)
    ann["root_id"] = pd.to_numeric(ann["root_id"], errors="coerce")
    ann = ann.dropna(subset=["root_id"])
    ann["root_id"] = ann["root_id"].astype("int64")
    ann = ann[ann["root_id"].isin(flyid2i.keys())]
    ct = ann["cell_type"].fillna("")

    groups = {
        "фоторецепторы": (ann["super_class"] == "sensory") & (ann["cell_class"] == "visual"),
        "ламина L1-L5": ct.str.match(r"^L[1-5]$"),
        "медулла Mi": ct.str.match(r"^Mi\d"),
        "медулла Tm": ct.str.match(r"^Tm\d"),
        "T4": ct.str.match(r"^T4"),
        "T5": ct.str.match(r"^T5"),
        "зрит. проекц.": ann["super_class"] == "visual_projection",
    }
    idx = {k: np.array([flyid2i[int(x)] for x in ann.loc[m, "root_id"]])
           for k, m in groups.items()}

    # ---------- медиаторы ----------
    print("\n----- медиаторы по ступеням -----")
    for k, m in groups.items():
        nt = ann.loc[m, "top_nt"].fillna("(нет)").value_counts()
        top = ", ".join(f"{a} {b}" for a, b in nt.head(3).items())
        print(f"  {k:<16s} {top}")

    print("\n  Гистамин и глутамат у дрозофилы — тормозные. Ацетилхолин — основной")
    print("  возбуждающий. Если путь идёт по тормозным медиаторам, стимуляция")
    print("  входа гасит выход, а не зажигает.")

    # ---------- связи между ступенями ----------
    conn = pd.read_parquet(path_con, columns=[
        "Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
    W = sp.coo_matrix(
        (conn["Excitatory x Connectivity"].to_numpy().astype(np.float32),
         (conn["Postsynaptic_Index"].to_numpy(), conn["Presynaptic_Index"].to_numpy())),
        shape=(n, n)).tocsr()
    del conn

    pairs = [
        ("фоторецепторы", "ламина L1-L5"),
        ("ламина L1-L5", "медулла Mi"),
        ("ламина L1-L5", "медулла Tm"),
        ("медулла Mi", "T4"),
        ("медулла Tm", "T5"),
        ("медулла Tm", "зрит. проекц."),
        ("T4", "зрит. проекц."),
        ("T5", "зрит. проекц."),
    ]
    print("\n----- связи между ступенями -----")
    print(f"  {'откуда -> куда':<34s} {'связей':>9s} {'сумма |вес|':>12s} "
          f"{'алгебр. сумма':>14s} {'доля торм.':>11s}")
    rows = []
    for a, b in pairs:
        src, dst = idx[a], idx[b]
        sub = W[dst][:, src]
        coo = sub.tocoo()
        if coo.nnz == 0:
            print(f"  {a + ' -> ' + b:<34s} {'0':>9s}")
            continue
        d = coo.data
        neg = float((d < 0).sum()) / len(d)
        print(f"  {a + ' -> ' + b:<34s} {coo.nnz:>9d} {np.abs(d).sum():>12.0f} "
              f"{d.sum():>+14.0f} {neg:>11.1%}")
        rows.append({"from": a, "to": b, "n_syn": int(coo.nnz),
                     "abs_weight": float(np.abs(d).sum()),
                     "net_weight": float(d.sum()), "frac_inhibitory": neg})

    # ---------- схождение входов: сколько нужно, чтобы нейрон выстрелил ----------
    print("\n" + "=" * 78)
    print(" СХОЖДЕНИЕ ВХОДОВ И ПОРОГ")
    print("=" * 78)
    print("  В модели прибавка к потенциалу за один пресинаптический спайк равна")
    print("  wScale * time_factor_mem * вес = 0.275 * 0.005 * вес = 0.001375 * вес.")
    print("  От покоя -52 мВ до порога -45 мВ нужно 7 мВ. Значит нейрону нужно")
    print("  накопить суммарно 7 / 0.001375 ≈ 5091 единиц «вес * спайк» за время")
    print("  мембраны (20 мс), иначе он не выстрелит никогда.\n")

    gain = 0.275 * 0.005
    need = 7.0 / gain
    print(f"  {'ступень':<16s} {'нейронов':>9s} {'входов на нейрон':>17s} "
          f"{'сумма весов входа':>18s} {'нужна частота входа':>20s}")
    conv_rows = []
    for k, ids in idx.items():
        if len(ids) == 0:
            continue
        sub = W[ids]
        in_deg = np.diff(sub.indptr)
        w_in = np.abs(sub).sum(axis=1).A.ravel()
        med_deg, med_w = float(np.median(in_deg)), float(np.median(w_in))
        # какая частота на каждом входе нужна, чтобы за 20 мс набрать порог
        req_hz = need / max(med_w, 1e-9) / 0.020
        print(f"  {k:<16s} {len(ids):>9d} {med_deg:>17.0f} {med_w:>18.0f} "
              f"{req_hz:>20.0f}")
        conv_rows.append({"stage": k, "n": len(ids), "median_in_degree": med_deg,
                          "median_input_weight": med_w, "required_input_hz": req_hz})

    print("\n  Последний столбец — какая частота должна быть у ВСЕХ входов нейрона,")
    print("  чтобы он достиг порога. Физиологический потолок около 200-400 Гц.")
    print("  Где требуемая частота на порядок выше — там спайковая модель")
    print("  сигнал не проведёт, сколько связей ни рисуй.")

    pd.DataFrame(conv_rows).to_csv(out("optic_lobe_convergence.csv"), index=False)
    pd.DataFrame(rows).to_csv(out("optic_lobe_diagnosis.csv"), index=False)
    print(f"\nсохранено: {out('optic_lobe_diagnosis.csv')}")
    print("\nЧитать так: если связи есть и их много, но алгебраическая сумма")
    print("отрицательная — путь тормозный, и спайковая модель по нему сигнал")
    print("не проведёт. Это граница применимости модели, а не ошибка настройки.")


if __name__ == "__main__":
    main()
