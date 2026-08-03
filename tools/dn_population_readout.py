"""Какую популяцию нисходящих можно считывать вместо одного DNp09?

Зачем. Повторность показала: сквозного поведения нет, потому что команда
считывается с ОДНОГО нейрона на сторону. При 100 Гц за окно 15 мс это полтора
спайка, пуассоновский шум около 80 процентов, и курс мухи превращается в
случайное блуждание с разбросом 47 градусов.

Лекарство очевидно: усреднять по популяции. Шум падает как корень из числа
нейронов. Но брать популяцию из анатомических списков нельзя — уже дважды
выяснялось, что вес связи не равен функциональному влиянию (Sugar GRNs до P9,
DNg108 от восходящих). Поэтому меряем отклик ВСЕХ 1299 нисходящих на широкую
зрительную стимуляцию и отбираем тех, кто реально загорается.

Для каждого нисходящего считаем:
  - частоту при стимуляции своей и противоположной стороны;
  - индекс латеральности (своя минус чужая, делённое на сумму);
  - вклад в подавление шума, если включить его в популяцию.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flypaths import ANNOTATIONS, add_fly_brain_to_path, out  # noqa: E402

add_fly_brain_to_path()
from benchmark import path_comp, path_con, path_wt  # noqa: E402
import run_pytorch as rp  # noqa: E402

DT = 0.1
SIM_MS = 2000.0          # длиннее обычного: нужна устойчивая оценка частот
TRANSIENT_MS = 300.0
STIM_HZ = 150.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print("=" * 78)
    print(" ОТБОР ПОПУЛЯЦИИ НИСХОДЯЩИХ ДЛЯ СЧИТЫВАНИЯ КОМАНДЫ")
    print("=" * 78)

    comp = pd.read_csv(path_comp, index_col=0)
    flyid2i = {int(j): i for i, j in enumerate(comp.index)}
    n = len(flyid2i)

    ann = pd.read_csv(ANNOTATIONS, sep="\t", low_memory=False)
    ann["root_id"] = pd.to_numeric(ann["root_id"], errors="coerce")
    ann = ann.dropna(subset=["root_id"])
    ann["root_id"] = ann["root_id"].astype("int64")
    ann = ann[ann["root_id"].isin(flyid2i.keys())]

    vis = ann[ann["super_class"] == "visual_projection"]
    vis_l = [flyid2i[i] for i in vis.loc[vis["side"] == "left", "root_id"]]
    vis_r = [flyid2i[i] for i in vis.loc[vis["side"] == "right", "root_id"]]
    print(f"зрительных проекционных: слева {len(vis_l)}, справа {len(vis_r)}")

    desc = ann[ann["super_class"] == "descending"].copy()
    desc = desc[desc["side"].isin(["left", "right"])]
    dn_ids = desc["root_id"].tolist()
    dn_idx = [flyid2i[int(x)] for x in dn_ids]
    dn_side = desc["side"].tolist()
    dn_type = desc["cell_type"].fillna("(без типа)").tolist()
    print(f"нисходящих с известной стороной: {len(dn_idx)}")

    weights = rp.get_weights(str(path_con), str(path_comp), str(path_wt), csr=True).to(DEVICE)
    idx_t = torch.tensor(dn_idx, dtype=torch.long, device=DEVICE)

    def measure(stim):
        model = rp.TorchModel(1, n, DT, rp.MODEL_PARAMS, weights,
                              exc_indices=list(stim), device=DEVICE)
        cond, delay_buf, spikes, v, refrac = model.state_init()
        rates = torch.zeros(1, n, device=DEVICE)
        rates[:, stim] = STIM_HZ
        gen = torch.Generator(device=DEVICE)
        gen.manual_seed(2024)
        acc = torch.zeros(len(dn_idx), device=DEVICE)
        n_tr, n_me = int(TRANSIENT_MS / DT), int(SIM_MS / DT)
        with torch.no_grad():
            for step in range(n_tr + n_me):
                cond, delay_buf, spikes, v, refrac = model(
                    rates, cond, delay_buf, spikes, v, refrac, generator=gen)
                if step >= n_tr:
                    acc.add_(spikes[0, idx_t])
        return np.array(acc.tolist()) / (SIM_MS / 1000.0)

    t0 = time.perf_counter()
    print(f"\nстимулирую зрительные слева @ {STIM_HZ:.0f} Гц...")
    hz_stim_left = measure(vis_l)
    print(f"  {time.perf_counter() - t0:.0f} с")
    t0 = time.perf_counter()
    print(f"стимулирую зрительные справа @ {STIM_HZ:.0f} Гц...")
    hz_stim_right = measure(vis_r)
    print(f"  {time.perf_counter() - t0:.0f} с")

    df = pd.DataFrame({
        "root_id": dn_ids, "cell_type": dn_type, "side": dn_side,
        "hz_stim_left": hz_stim_left, "hz_stim_right": hz_stim_right,
    })
    df["hz_ipsi"] = np.where(df["side"] == "left", df["hz_stim_left"], df["hz_stim_right"])
    df["hz_contra"] = np.where(df["side"] == "left", df["hz_stim_right"], df["hz_stim_left"])
    tot = df["hz_ipsi"] + df["hz_contra"]
    df["lat_index"] = np.where(tot > 0, (df["hz_ipsi"] - df["hz_contra"]) / np.maximum(tot, 1e-9), 0.0)

    active = df[df["hz_ipsi"] > 5.0]
    print(f"\nотозвались (своя сторона > 5 Гц): {len(active)} из {len(df)}")

    good = active[active["lat_index"] > 0.5].sort_values("hz_ipsi", ascending=False)
    print(f"из них латерализованных (индекс > 0.5): {len(good)}")
    print(f"  слева {int((good['side'] == 'left').sum())}, "
          f"справа {int((good['side'] == 'right').sum())}")

    print(f"\n----- топ-25 латерализованных по частоте -----")
    print(f"  {'cell_type':<16s} {'сторона':>8s} {'своя,Гц':>9s} {'чужая,Гц':>9s} {'индекс':>8s}")
    for _, r in good.head(25).iterrows():
        print(f"  {str(r['cell_type'])[:16]:<16s} {r['side']:>8s} "
              f"{r['hz_ipsi']:>9.1f} {r['hz_contra']:>9.1f} {r['lat_index']:>8.2f}")

    # оценка выигрыша по шуму
    print("\n----- оценка подавления шума -----")
    win_s = 0.015
    for label, sub in (("только DNp09", df[df["cell_type"] == "DNp09"]),
                       ("латерализованные", good)):
        for side in ("left", "right"):
            g = sub[sub["side"] == side]
            if not len(g):
                continue
            rates_hz = g["hz_ipsi"].to_numpy()
            # ожидаемое число спайков популяции за окно и относительный шум
            k = rates_hz.sum() * win_s
            rel = 1.0 / np.sqrt(max(k, 1e-9))
            print(f"  {label:<18s} {side:>6s}: нейронов {len(g):>3d}, "
                  f"спайков за окно {k:>6.1f}, относительный шум {rel:>6.1%}")

    p = out("dn_population_readout.csv")
    df.to_csv(p, index=False)
    good.to_csv(out("dn_population_selected.csv"), index=False)
    print(f"\nсохранено: {p}")
    print(f"сохранено: {out('dn_population_selected.csv')}")
    print("\nЧитать так: относительный шум — это 1/sqrt(N спайков за окно).")
    print("У одиночного DNp09 он около 80 процентов, что и разваливало поведение.")


if __name__ == "__main__":
    main()
