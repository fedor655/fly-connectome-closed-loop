"""Есть ли у пути к P9 второй этаж?

Контекст. Сейчас обратная связь входит в прямых партнёров P9 — один синапс.
Это работает, но вход бедный. Попытка найти сенсорные популяции структурно
(нулевая входная степень) провалилась: критерий не опознал ни одного из 21
известного Sugar GRN, а найденные кандидаты не вызывают у P9 отклика.

Здесь проверяем другое: транзитивен ли путь. Если стимулировать не самих
драйверов P9, а тех, кто питает драйверов (два синапса до P9), откликнется ли
P9? Положительный ответ означает, что вход можно поднимать вверх по сети,
получая более богатую и менее рукотворную точку подключения.

Уровни:
  L1 — топ-K возбуждающих пресинаптических партнёров P9 left/right;
  L2 — топ-K возбуждающих пресинаптических партнёров нейронов L1
       (за вычетом самих L1 и P9, чтобы не стимулировать путь напрямую);
  L3 — то же ещё на шаг выше.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flypaths import add_fly_brain_to_path, out  # noqa: E402

add_fly_brain_to_path()
from benchmark import path_comp, path_con, path_wt  # noqa: E402
import run_pytorch as rp  # noqa: E402

# Стороны по аннотациям FlyWire. В benchmark.py подписи перепутаны местами.
P9_LEFT = 720575940635872101   # side=left
P9_RIGHT = 720575940627652358  # side=right

TOP_K = 20
DT = 0.1
SIM_MS = 1000.0
TRANSIENT_MS = 200.0
RATES_HZ = [100.0, 200.0, 300.0]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_CSV = out("p9_upstream_levels.csv")


def top_exc_presyn(W, targets, k, exclude):
    """Топ-k возбуждающих пресинаптических партнёров для набора targets."""
    scores = {}
    for t in targets:
        row = W.getrow(int(t)).tocoo()
        for col, val in zip(row.col, row.data):
            if val > 0 and int(col) not in exclude:
                scores[int(col)] = scores.get(int(col), 0.0) + float(val)
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
    return [c for c, _ in ranked], [v for _, v in ranked]


def measure(weights, n, stim, rate, p9_idx, l1_idx):
    model = rp.TorchModel(1, n, DT, rp.MODEL_PARAMS, weights,
                          exc_indices=list(stim), device=DEVICE)
    cond, delay_buf, spikes, v, refrac = model.state_init()
    rates = torch.zeros(1, n, device=DEVICE)
    rates[:, list(stim)] = rate
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(31337)

    acc_p9 = torch.zeros(2, device=DEVICE)
    acc_l1 = torch.zeros(len(l1_idx), device=DEVICE)
    pop = torch.zeros(n, device=DEVICE)
    n_trans, n_meas = int(TRANSIENT_MS / DT), int(SIM_MS / DT)

    with torch.no_grad():
        for step in range(n_trans + n_meas):
            cond, delay_buf, spikes, v, refrac = model(
                rates, cond, delay_buf, spikes, v, refrac, generator=gen)
            if step >= n_trans:
                acc_p9 += spikes[0, p9_idx]
                acc_l1 += spikes[0, l1_idx]
                pop += spikes[0]
    t = SIM_MS / 1000.0
    return {
        "p9_left_hz": float(acc_p9[0]) / t,
        "p9_right_hz": float(acc_p9[1]) / t,
        "l1_mean_hz": float(acc_l1.sum()) / t / len(l1_idx),
        "n_active": int((pop > 0).sum().item()),
    }


def main():
    print("=" * 78)
    print(" ТРАНЗИТИВНОСТЬ ПУТИ К P9: ЕСТЬ ЛИ ВТОРОЙ ЭТАЖ")
    print("=" * 78)

    df_comp = pd.read_csv(path_comp, index_col=0)
    flyid2i = {j: i for i, j in enumerate(df_comp.index)}
    n = len(flyid2i)
    idx_l, idx_r = flyid2i[P9_LEFT], flyid2i[P9_RIGHT]

    conn = pd.read_parquet(path_con, columns=[
        "Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
    W = sp.coo_matrix(
        (conn["Excitatory x Connectivity"].to_numpy().astype(np.float32),
         (conn["Postsynaptic_Index"].to_numpy(), conn["Presynaptic_Index"].to_numpy())),
        shape=(n, n)).tocsr()
    del conn

    p9 = {idx_l, idx_r}
    l1, w1 = top_exc_presyn(W, [idx_l, idx_r], TOP_K * 2, exclude=p9)
    l2, w2 = top_exc_presyn(W, l1, TOP_K * 2, exclude=p9 | set(l1))
    l3, w3 = top_exc_presyn(W, l2, TOP_K * 2, exclude=p9 | set(l1) | set(l2))
    del W

    print(f"\nL1 (1 синапс до P9): {len(l1)} нейронов, суммарный вес {w1[0]:.0f}..{w1[-1]:.0f}")
    print(f"L2 (2 синапса до P9): {len(l2)} нейронов, суммарный вес {w2[0]:.0f}..{w2[-1]:.0f}")
    print(f"L3 (3 синапса до P9): {len(l3)} нейронов, суммарный вес {w3[0]:.0f}..{w3[-1]:.0f}")

    weights = rp.get_weights(str(path_con), str(path_comp), str(path_wt), csr=True).to(DEVICE)
    p9_idx = torch.tensor([idx_l, idx_r], dtype=torch.long, device=DEVICE)
    l1_idx = torch.tensor(l1, dtype=torch.long, device=DEVICE)

    rows = []
    print(f"\n  {'уровень':>8s} {'вход, Гц':>10s} {'P9 L':>8s} {'P9 R':>8s} "
          f"{'L1 средн.':>11s} {'активных':>10s}")
    for name, stim in (("L1", l1), ("L2", l2), ("L3", l3)):
        for rate in RATES_HZ:
            t0 = time.perf_counter()
            r = measure(weights, n, stim, rate, p9_idx, l1_idx)
            print(f"  {name:>8s} {rate:>10.0f} {r['p9_left_hz']:>8.1f} {r['p9_right_hz']:>8.1f} "
                  f"{r['l1_mean_hz']:>11.1f} {r['n_active']:>10d}   [{time.perf_counter() - t0:.0f} с]")
            rows.append({"level": name, "n_stim": len(stim), "stim_rate_hz": rate, **r})
            pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    print(f"\nсохранено: {OUT_CSV}")
    print("\nЧитать так: если при стимуляции L2 частота P9 заметно выше нуля,")
    print("путь транзитивен и точку подключения можно поднимать вверх по сети.")


if __name__ == "__main__":
    main()
