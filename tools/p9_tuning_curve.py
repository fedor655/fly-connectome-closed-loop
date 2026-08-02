"""Тюнинг-кривая P9: как частота нисходящих нейронов зависит от входа.

Зачем. Прошлый MVP нормировал командный сигнал как clip(rate/100, 0, 1), взяв
100 Гц с потолка. Результат — константа 1.0 во всех циклах. Нормировку надо
выводить из измерения, а не назначать.

Что меряем. Эталонной моделью (run_pytorch.TorchModel, валидирована против
Brian2) прогоняем несколько условий стимуляции и строим зависимость частоты
P9 left/right от частоты входа.

Популяции стимуляции:
  drivers_l / drivers_r — топ-K прямых ВОЗБУЖДАЮЩИХ пресинаптических партнёров
                          P9 left / right (взяты прямо из коннектома);
  p9_direct             — сам P9 (положительный контроль, оптогенетический
                          аналог из статьи Shiu et al.);
  sugar                 — Sugar GRNs (отрицательный контроль, ожидаем 0).

Первые TRANSIENT_MS отбрасываются как переходный процесс.
"""
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

FLY_BRAIN_CODE = "/mnt/d/временное использование федей/мозг мухи/fly-brain/code"
sys.path.insert(0, FLY_BRAIN_CODE)
from benchmark import EXPERIMENTS, path_comp, path_con, path_wt  # noqa: E402
import run_pytorch as rp  # noqa: E402

P9_LEFT = 720575940627652358
P9_RIGHT = 720575940635872101

DT = 0.1
TRANSIENT_MS = 200.0
MEASURE_MS = 1000.0
RATES_HZ = [0.0, 10.0, 25.0, 50.0, 100.0, 150.0, 200.0, 300.0]
TOP_K = 20
SEED = 20260802

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_CSV = "/mnt/d/временное использование федей/мозг мухи/output/p9_tuning_curve.csv"


def top_excitatory_drivers(W, idx, k):
    """Топ-k пресинаптических партнёров нейрона idx с положительным весом."""
    row = W.getrow(idx).tocoo()
    pos = row.data > 0
    cols, vals = row.col[pos], row.data[pos]
    order = np.argsort(-vals)[:k]
    return cols[order].tolist(), vals[order].tolist()


def run_condition(model_ctor, num_neurons, stim_indices, rate_hz, idx_l, idx_r):
    """Один прогон: стимуляция stim_indices на rate_hz, замер частоты P9."""
    model = model_ctor(stim_indices)
    cond, delay_buf, spikes, v, refrac = model.state_init()

    rates = torch.zeros(1, num_neurons, device=DEVICE)
    if stim_indices and rate_hz > 0:
        rates[:, stim_indices] = rate_hz

    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(SEED)

    n_trans = int(TRANSIENT_MS / DT)
    n_meas = int(MEASURE_MS / DT)

    p9_l = p9_r = 0
    pop_spikes = 0
    active = torch.zeros(num_neurons, device=DEVICE)

    with torch.no_grad():
        for step in range(n_trans + n_meas):
            cond, delay_buf, spikes, v, refrac = model(
                rates, cond, delay_buf, spikes, v, refrac, generator=gen
            )
            if step >= n_trans:
                p9_l += int(spikes[0, idx_l].item())
                p9_r += int(spikes[0, idx_r].item())
                pop_spikes += int(spikes.sum().item())
                active += spikes[0]

    t_sec = MEASURE_MS / 1000.0
    return {
        "p9_left_hz": p9_l / t_sec,
        "p9_right_hz": p9_r / t_sec,
        "pop_rate_hz": pop_spikes / (num_neurons * t_sec),
        "n_active": int((active > 0).sum().item()),
    }


def main():
    print("=" * 78)
    print(" ТЮНИНГ-КРИВАЯ P9 (эталонная модель)")
    print("=" * 78)
    print(f"device={DEVICE}, транзиент={TRANSIENT_MS} мс, замер={MEASURE_MS} мс")

    df_comp = pd.read_csv(path_comp, index_col=0)
    flyid2i = {j: i for i, j in enumerate(df_comp.index)}
    i2flyid = {i: j for j, i in flyid2i.items()}
    n = len(flyid2i)
    idx_l, idx_r = flyid2i[P9_LEFT], flyid2i[P9_RIGHT]

    conn = pd.read_parquet(path_con)
    W = sp.coo_matrix(
        (conn["Excitatory x Connectivity"].to_numpy().astype(np.float32),
         (conn["Postsynaptic_Index"].to_numpy(), conn["Presynaptic_Index"].to_numpy())),
        shape=(n, n),
    ).tocsr()
    del conn

    drv_l, w_l = top_excitatory_drivers(W, idx_l, TOP_K)
    drv_r, w_r = top_excitatory_drivers(W, idx_r, TOP_K)
    print(f"\nдрайверы P9 left:  {len(drv_l)} нейронов, веса {w_l[0]:.0f}..{w_l[-1]:.0f}")
    print(f"драйверы P9 right: {len(drv_r)} нейронов, веса {w_r[0]:.0f}..{w_r[-1]:.0f}")
    print("  P9 left  топ-5:", [i2flyid[c] for c in drv_l[:5]])
    print("  P9 right топ-5:", [i2flyid[c] for c in drv_r[:5]])
    del W

    sugar = [flyid2i[x] for x in EXPERIMENTS["sugar"]["neu_exc"] if x in flyid2i]
    populations = {
        "drivers_both": sorted(set(drv_l) | set(drv_r)),
        "drivers_left": drv_l,
        "drivers_right": drv_r,
        "p9_direct": [idx_l, idx_r],
        "sugar": sugar,
    }

    print("\nЗагружаю веса...")
    weights = rp.get_weights(str(path_con), str(path_comp), str(path_wt), csr=True).to(DEVICE)

    def ctor(stim_indices):
        return rp.TorchModel(
            1, n, DT, rp.MODEL_PARAMS, weights,
            exc_indices=list(stim_indices) if stim_indices else None,
            device=DEVICE,
        )

    rows = []
    total = sum(len(RATES_HZ) for _ in populations)
    done = 0
    t_all = time.perf_counter()

    for pop_name, stim in populations.items():
        print(f"\n===== популяция: {pop_name} ({len(stim)} нейронов) =====")
        print(f"  {'вход, Гц':>10s} {'P9 L, Гц':>10s} {'P9 R, Гц':>10s} "
              f"{'популяция, Гц':>15s} {'активных':>10s}")
        for rate in RATES_HZ:
            t0 = time.perf_counter()
            res = run_condition(ctor, n, stim, rate, idx_l, idx_r)
            done += 1
            print(f"  {rate:>10.0f} {res['p9_left_hz']:>10.1f} {res['p9_right_hz']:>10.1f} "
                  f"{res['pop_rate_hz']:>15.4f} {res['n_active']:>10d}"
                  f"   [{done}/{total}, {time.perf_counter() - t0:.0f}с]")
            rows.append({"population": pop_name, "n_stim": len(stim),
                         "stim_rate_hz": rate, **res})
            pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    print(f"\nвсего: {time.perf_counter() - t_all:.0f} с")
    print(f"сохранено: {OUT_CSV}")


if __name__ == "__main__":
    main()
