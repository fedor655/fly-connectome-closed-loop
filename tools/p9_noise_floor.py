"""Предел разрешения командного сигнала: сколько нужно усреднять, чтобы
обратная связь была отличима от пуассоновского шума.

Постановка задачи. В замкнутом контуре обратная связь меняет частоту входа
между рабочими точками (при ходьбе число касаний на сторону скачет между 2/3 и
3/3). Считываем мы всего два нейрона — P9 left и right. За окно 15 мс каждый
даёт единицы спайков, поэтому оценка частоты шумная. Вопрос: при каком времени
усреднения два рабочих режима становятся различимы?

Метод. Гоняем мозг при фиксированных частотах входа, копим спайки P9 по окнам
15 мс, затем офлайн считаем экспоненциальное сглаживание при разных tau и меряем
различимость соседних режимов через d-prime:

    d' = |m_hi - m_lo| / sqrt((s_hi^2 + s_lo^2) / 2)

d' = 1 — режимы едва различимы, d' > 2 — уверенно.
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

P9_LEFT = 720575940627652358
P9_RIGHT = 720575940635872101

DT = 0.1
SYNC_MS = 15.0
STEPS_PER_WINDOW = int(SYNC_MS / DT)
N_WINDOWS = 400          # 400 * 15 мс = 6 с на режим
N_DRIVERS = 20
WARMUP_WINDOWS = 20

# Рабочие точки из замкнутого прогона: 1/3, 2/3 и 3/3 касаний на сторону
# при настройке --fb-base 5 --fb-span 195
RATES_HZ = [70.0, 135.0, 200.0]
TAUS_MS = [0.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def ema_series(x, tau_ms):
    """Экспоненциальное сглаживание ряда пооконных оценок частоты."""
    if tau_ms <= 0:
        return np.asarray(x, dtype=float)
    alpha = 1.0 - np.exp(-SYNC_MS / tau_ms)
    out = np.empty(len(x), dtype=float)
    acc = float(x[0])
    for i, v in enumerate(x):
        acc += alpha * (v - acc)
        out[i] = acc
    return out


def main():
    print("=" * 78)
    print(" ПРЕДЕЛ РАЗРЕШЕНИЯ КОМАНДНОГО СИГНАЛА P9")
    print("=" * 78)
    print(f"device={DEVICE}, окно={SYNC_MS} мс, окон на режим={N_WINDOWS} "
          f"({N_WINDOWS * SYNC_MS / 1000:.1f} с)")

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

    def top_drivers(idx):
        row = W.getrow(idx).tocoo()
        pos = row.data > 0
        cols, vals = row.col[pos], row.data[pos]
        return cols[np.argsort(-vals)[:N_DRIVERS]].tolist()

    drv_l, drv_r = top_drivers(idx_l), top_drivers(idx_r)
    del W

    weights = rp.get_weights(str(path_con), str(path_comp), str(path_wt), csr=True).to(DEVICE)
    stim_all = sorted(set(drv_l) | set(drv_r))

    p9_idx = torch.tensor([idx_l, idx_r], dtype=torch.long, device=DEVICE)
    counts_by_rate = {}

    for rate in RATES_HZ:
        model = rp.TorchModel(1, n, DT, rp.MODEL_PARAMS, weights,
                              exc_indices=stim_all, device=DEVICE)
        cond, delay_buf, spikes, v, refrac = model.state_init()
        rates = torch.zeros(1, n, device=DEVICE)
        rates[:, drv_l] = rate
        rates[:, drv_r] = rate
        gen = torch.Generator(device=DEVICE)
        gen.manual_seed(4242)

        per_window = []
        acc = torch.zeros(2, device=DEVICE)
        t0 = time.perf_counter()
        with torch.no_grad():
            for w in range(WARMUP_WINDOWS + N_WINDOWS):
                acc.zero_()
                for _ in range(STEPS_PER_WINDOW):
                    cond, delay_buf, spikes, v, refrac = model(
                        rates, cond, delay_buf, spikes, v, refrac, generator=gen)
                    acc += spikes[0, p9_idx]
                if w >= WARMUP_WINDOWS:
                    per_window.append([float(x) for x in acc.tolist()])
        arr = np.array(per_window) / (SYNC_MS / 1000.0)  # -> Гц
        counts_by_rate[rate] = arr
        print(f"  вход {rate:>5.0f} Гц: P9 L {arr[:, 0].mean():6.1f} +- {arr[:, 0].std():5.1f} Гц, "
              f"P9 R {arr[:, 1].mean():6.1f} +- {arr[:, 1].std():5.1f} Гц "
              f"[{time.perf_counter() - t0:.0f} с]")

    # ---------- различимость соседних режимов ----------
    print("\n" + "=" * 78)
    print(" РАЗЛИЧИМОСТЬ СОСЕДНИХ РЕЖИМОВ (d-prime, канал P9 left)")
    print("=" * 78)
    print(f"  {'tau, мс':>9s} {'задержка':>10s}", end="")
    pairs = list(zip(RATES_HZ[:-1], RATES_HZ[1:]))
    for lo, hi in pairs:
        print(f" {f'{lo:.0f}->{hi:.0f} Гц':>14s}", end="")
    print()

    rows = []
    for tau in TAUS_MS:
        lag = "мгновенно" if tau <= 0 else f"~{tau:.0f} мс"
        print(f"  {tau:>9.0f} {lag:>10s}", end="")
        row = {"tau_ms": tau}
        for lo, hi in pairs:
            a = ema_series(counts_by_rate[lo][:, 0], tau)
            b = ema_series(counts_by_rate[hi][:, 0], tau)
            # отбрасываем начальный участок, где сглаживание ещё не установилось
            skip = 0 if tau <= 0 else min(len(a) // 4, int(4 * tau / SYNC_MS))
            a, b = a[skip:], b[skip:]
            d = abs(b.mean() - a.mean()) / np.sqrt((a.var() + b.var()) / 2)
            print(f" {d:>14.2f}", end="")
            row[f"dprime_{lo:.0f}_{hi:.0f}"] = d
        print()
        rows.append(row)

    out_csv = out("p9_noise_floor.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nсохранено: {out_csv}")
    print("\nЧитать так: d' = 1 — режимы едва различимы, d' > 2 — уверенно.")
    print("Столбец tau показывает, какой ценой по задержке это достигается.")


if __name__ == "__main__":
    main()
