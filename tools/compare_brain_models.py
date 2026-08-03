"""Сравнение двух реализаций модели мозга при ИДЕНТИЧНОМ входе.

A) Эталон  — TorchModel из fly-brain/code/run_pytorch.py (валидирован против
              Brian2 CPU ground truth, статья Shiu et al., Nature 2024).
B) MVP      — BrainNetwork, переписанная вручную в closed_loop_mvp.py.

Гипотеза: в (B) рекуррентный путь завышен примерно в 700 раз из-за двух
потерянных множителей — wScale=0.275 и time_factor_mem=dt/tauMem=0.005.
Проверяем измерением частот, а не рассуждением.

Вход в обоих случаях: Sugar GRNs (21 нейрон) @ 200 Гц, 1000 мс, seed фиксирован.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flypaths import DATA_DIR, PROJECT_DIR, add_fly_brain_to_path, out  # noqa: E402

add_fly_brain_to_path()
from benchmark import EXPERIMENTS, path_comp, path_con, path_wt  # noqa: E402
import run_pytorch as rp  # noqa: E402

DT = 0.1
T_MS = 1000.0
NUM_STEPS = int(T_MS / DT)
SEED = 12345

# Стороны по аннотациям FlyWire. В benchmark.py подписи перепутаны местами.
P9_LEFT = 720575940635872101   # side=left
P9_RIGHT = 720575940627652358  # side=right

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def summarize(tag, spike_counts, p9_l, p9_r, n_steps, elapsed):
    t_sec = n_steps * DT / 1000.0
    total = int(spike_counts.sum().item())
    active = int((spike_counts > 0).sum().item())
    pop_rate = total / (spike_counts.numel() * t_sec)
    print(f"\n----- {tag} -----")
    print(f"  время расчёта:        {elapsed:.1f} с")
    print(f"  всего спайков:        {total}")
    print(f"  активных нейронов:    {active} из {spike_counts.numel()}")
    print(f"  средняя частота попул.: {pop_rate:.3f} Гц")
    print(f"  P9 left:              {p9_l} спайков  ({p9_l / t_sec:.1f} Гц)")
    print(f"  P9 right:             {p9_r} спайков  ({p9_r / t_sec:.1f} Гц)")
    return {
        "tag": tag, "total_spikes": total, "active_neurons": active,
        "pop_rate_hz": pop_rate, "p9_left_hz": p9_l / t_sec,
        "p9_right_hz": p9_r / t_sec,
    }


def run_reference(flyid2i, weights_csr, sugar_ids):
    """Эталонная модель ровно так, как её гоняет бенчмарк."""
    exc_indices = [flyid2i[n] for n in sugar_ids if n in flyid2i]
    num_neurons = weights_csr.shape[0]

    model = rp.TorchModel(
        1, num_neurons, DT, rp.MODEL_PARAMS, weights_csr,
        exc_indices=exc_indices, device=DEVICE,
    )
    cond, delay_buf, spikes, v, refrac = model.state_init()

    rates = torch.zeros(1, num_neurons, device=DEVICE)
    rates[:, exc_indices] = EXPERIMENTS["sugar"]["stim_rate"]

    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(SEED)

    counts = torch.zeros(num_neurons, device=DEVICE)
    idx_l, idx_r = flyid2i[P9_LEFT], flyid2i[P9_RIGHT]
    p9_l = p9_r = 0

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(NUM_STEPS):
            cond, delay_buf, spikes, v, refrac = model(
                rates, cond, delay_buf, spikes, v, refrac, generator=gen
            )
            counts += spikes[0]
            p9_l += int(spikes[0, idx_l].item())
            p9_r += int(spikes[0, idx_r].item())
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return counts.cpu(), p9_l, p9_r, time.perf_counter() - t0


def run_mvp(flyid2i, weights_coo, sugar_ids):
    """Модель из closed_loop_mvp.py — импортируем классы, не запуская main()."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "clm", str(PROJECT_DIR / "closed_loop_mvp.py")
    )
    clm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(clm)

    sensory_indices = [flyid2i[n] for n in sugar_ids if n in flyid2i]
    num_neurons = weights_coo.shape[0]

    torch.manual_seed(SEED)
    brain = clm.BrainNetwork(
        batch=1, size=num_neurons, dt=DT, params=clm.MODEL_PARAMS,
        weights=weights_coo, sensory_indices=sensory_indices, device=DEVICE,
    )
    brain.eval()
    cond, delay_buf, v, refrac = brain.state_init()

    counts = torch.zeros(num_neurons, device=DEVICE)
    idx_l, idx_r = flyid2i[P9_LEFT], flyid2i[P9_RIGHT]
    p9_l = p9_r = 0
    rate = EXPERIMENTS["sugar"]["stim_rate"]

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(NUM_STEPS):
            cond, delay_buf, v, refrac, spikes = brain(rate, cond, delay_buf, v, refrac)
            counts += spikes[0]
            p9_l += int(spikes[0, idx_l].item())
            p9_r += int(spikes[0, idx_r].item())
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return counts.cpu(), p9_l, p9_r, time.perf_counter() - t0


def main():
    print("=" * 78)
    print(" СРАВНЕНИЕ МОДЕЛЕЙ МОЗГА: эталон run_pytorch.py против MVP")
    print("=" * 78)
    print(f"device={DEVICE}, dt={DT} мс, длительность={T_MS} мс, seed={SEED}")

    flyid2i, i2flyid = rp.get_hash_tables(str(path_comp))
    sugar_ids = EXPERIMENTS["sugar"]["neu_exc"]
    print(f"нейронов: {len(flyid2i)}, стимулируем Sugar GRNs: {len(sugar_ids)} @ 200 Гц")

    results = []

    print("\nЗагружаю веса CSR для эталона...")
    w_csr = rp.get_weights(str(path_con), str(path_comp), str(path_wt), csr=True).to(DEVICE)
    c, l, r, el = run_reference(flyid2i, w_csr, sugar_ids)
    results.append(summarize("A. ЭТАЛОН (run_pytorch.TorchModel)", c, l, r, NUM_STEPS, el))
    ref_counts = c
    del w_csr
    torch.cuda.empty_cache() if DEVICE == "cuda" else None

    print("\nЗагружаю веса COO для MVP...")
    w_coo = rp.get_weights(str(path_con), str(path_comp), str(path_wt), csr=False).to(DEVICE)
    c, l, r, el = run_mvp(flyid2i, w_coo, sugar_ids)
    results.append(summarize("B. MVP (closed_loop_mvp.BrainNetwork)", c, l, r, NUM_STEPS, el))
    mvp_counts = c
    del w_coo
    torch.cuda.empty_cache() if DEVICE == "cuda" else None

    # --- сверка эталона с сохранённым результатом прошлого прогона ---
    ref_parquet = str(DATA_DIR / "results" / "pytorch_t1.0s_n1.parquet")
    if os.path.exists(ref_parquet):
        df = pd.read_parquet(ref_parquet)
        print("\n----- сверка эталона с сохранённым pytorch_t1.0s_n1.parquet -----")
        print(f"  спайков в файле:      {len(df)}")
        print(f"  активных в файле:     {df['flywire_id'].nunique()}")
        print(f"  спайков сейчас:       {int(ref_counts.sum().item())}")
        print(f"  активных сейчас:      {int((ref_counts > 0).sum().item())}")

    print("\n" + "=" * 78)
    print(" ИТОГ")
    print("=" * 78)
    a, b = results[0], results[1]
    print(f"  спайков:   эталон {a['total_spikes']:>10d}   MVP {b['total_spikes']:>10d}"
          f"   отношение {b['total_spikes'] / max(a['total_spikes'], 1):.1f}x")
    print(f"  активных:  эталон {a['active_neurons']:>10d}   MVP {b['active_neurons']:>10d}")
    print(f"  частота:   эталон {a['pop_rate_hz']:>10.3f}   MVP {b['pop_rate_hz']:>10.3f} Гц")
    print(f"  P9 left:   эталон {a['p9_left_hz']:>10.1f}   MVP {b['p9_left_hz']:>10.1f} Гц")
    print(f"  P9 right:  эталон {a['p9_right_hz']:>10.1f}   MVP {b['p9_right_hz']:>10.1f} Гц")

    out_csv = out("brain_model_comparison.csv")
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"\nсохранено: {out_csv}")


if __name__ == "__main__":
    main()
