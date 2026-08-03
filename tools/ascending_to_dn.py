"""Доходит ли сигнал от ВОСХОДЯЩИХ нейронов до нисходящих DNp09?

Зачем. Сейчас обратная связь от лапок подаётся в вручную подобранных
пресинаптических партнёров DNp09. Это работает, но выбрано нами.
Аннотации FlyWire (Schlegel et al., Nature 2024) показали, что в датасете есть
1736 восходящих нейронов, все через шейный коннектив. Именно они несут в мозг
информацию от вентрального нервного тяжа, где сидят механосенсоры ног.
Это биологически правильная точка входа — но работает ли она на самом деле,
надо измерить, а не предположить.

Проверяем заодно вторую находку аннотаций: подписи сторон в EXPERIMENTS
перепутаны. 720575940627652358 подписан как «P9 left», а по FlyWire это
side=right, и наоборот. Здесь стороны берутся из аннотаций.
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

# Стороны по аннотациям FlyWire, а не по подписям в benchmark.py
DNP09_LEFT = 720575940635872101
DNP09_RIGHT = 720575940627652358

DT = 0.1
SIM_MS = 1000.0
TRANSIENT_MS = 200.0
RATES_HZ = [0.0, 50.0, 100.0, 200.0]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def measure(weights, n, stim, rate, dn_idx, gen_seed=555):
    model = rp.TorchModel(1, n, DT, rp.MODEL_PARAMS, weights,
                          exc_indices=list(stim) if stim else None, device=DEVICE)
    cond, delay_buf, spikes, v, refrac = model.state_init()
    rates = torch.zeros(1, n, device=DEVICE)
    if stim and rate > 0:
        rates[:, list(stim)] = rate
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(gen_seed)

    acc = torch.zeros(2, device=DEVICE)
    pop = torch.zeros(n, device=DEVICE)
    n_trans, n_meas = int(TRANSIENT_MS / DT), int(SIM_MS / DT)
    with torch.no_grad():
        for step in range(n_trans + n_meas):
            cond, delay_buf, spikes, v, refrac = model(
                rates, cond, delay_buf, spikes, v, refrac, generator=gen)
            if step >= n_trans:
                acc += spikes[0, dn_idx]
                pop += spikes[0]
    t = SIM_MS / 1000.0
    return (float(acc[0]) / t, float(acc[1]) / t, int((pop > 0).sum().item()))


def main():
    print("=" * 78)
    print(" ВОСХОДЯЩИЕ НЕЙРОНЫ -> DNp09")
    print("=" * 78)

    comp = pd.read_csv(path_comp, index_col=0)
    flyid2i = {int(j): i for i, j in enumerate(comp.index)}
    n = len(flyid2i)

    ann = pd.read_csv(ANNOTATIONS, sep="\t", low_memory=False)
    ann["root_id"] = pd.to_numeric(ann["root_id"], errors="coerce")
    ann = ann.dropna(subset=["root_id"])
    ann["root_id"] = ann["root_id"].astype("int64")
    ann = ann[ann["root_id"].isin(flyid2i.keys())]

    asc = ann[ann["super_class"] == "ascending"]
    asc_l = [flyid2i[i] for i in asc.loc[asc["side"] == "left", "root_id"]]
    asc_r = [flyid2i[i] for i in asc.loc[asc["side"] == "right", "root_id"]]
    print(f"восходящих: слева {len(asc_l)}, справа {len(asc_r)}")

    mech = ann[(ann["super_class"] == "sensory") & (ann["cell_class"] == "mechanosensory")]
    mech_l = [flyid2i[i] for i in mech.loc[mech["side"] == "left", "root_id"]]
    mech_r = [flyid2i[i] for i in mech.loc[mech["side"] == "right", "root_id"]]
    print(f"механосенсорных: слева {len(mech_l)}, справа {len(mech_r)}")

    idx_dn_l, idx_dn_r = flyid2i[DNP09_LEFT], flyid2i[DNP09_RIGHT]
    print(f"DNp09 left  index={idx_dn_l} (id {DNP09_LEFT})")
    print(f"DNp09 right index={idx_dn_r} (id {DNP09_RIGHT})")

    weights = rp.get_weights(str(path_con), str(path_comp), str(path_wt), csr=True).to(DEVICE)
    dn_idx = torch.tensor([idx_dn_l, idx_dn_r], dtype=torch.long, device=DEVICE)

    populations = {
        "восходящие слева": asc_l,
        "восходящие справа": asc_r,
        "механосенс. слева": mech_l,
        "механосенс. справа": mech_r,
    }

    rows = []
    print(f"\n  {'популяция':>20s} {'n':>5s} {'вход,Гц':>9s} "
          f"{'DNp09 L':>9s} {'DNp09 R':>9s} {'активных':>9s}")
    for name, stim in populations.items():
        if not stim:
            continue
        for rate in RATES_HZ:
            t0 = time.perf_counter()
            l, r, act = measure(weights, n, stim, rate, dn_idx)
            print(f"  {name:>20s} {len(stim):>5d} {rate:>9.0f} "
                  f"{l:>9.1f} {r:>9.1f} {act:>9d}   [{time.perf_counter()-t0:.0f}с]")
            rows.append({"population": name, "n_stim": len(stim), "stim_rate_hz": rate,
                         "dnp09_left_hz": l, "dnp09_right_hz": r, "n_active": act})
            pd.DataFrame(rows).to_csv(out("ascending_to_dn.csv"), index=False)

    print(f"\nсохранено: {out('ascending_to_dn.csv')}")
    print("\nЧитать так: если восходящие дают ненулевую частоту DNp09, обратную")
    print("связь от лапок можно подавать в них — это биологически правильный вход,")
    print("а не подобранные вручную партнёры.")


if __name__ == "__main__":
    main()
