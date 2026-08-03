"""Положительный контроль: зажигают ли ЗРИТЕЛЬНЫЕ нейроны DNp09?

Контекст. Разбор входов показал, что единственный крупный источник чистого
возбуждения для DNp09 — зрительные проекционные нейроны (+880 при 22.8% массы
веса), сильнейшие типы LC9 и LC31a. Отрицательные контроли это подтверждают
косвенно: восходящие (1736 нейронов) и механосенсорные (2656) не дают DNp09
ни одного спайка на частотах до 200 Гц, вкусовые тоже.

Но отрицательные контроли сами по себе ничего не доказывают: может, до DNp09
вообще ничем не достучаться, кроме его ближайших партнёров. Нужен
положительный контроль. Если стимуляция LC-нейронов, отобранных ПО АННОТАЦИЯМ
(а не по весу связи с DNp09), даёт отклик — вывод про зрение замыкается.

Ключевая деталь: здесь популяции отбираются по типу клетки, а не по силе связи
с DNp09. Иначе получилось бы рассуждение по кругу.
"""
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flypaths import ANNOTATIONS, add_fly_brain_to_path, out  # noqa: E402

add_fly_brain_to_path()
from benchmark import path_comp, path_con, path_wt  # noqa: E402
import run_pytorch as rp  # noqa: E402

DNP09_LEFT = 720575940635872101
DNP09_RIGHT = 720575940627652358

DT = 0.1
SIM_MS = 1000.0
TRANSIENT_MS = 200.0
RATES_HZ = [0.0, 50.0, 100.0, 200.0]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def measure(weights, n, stim, rate, dn_idx):
    model = rp.TorchModel(1, n, DT, rp.MODEL_PARAMS, weights,
                          exc_indices=list(stim) if stim else None, device=DEVICE)
    cond, delay_buf, spikes, v, refrac = model.state_init()
    rates = torch.zeros(1, n, device=DEVICE)
    if stim and rate > 0:
        rates[:, list(stim)] = rate
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(909)

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
    return float(acc[0]) / t, float(acc[1]) / t, int((pop > 0).sum().item())


def main():
    print("=" * 78)
    print(" ПОЛОЖИТЕЛЬНЫЙ КОНТРОЛЬ: ЗРИТЕЛЬНЫЕ НЕЙРОНЫ -> DNp09")
    print("=" * 78)

    comp = pd.read_csv(path_comp, index_col=0)
    flyid2i = {int(j): i for i, j in enumerate(comp.index)}
    n = len(flyid2i)

    ann = pd.read_csv(ANNOTATIONS, sep="\t", low_memory=False)
    ann["root_id"] = pd.to_numeric(ann["root_id"], errors="coerce")
    ann = ann.dropna(subset=["root_id"])
    ann["root_id"] = ann["root_id"].astype("int64")
    ann = ann[ann["root_id"].isin(flyid2i.keys())]

    def pick(mask, side=None):
        sub = ann[mask]
        if side:
            sub = sub[sub["side"] == side]
        return [flyid2i[i] for i in sub["root_id"]]

    is_vp = ann["super_class"] == "visual_projection"
    populations = {
        "LC9 слева": pick((ann["cell_type"] == "LC9"), "left"),
        "LC9 справа": pick((ann["cell_type"] == "LC9"), "right"),
        "LC31a обе стороны": pick(ann["cell_type"] == "LC31a"),
        "все зрит. проекц. слева": pick(is_vp, "left"),
        "все зрит. проекц. справа": pick(is_vp, "right"),
    }
    for k, v in populations.items():
        print(f"  {k:<26s} {len(v):>5d} нейронов")

    idx_l, idx_r = flyid2i[DNP09_LEFT], flyid2i[DNP09_RIGHT]
    weights = rp.get_weights(str(path_con), str(path_comp), str(path_wt), csr=True).to(DEVICE)
    dn_idx = torch.tensor([idx_l, idx_r], dtype=torch.long, device=DEVICE)

    rows = []
    print(f"\n  {'популяция':>26s} {'n':>5s} {'вход,Гц':>8s} "
          f"{'DNp09 L':>9s} {'DNp09 R':>9s} {'активных':>9s}")
    for name, stim in populations.items():
        if not stim:
            continue
        for rate in RATES_HZ:
            t0 = time.perf_counter()
            l, r, act = measure(weights, n, stim, rate, dn_idx)
            print(f"  {name:>26s} {len(stim):>5d} {rate:>8.0f} "
                  f"{l:>9.1f} {r:>9.1f} {act:>9d}   [{time.perf_counter()-t0:.0f}с]")
            rows.append({"population": name, "n_stim": len(stim), "stim_rate_hz": rate,
                         "dnp09_left_hz": l, "dnp09_right_hz": r, "n_active": act})
            pd.DataFrame(rows).to_csv(out("visual_to_dn.csv"), index=False)

    print(f"\nсохранено: {out('visual_to_dn.csv')}")


if __name__ == "__main__":
    main()
