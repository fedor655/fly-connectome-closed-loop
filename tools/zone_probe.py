"""Стимуляция одной зоны листа и замер отклика нисходящих."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flypaths import ANNOTATIONS, add_fly_brain_to_path  # noqa: E402
from tools.retina_zone_map import load_or_build, N_ZONES  # noqa: E402
from closed_loop_vision import load_brain_assets, DT_BRAIN_MS  # noqa: E402

add_fly_brain_to_path()
import run_pytorch as rp  # noqa: E402
from benchmark import path_comp  # noqa: E402


def descending_indices(device):
    """Индексы всех нисходящих нейронов и сторона каждого."""
    comp = pd.read_csv(path_comp, index_col=0)
    flyid2i = {int(j): i for i, j in enumerate(comp.index)}
    ann = pd.read_csv(ANNOTATIONS, sep="\t", low_memory=False)
    ann["root_id"] = pd.to_numeric(ann["root_id"], errors="coerce")
    ann = ann.dropna(subset=["root_id"])
    ann["root_id"] = ann["root_id"].astype("int64")
    desc = ann[(ann["super_class"] == "descending")
               & ann["side"].isin(["left", "right"])
               & ann["root_id"].isin(flyid2i.keys())]
    idx = np.array([flyid2i[int(x)] for x in desc["root_id"]], dtype=np.int64)
    return idx, np.array(desc["side"].tolist())


def probe_zone(weights, n, dn_idx, stim_idx, rate_hz, seed, device,
               transient_ms=300.0, measure_ms=500.0):
    """Частоты нисходящих при стимуляции одной зоны."""
    stim = list(stim_idx)
    model = rp.TorchModel(1, n, DT_BRAIN_MS, rp.MODEL_PARAMS, weights,
                          exc_indices=stim, device=device)
    c, d, s, v, r = model.state_init()
    rates = torch.zeros(1, n, device=device)
    rates[:, stim] = rate_hz
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    idx_t = torch.tensor(dn_idx, dtype=torch.long, device=device)
    acc = torch.zeros(len(dn_idx), device=device)
    n_tr, n_me = int(transient_ms / DT_BRAIN_MS), int(measure_ms / DT_BRAIN_MS)
    with torch.no_grad():
        for step in range(n_tr + n_me):
            c, d, s, v, r = model(rates, c, d, s, v, r, generator=g)
            if step >= n_tr:
                acc.add_(s[0, idx_t])
    return np.array(acc.tolist()) / (measure_ms / 1000.0)


def self_check() -> None:
    """Отклик должен расти с частотой и не должен упираться в полку.

    Полка означает насыщение: сеть отвечает одинаково на 50 и на 400 Гц,
    и никакая пространственная структура через неё не пройдёт.
    """
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assets = load_brain_assets(device, verbose=False)
    dn_idx, _ = descending_indices(device)
    zones = load_or_build()
    idx, zn = zones["left"]
    stim = idx[zn == 0].tolist()

    rates = [25.0, 50.0, 100.0, 200.0, 400.0]
    resp = [float(probe_zone(assets["weights"], assets["n"], dn_idx,
                             stim, r, seed=0, device=device).mean())
            for r in rates]
    for r, v in zip(rates, resp):
        print(f"  {r:5.0f} Гц -> средний отклик нисходящих {v:6.2f} Гц")

    assert resp[-1] > resp[0], "отклик не растёт с частотой — стимуляция не доходит"
    top = (resp[-1] - resp[-2]) / max(resp[-1], 1e-9)
    assert top > 0.02, (
        f"полка между {rates[-2]} и {rates[-1]} Гц (прирост {top:.1%}): "
        "сеть в насыщении, рабочую точку надо брать ниже")
    print("самопроверка стимуляции зоны: ОК")


if __name__ == "__main__":
    self_check()
