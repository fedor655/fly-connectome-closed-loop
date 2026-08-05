"""Проводит ли наша LIF от детекторов движения дальше, к лобуле и нисходящим?

Это ворота перед стыковкой с flyvis, и они решают развилку целиком.

Зачем. Путь А — гибрид: flyvis считает оптические доли (он для того и обучен),
наша LIF считает всё после. Стык надо класть туда, где типы клеток есть в обеих
моделях. Проверено: из 65 типов flyvis 50 совпадают по имени с FlyWire и
покрывают 55 029 наших нейронов, 27 602 слева и 27 427 справа. Последняя ступень
flyvis — T4/T5/Tm/TmY, то есть детекторы движения и вход в лобулу. LC-нейронов у
flyvis нет вовсе, поэтому ниже стыка всё считает наша модель.

Почему это не очевидно и требует замера. Раньше измерено, что стимуляция
фоторецепторов на 900 Гц даёт T4 и T5 ровно 0.00 Гц: сигнал не доходит до
детекторов движения. Гибрид эту ступень обходит — рейты на T4/T5 приходят
снаружи, от flyvis. Но остаётся вторая половина вопроса, на которую ответа нет:
проводит ли модель ОТ T4/T5 дальше, к зрительным проекционным и нисходящим.
Если и там обрыв, гибрид не поможет и надо идти путём Б.

Что меряем. Стимулируем нейроны стыка на левой стороне и смотрим отклик
зрительных проекционных (LC) и нисходящих с ОБЕИХ сторон. Три набора стимуляции,
чтобы отделить вклад ступеней, и нулевая частота как отрицательный контроль.
Латеральность — второй контроль: если отвечают обе стороны одинаково, значит
отвечают не на сигнал.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from flypaths import ANNOTATIONS, add_fly_brain_to_path, out  # noqa: E402

add_fly_brain_to_path()
from benchmark import path_comp, path_con, path_wt  # noqa: E402
import run_pytorch as rp  # noqa: E402

DT = 0.1
SIM_MS = 1000.0
TRANSIENT_MS = 200.0
# Физиологический диапазон T4/T5 — десятки герц. Берём с запасом вверх, чтобы
# отличить «не проводит» от «проводит, но нужен вход выше физиологического».
RATES_HZ = [0.0, 25.0, 50.0, 100.0, 200.0, 400.0]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 777

# 50 типов flyvis, совпавших по имени с FlyWire. Список получен пересечением
# ConnectomeFromAvgFilters.unique_cell_types с колонкой cell_type аннотаций.
# Не совпали 15: R1..R6 (во FlyWire сгруппированы как R1-6), CT1(Lo1), CT1(M10),
# Mi3, Mi11, Mi12, Tm5Y, Tm28, Tm30, TmY13.
FLYVIS_TYPES = [
    "R7", "R8",
    "L1", "L2", "L3", "L4", "L5", "Lawf1", "Lawf2", "Am", "C2", "C3",
    "Mi1", "Mi2", "Mi4", "Mi9", "Mi10", "Mi13", "Mi14", "Mi15",
    "T1", "T2", "T2a", "T3",
    "T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d",
    "Tm1", "Tm2", "Tm3", "Tm4", "Tm5a", "Tm5b", "Tm5c", "Tm9", "Tm16", "Tm20",
    "TmY3", "TmY4", "TmY5a", "TmY9", "TmY10", "TmY14", "TmY15", "TmY18",
]
T4T5 = [t for t in FLYVIS_TYPES if t.startswith(("T4", "T5"))]
LOBULA_IN = T4T5 + [t for t in FLYVIS_TYPES if t.startswith(("Tm", "TmY"))]


def main():
    print("=" * 78)
    print(" ВОРОТА ГИБРИДА: проводит ли LIF от стыка flyvis к лобуле и нисходящим")
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
    L = ann["side"] == "left"
    R = ann["side"] == "right"

    def idx_of(mask):
        return [flyid2i[int(x)] for x in ann.loc[mask, "root_id"]]

    # ---------- наборы стимуляции: слева, чтобы латеральность была контролем ----------
    stim_sets = {
        "только T4+T5": idx_of(ct.isin(T4T5) & L),
        "T4,T5,Tm,TmY": idx_of(ct.isin(LOBULA_IN) & L),
        "весь стык flyvis": idx_of(ct.isin(FLYVIS_TYPES) & L),
    }
    print("\n----- наборы стимуляции (левое полушарие) -----")
    for k, v in stim_sets.items():
        print(f"  {k:<20s} {len(v):>7d} нейронов")

    # ---------- что слушаем ----------
    vp = ann["super_class"] == "visual_projection"
    dn = ann["super_class"] == "descending"
    stages = {
        "LC своя (лев)": idx_of(vp & L),
        "LC чужая (прав)": idx_of(vp & R),
        "DN своя (лев)": idx_of(dn & L),
        "DN чужая (прав)": idx_of(dn & R),
    }
    print("\n----- что слушаем -----")
    for k, v in stages.items():
        print(f"  {k:<20s} {len(v):>7d} нейронов")

    print(f"\nустройство: {DEVICE}; окно {SIM_MS:.0f} мс после {TRANSIENT_MS:.0f} мс транзиента")
    weights = rp.get_weights(str(path_con), str(path_comp), str(path_wt), csr=True).to(DEVICE)
    tens = {k: torch.tensor(v, dtype=torch.long, device=DEVICE)
            for k, v in stages.items() if v}
    names = list(stages)

    rows = []
    for set_name, stim_idx in stim_sets.items():
        print("\n" + "-" * 78)
        print(f" стимуляция: {set_name} ({len(stim_idx)} нейронов слева)")
        print("-" * 78)
        print(f"  {'вход, Гц':>9s}" + "".join(f"{k:>17s}" for k in names))
        for rate in RATES_HZ:
            model = rp.TorchModel(1, n, DT, rp.MODEL_PARAMS, weights,
                                  exc_indices=stim_idx, device=DEVICE)
            cond, delay_buf, spikes, v, refrac = model.state_init()
            rates = torch.zeros(1, n, device=DEVICE)
            if rate > 0:
                rates[:, stim_idx] = rate
            gen = torch.Generator(device=DEVICE)
            gen.manual_seed(SEED)

            acc = {k: torch.zeros((), device=DEVICE) for k in tens}
            act = {k: torch.zeros(len(stages[k]), device=DEVICE) for k in tens}
            n_tr, n_me = int(TRANSIENT_MS / DT), int(SIM_MS / DT)
            t0 = time.perf_counter()
            with torch.no_grad():
                for step in range(n_tr + n_me):
                    cond, delay_buf, spikes, v, refrac = model(
                        rates, cond, delay_buf, spikes, v, refrac, generator=gen)
                    if step >= n_tr:
                        for k, t in tens.items():
                            s = spikes[0, t]
                            acc[k] += s.sum()
                            act[k] += s
            t_sec = SIM_MS / 1000.0
            hz = {k: float(acc[k]) / len(stages[k]) / t_sec for k in tens}
            frac = {k: float((act[k] > 0).float().mean()) for k in tens}
            print(f"  {rate:>9.0f}" + "".join(f"{hz.get(k, float('nan')):>17.2f}" for k in names)
                  + f"   [{time.perf_counter() - t0:.0f} с]")
            print(f"  {'активных':>9s}"
                  + "".join(f"{frac.get(k, float('nan')):>16.1%} " for k in names))
            rows.append({"stim_set": set_name, "n_stim": len(stim_idx),
                         "stim_rate_hz": rate,
                         **{k: hz.get(k) for k in names},
                         **{f"active_{k}": frac.get(k) for k in names}})

    df = pd.DataFrame(rows)
    df.to_csv(out("flyvis_seam_probe.csv"), index=False)
    print(f"\nсохранено: {out('flyvis_seam_probe.csv')}")

    print("\n" + "=" * 78)
    print(" КРИТЕРИЙ ПРИЁМКИ")
    print("=" * 78)
    print("  Гибрид имеет смысл, если при физиологическом входе (25-100 Гц) LC своей")
    print("  стороны отзывается заметно выше нуля И заметно сильнее чужой стороны.")
    print("  Нисходящие — второй рубеж: если LC горит, а DN нет, стык придётся")
    print("  двигать ниже. Нулевая частота обязана давать нули везде.")

    z = df[df["stim_rate_hz"] == 0]
    if not z.empty and float(z[names].to_numpy().max()) > 0:
        print("\n  ВНИМАНИЕ: на нулевой частоте отклик ненулевой — это фон, а не сигнал.")


if __name__ == "__main__":
    main()
