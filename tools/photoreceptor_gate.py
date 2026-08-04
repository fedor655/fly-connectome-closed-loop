"""Доходит ли сигнал от фоторецепторов до нисходящих нейронов?

Это проверка-ворота, до всякого построения отображения омматидиев.

Причина сомневаться. Раньше измерено, что в этой LIF-модели сигнал затухает
быстро: стимуляция партнёров партнёров партнёров DNp09 (три синапса) давала
0-6 Гц против 146 Гц при одном синапсе. От фоторецепторов до нисходящих
ступеней БОЛЬШЕ: сетчатка -> ламина -> медулла -> лобула -> LC -> DN.
Если сигнал не доходит, строить ретинотопию бессмысленно.

Что меряем. Стимулируем фоторецепторы одного глаза на нескольких частотах и
смотрим отклик на каждой ступени: ламина (L1-L5), медулла (Mi, Tm), детекторы
движения (T4, T5), зрительные проекционные, нисходящие. Так будет видно не
только доходит или нет, но и ГДЕ обрывается, если обрывается.
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
SIM_MS = 1500.0
TRANSIENT_MS = 300.0
# Первый заход брал 50-200 Гц и показал, что сигнал глохнет за ламиной.
# Расчёт установившегося режима объяснил почему: ламине нужно около 5091/12.5
# то есть примерно 400 Гц на фоторецепторах, а мы давали вчетверо меньше.
# Фоторецепторы получают Пуассон напрямую (wScale*scalePoisson = 68.75 мВ на
# спайк, порог 7 мВ), поэтому они выдают почти всё, что им подали, и высокие
# частоты для них — вопрос настройки, а не физиологии.
RATES_HZ = [0.0, 200.0, 400.0, 600.0, 900.0]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print("=" * 78)
    print(" ВОРОТА: доходит ли сигнал от фоторецепторов до нисходящих")
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

    def idx_of(mask):
        return [flyid2i[int(x)] for x in ann.loc[mask, "root_id"]]

    # Стимулируем ЛЕВЫЙ глаз и меряем только левые ступени. Первый заход
    # усреднял по обоим полушариям при односторонней стимуляции — половина
    # нейронов не получала ничего и вдвое занижала все числа.
    L = ann["side"] == "left"
    photo_l = idx_of((ann["super_class"] == "sensory") &
                     (ann["cell_class"] == "visual") & L)
    stages = {
        "ламина L1-L5": idx_of(ct.str.match(r"^L[1-5]$") & L),
        "медулла Mi": idx_of(ct.str.match(r"^Mi\d") & L),
        "медулла Tm": idx_of(ct.str.match(r"^Tm\d") & L),
        "движение T4": idx_of(ct.str.match(r"^T4") & L),
        "движение T5": idx_of(ct.str.match(r"^T5") & L),
        "зрит. проекц.": idx_of((ann["super_class"] == "visual_projection") & L),
        "нисходящие": idx_of((ann["super_class"] == "descending") & L),
    }
    print(f"фоторецепторов левого глаза: {len(photo_l)}")
    for k, v in stages.items():
        print(f"  {k:<16s} {len(v):>7d}")

    weights = rp.get_weights(str(path_con), str(path_comp), str(path_wt), csr=True).to(DEVICE)
    names = list(stages)
    tens = {k: torch.tensor(v, dtype=torch.long, device=DEVICE) for k, v in stages.items() if v}

    print(f"\n  {'вход, Гц':>9s}" + "".join(f"{k[:14]:>15s}" for k in names))
    rows = []
    for rate in RATES_HZ:
        model = rp.TorchModel(1, n, DT, rp.MODEL_PARAMS, weights,
                              exc_indices=photo_l, device=DEVICE)
        cond, delay_buf, spikes, v, refrac = model.state_init()
        rates = torch.zeros(1, n, device=DEVICE)
        if rate > 0:
            rates[:, photo_l] = rate
        gen = torch.Generator(device=DEVICE)
        gen.manual_seed(777)

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
        vals = {k: float(acc[k]) / len(stages[k]) / t_sec for k in tens}
        frac = {k: float((act[k] > 0).float().mean()) for k in tens}
        print(f"  {rate:>9.0f}" + "".join(f"{vals.get(k, float('nan')):>15.2f}" for k in names)
              + f"   [{time.perf_counter() - t0:.0f} с]")
        print(f"  {'доля активных':>9s}"
              + "".join(f"{frac.get(k, float('nan')):>14.1%} " for k in names))
        rows.append({"stim_rate_hz": rate,
                     **{k: vals.get(k) for k in names},
                     **{f"active_{k}": frac.get(k) for k in names}})

    pd.DataFrame(rows).to_csv(out("photoreceptor_gate.csv"), index=False)
    print(f"\nсохранено: {out('photoreceptor_gate.csv')}")
    print("\nЧисла — средняя частота нейрона ступени, Гц. Читать так: где столбец")
    print("падает до нуля, там сигнал и обрывается. Если нисходящие отзываются,")
    print("настоящее зрение через оптические доли построить можно.")


if __name__ == "__main__":
    main()
