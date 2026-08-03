"""Есть ли у обратной связи от ног настоящий адрес среди нисходящих?

Контекст. Стимуляция восходящих нейронов не вызывает у DNp09 ни одного спайка —
это измерено. Но обзор всех 1299 нисходящих показал, что 223 из них получают
существенный чистый вход от восходящих, причём у сильнейших зрительного входа
практически нет:

    DNg108   восходящие 1004/951   зрение 1/1
    DNge129  восходящие 903/651    зрение -17/-8
    DNg74_b  восходящие 757/655    зрение -1/-5
    DNp06    восходящие 702/637    зрение 1647/1535   (интегратор двух модальностей)

То есть прежний вывод про недостижимость верен ТОЛЬКО для DNp09. Здесь
проверяем гипотезу прямо: стимулируем восходящие и смотрим отклик этих
нейронов. DNp09 держим в списке как заведомо отрицательный контроль.

Если гипотеза подтвердится, обратную связь от лапок можно будет вести честно —
через нейроны, которые в живой мухе действительно слушают ноги, параллельным
каналом к зрительному.
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

TARGET_TYPES = ["DNg108", "DNge129", "DNg74_b", "DNge141", "DNp06", "DNp09"]
RATES_HZ = [0.0, 50.0, 100.0, 200.0]
DT = 0.1
SIM_MS = 1000.0
TRANSIENT_MS = 200.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print("=" * 78)
    print(" ВОСХОДЯЩИЕ -> НИСХОДЯЩИЕ, КОТОРЫЕ ДЕЙСТВИТЕЛЬНО СЛУШАЮТ НОГИ")
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

    # цели: по одному нейрону каждого типа с каждой стороны
    targets = []
    for ct in TARGET_TYPES:
        sub = ann[(ann["cell_type"] == ct) & (ann["super_class"] == "descending")]
        for side in ("left", "right"):
            s = sub[sub["side"] == side]
            if len(s):
                rid = int(s["root_id"].iloc[0])
                targets.append((f"{ct} {side}", flyid2i[rid]))
    if not targets:
        print("целевые нейроны не найдены")
        return
    print(f"целей: {len(targets)}")
    for name, _ in targets:
        print(f"    {name}")

    tgt_idx = torch.tensor([i for _, i in targets], dtype=torch.long, device=DEVICE)
    weights = rp.get_weights(str(path_con), str(path_comp), str(path_wt), csr=True).to(DEVICE)

    rows = []
    for stim_name, stim in (("восходящие слева", asc_l), ("восходящие справа", asc_r)):
        print(f"\n===== стимуляция: {stim_name} ({len(stim)} нейронов) =====")
        header = f"  {'вход, Гц':>9s}" + "".join(f"{nm[:13]:>14s}" for nm, _ in targets)
        print(header)
        for rate in RATES_HZ:
            model = rp.TorchModel(1, n, DT, rp.MODEL_PARAMS, weights,
                                  exc_indices=list(stim), device=DEVICE)
            cond, delay_buf, spikes, v, refrac = model.state_init()
            rates = torch.zeros(1, n, device=DEVICE)
            if rate > 0:
                rates[:, stim] = rate
            gen = torch.Generator(device=DEVICE)
            gen.manual_seed(4321)

            acc = torch.zeros(len(targets), device=DEVICE)
            n_trans, n_meas = int(TRANSIENT_MS / DT), int(SIM_MS / DT)
            t0 = time.perf_counter()
            with torch.no_grad():
                for step in range(n_trans + n_meas):
                    cond, delay_buf, spikes, v, refrac = model(
                        rates, cond, delay_buf, spikes, v, refrac, generator=gen)
                    if step >= n_trans:
                        acc.add_(spikes[0, tgt_idx])
            hz = [float(x) / (SIM_MS / 1000.0) for x in acc.tolist()]
            print(f"  {rate:>9.0f}" + "".join(f"{h:>14.1f}" for h in hz)
                  + f"   [{time.perf_counter() - t0:.0f} с]")
            for (nm, _), h in zip(targets, hz):
                rows.append({"stim": stim_name, "n_stim": len(stim),
                             "stim_rate_hz": rate, "target": nm, "target_hz": h})
            pd.DataFrame(rows).to_csv(out("ascending_to_leg_dns.csv"), index=False)

    print(f"\nсохранено: {out('ascending_to_leg_dns.csv')}")
    print("\nЧитать так: DNp09 обязан остаться на нуле — это известный")
    print("отрицательный контроль. Если DNg108 или DNge129 отзовутся, значит у")
    print("обратной связи от ног есть настоящий адрес, и её можно вести честно.")


if __name__ == "__main__":
    main()
