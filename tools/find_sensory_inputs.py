"""Поиск сенсорных популяций, у которых есть реальный путь к нисходящим нейронам P9.

Зачем. Сейчас обратная связь от тела подаётся в прямых партнёров P9 — это честно,
но бедно: шесть флагов касания. Чтобы дать мухе более богатый вход, надо знать,
какие сенсорные входы коннектома вообще способны влиять на P9. Предыдущее
измерение показало, что Sugar GRNs не влияют вовсе (0 спайков на любой частоте).

Проблема: аннотаций типов клеток в локальных данных нет, только ID нейронов.

Метод.
1. Кандидатов в сенсорные афференты определяем структурно: у сенсорного нейрона
   вход приходит ИЗВНЕ мозга, поэтому внутри коннектома у него нулевая входная
   степень. Это проверяемый признак, не требующий аннотаций.
2. Влияние на P9 считаем обратным распространением по ВХОД-НОРМИРОВАННОЙ матрице.
   Нормировка обязательна: наивное распространение |W|^k растёт экспоненциально
   (в прошлом анализе на 4 шаге получилось 1e9) и ничего не означает.
   Здесь строка нормирована на сумму входящих весов, поэтому величина остаётся
   долей влияния и её можно сравнивать между нейронами.
3. Топ-кандидатов проверяем симуляцией: стимулируем и смотрим отклик P9.
"""
import json
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
from benchmark import EXPERIMENTS, path_comp, path_con, path_wt  # noqa: E402
import run_pytorch as rp  # noqa: E402

P9_LEFT = 720575940627652358
P9_RIGHT = 720575940635872101

MAX_HOPS = 4
DECAY = 0.6          # вклад дальних шагов гасим, иначе всё сводится к среднему по сети
TOP_REPORT = 25
DT = 0.1
SIM_MS = 1000.0
TRANSIENT_MS = 200.0
TEST_RATE_HZ = 200.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print("=" * 78)
    print(" ПОИСК СЕНСОРНЫХ ВХОДОВ С ПУТЁМ К P9")
    print("=" * 78)

    df_comp = pd.read_csv(path_comp, index_col=0)
    flyid2i = {j: i for i, j in enumerate(df_comp.index)}
    i2flyid = {i: j for j, i in flyid2i.items()}
    n = len(flyid2i)
    idx_l, idx_r = flyid2i[P9_LEFT], flyid2i[P9_RIGHT]

    conn = pd.read_parquet(path_con, columns=[
        "Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
    pre = conn["Presynaptic_Index"].to_numpy()
    post = conn["Postsynaptic_Index"].to_numpy()
    w = conn["Excitatory x Connectivity"].to_numpy().astype(np.float32)
    del conn

    W_abs = sp.coo_matrix((np.abs(w), (post, pre)), shape=(n, n)).tocsr()

    # ---------- 1. структурная идентификация кандидатов в афференты ----------
    in_deg = np.diff(W_abs.indptr)                    # число входов у каждого нейрона
    out_deg = np.asarray((W_abs > 0).sum(axis=0)).ravel()  # число выходов

    print(f"\nнейронов: {n}, связей: {len(w)}")
    print(f"входная степень: медиана {np.median(in_deg):.0f}, "
          f"среднее {in_deg.mean():.1f}, максимум {in_deg.max()}")

    afferent = (in_deg == 0) & (out_deg > 0)
    print(f"\nкандидатов в сенсорные афференты (вход внутри мозга = 0, выход > 0): "
          f"{int(afferent.sum())}")
    print(f"из них с выходом >= 5 связей: {int((afferent & (out_deg >= 5)).sum())}")

    # сколько сахарных нейронов попало в кандидаты — проверка признака
    sugar = [flyid2i[x] for x in EXPERIMENTS["sugar"]["neu_exc"] if x in flyid2i]
    print(f"проверка признака: из 21 Sugar GRN афферентами опознаны "
          f"{int(afferent[sugar].sum())}")

    # ---------- 2. вход-нормированное обратное влияние на P9 ----------
    row_sum = np.asarray(W_abs.sum(axis=1)).ravel()
    row_sum[row_sum == 0] = 1.0
    T = sp.diags(1.0 / row_sum) @ W_abs      # T[post, pre] — доля входа post от pre
    Tt = T.T.tocsr()

    influence = np.zeros(n, dtype=np.float64)
    frontier = np.zeros(n, dtype=np.float64)
    frontier[[idx_l, idx_r]] = 0.5           # стартуем с пары P9

    print(f"\nобратное распространение по нормированной матрице, "
          f"{MAX_HOPS} шага, затухание {DECAY}")
    for hop in range(1, MAX_HOPS + 1):
        frontier = Tt @ frontier
        contrib = (DECAY ** (hop - 1)) * frontier
        influence += contrib
        print(f"  шаг {hop}: ненулевых источников {int((frontier > 0).sum()):>6d}, "
              f"суммарный вклад {contrib.sum():.4f}, максимум {contrib.max():.6f}")

    # ---------- 3. ранжирование афферентов ----------
    aff_idx = np.where(afferent)[0]
    aff_infl = influence[aff_idx]
    order = np.argsort(-aff_infl)

    print(f"\n===== топ-{TOP_REPORT} афферентов по влиянию на P9 =====")
    print(f"  {'flywire_id':>20s} {'влияние':>12s} {'выходов':>9s}")
    top_ids = []
    for k in order[:TOP_REPORT]:
        i = aff_idx[k]
        top_ids.append(int(i2flyid[i]))
        print(f"  {i2flyid[i]:>20d} {aff_infl[k]:>12.8f} {out_deg[i]:>9d}")

    sugar_infl = influence[sugar]
    print(f"\nдля сравнения, Sugar GRNs: влияние "
          f"{sugar_infl.min():.8f}..{sugar_infl.max():.8f} "
          f"(среднее {sugar_infl.mean():.8f})")
    print(f"топ-афферент сильнее среднего сахарного в "
          f"{aff_infl[order[0]] / max(sugar_infl.mean(), 1e-12):.0f} раз")

    # набор для проверки: топ-50 афферентов
    probe_n = min(50, len(order))
    probe_idx = [int(aff_idx[k]) for k in order[:probe_n]]

    # ---------- 4. проверка симуляцией ----------
    print(f"\n===== проверка симуляцией: стимулируем топ-{probe_n} афферентов "
          f"@ {TEST_RATE_HZ:.0f} Гц =====")
    del W_abs, T, Tt

    weights = rp.get_weights(str(path_con), str(path_comp), str(path_wt), csr=True).to(DEVICE)
    p9_idx = torch.tensor([idx_l, idx_r], dtype=torch.long, device=DEVICE)

    results = []
    for label, stim in (("топ-афференты", probe_idx),
                        ("Sugar GRNs (контроль)", sugar)):
        model = rp.TorchModel(1, n, DT, rp.MODEL_PARAMS, weights,
                              exc_indices=stim, device=DEVICE)
        cond, delay_buf, spikes, v, refrac = model.state_init()
        rates = torch.zeros(1, n, device=DEVICE)
        rates[:, stim] = TEST_RATE_HZ
        gen = torch.Generator(device=DEVICE)
        gen.manual_seed(777)

        acc = torch.zeros(2, device=DEVICE)
        pop = torch.zeros(n, device=DEVICE)
        n_trans, n_meas = int(TRANSIENT_MS / DT), int(SIM_MS / DT)
        t0 = time.perf_counter()
        with torch.no_grad():
            for step in range(n_trans + n_meas):
                cond, delay_buf, spikes, v, refrac = model(
                    rates, cond, delay_buf, spikes, v, refrac, generator=gen)
                if step >= n_trans:
                    acc += spikes[0, p9_idx]
                    pop += spikes[0]
        hz = [float(x) / (SIM_MS / 1000.0) for x in acc.tolist()]
        active = int((pop > 0).sum().item())
        print(f"  {label:>24s}: P9 L {hz[0]:6.1f} Гц, P9 R {hz[1]:6.1f} Гц, "
              f"активных нейронов {active:>6d}  [{time.perf_counter() - t0:.0f} с]")
        results.append({"population": label, "n_stim": len(stim),
                        "p9_left_hz": hz[0], "p9_right_hz": hz[1],
                        "n_active": active})

    out_csv = out("sensory_inputs_screen.csv")
    pd.DataFrame(results).to_csv(out_csv, index=False)
    out_json = out("p9_afferent_candidates.json")
    with open(out_json, "w") as f:
        json.dump({"top_afferents": [int(i2flyid[i]) for i in probe_idx],
                   "p9_left": P9_LEFT, "p9_right": P9_RIGHT}, f, indent=2)
    print(f"\nсохранено: {out_csv}")
    print(f"сохранено: {out_json}")


if __name__ == "__main__":
    main()
