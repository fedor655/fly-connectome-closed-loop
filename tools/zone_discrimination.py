"""Этап 0: различает ли выход мозга подмножества зрительного входа.

Шлагбаум перед картой. Если стимуляция разных групп проекционных нейронов даёт
нисходящим один и тот же отклик, никакая ретинотопия не поможет: пространственная
информация не доходит до того места, откуда идут команды телу.

Группы здесь ПРОИЗВОЛЬНЫЕ — карта на этом этапе не нужна и намеренно не
используется, чтобы шлагбаум не зависел от её правильности.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flypaths import add_fly_brain_to_path, out  # noqa: E402
from tools.zone_probe import descending_indices  # noqa: E402
from closed_loop_vision import DT_BRAIN_MS, load_brain_assets  # noqa: E402

add_fly_brain_to_path()
import run_pytorch as rp  # noqa: E402


SWEEP_RATES = (25.0, 50.0, 100.0, 200.0)

# Рабочая точка стимуляции. Измерено развёрткой sweep_rate() по SWEEP_RATES:
# 25->50 Гц +57.8%, 50->100 Гц +41.5%, 100->200 Гц +37.4% — полки нет вплоть до
# верхней ступени, отклик продолжает расти, поэтому взята верхняя граница
# развёртки, 200 Гц (см. output/zone_rate_sweep.csv).
WORK_RATE_HZ = 200.0


def sweep_rate(assets, dn_idx, device, seed=0):
    """Выбрать рабочую точку: где отклик ещё растёт, а не упёрся в полку.

    4000 нейронов вместо 87 уводят сеть в насыщение — это уже наблюдалось при
    переходе на широкий зрительный вход, где частоты пришлось снизить вчетверо.
    Полка означает, что сеть отвечает одинаково на 50 и на 200 Гц, и никакая
    пространственная структура через неё не пройдёт.
    """
    grp = split_groups(np.array(assets["lc_l"]), n_groups=4, seed=0)[0]
    resp = probe_batched(assets["weights"], assets["n"], dn_idx,
                         [grp] * len(SWEEP_RATES), rate_hz=None, seed=seed,
                         device=device, rates_per_item=SWEEP_RATES)
    rows = []
    prev = None
    best = SWEEP_RATES[0]
    for rate, r in zip(SWEEP_RATES, resp):
        m = float(r.mean())
        grow = float("nan") if prev is None else (m - prev) / max(m, 1e-9)
        rows.append({"rate_hz": rate, "dn_mean_hz": m, "dn_active": int((r > 1.0).sum()),
                     "rel_growth": grow})
        if prev is None or grow > 0.02:
            best = rate
        prev = m
    return best, pd.DataFrame(rows)


def split_groups(idx, n_groups=4, seed=0):
    """Разбить индексы на равные произвольные группы.

    Произвольность здесь — свойство, а не небрежность: на этапе 0 проверяется
    сама различимость подмножеств, и группы по анатомии её бы подпёрли.
    """
    rng = np.random.default_rng(seed)
    return np.array_split(rng.permutation(np.asarray(idx)), n_groups)


def probe_batched(weights, n, dn_idx, stim_sets, rate_hz, seed, device,
                  transient_ms=300.0, measure_ms=500.0, rates_per_item=None):
    """Частоты нисходящих при стимуляции нескольких наборов сразу.

    Каждый элемент батча — свой набор стимуляции и свой поток пуассоновского
    шума. Батч нужен по цене: на этой машине одиночный прогон стоит 270 с на
    секунду симуляции, батч на 8 условий — около 60 с на условие.

    exc_indices общий на весь батч: это список нейронов, которым обнуляется
    рефрактерность, и он обязан покрывать объединение наборов. Нейрон чужого
    набора получает при этом нулевую частоту, то есть не стимулируется.
    """
    stim_sets = [np.asarray(s, dtype=np.int64) for s in stim_sets]
    batch = len(stim_sets)
    exc = sorted(set(np.concatenate(stim_sets).tolist()))
    model = rp.TorchModel(batch, n, DT_BRAIN_MS, rp.MODEL_PARAMS, weights,
                          exc_indices=exc, device=device)
    c, d, s, v, r = model.state_init()
    rates = torch.zeros(batch, n, device=device)
    per_item = rates_per_item if rates_per_item is not None else [rate_hz] * batch
    for b, (sset, hz) in enumerate(zip(stim_sets, per_item)):
        rates[b, torch.tensor(sset, dtype=torch.long, device=device)] = float(hz)
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    idx_t = torch.tensor(np.asarray(dn_idx), dtype=torch.long, device=device)
    acc = torch.zeros(batch, len(dn_idx), device=device)
    n_tr, n_me = int(transient_ms / DT_BRAIN_MS), int(measure_ms / DT_BRAIN_MS)
    with torch.no_grad():
        for step in range(n_tr + n_me):
            c, d, s, v, r = model(rates, c, d, s, v, r, generator=g)
            if step >= n_tr:
                acc.add_(s[:, idx_t])
    return np.array(acc.tolist()) / (measure_ms / 1000.0)


def self_check() -> None:
    """Батч обязан давать элементам независимый шум и тот же мозг.

    Поэлементного равенства с одиночным прогоном здесь НЕ требуется и быть его
    не может: torch.bernoulli на входе формы (batch, n) забирает batch*n
    бросков за шаг, поэтому со второго шага потоки шума расходятся. Измерено:
    отклик одной группы при разном шуме коррелирует на 0.789, двух разных
    групп — на 0.014. Требуется именно это: элементы батча независимы, а одна
    и та же группа в батче и поодиночке даёт согласованный отклик.
    """
    device = "cpu"
    assets = load_brain_assets(device, verbose=False)
    dn_idx, _ = descending_indices(device)
    groups = split_groups(np.array(assets["lc_l"]), n_groups=4, seed=0)

    c = [len(g) for g in groups]
    assert sum(c) == len(assets["lc_l"]), f"группы потеряли нейроны: {c}"
    assert max(c) - min(c) <= 1, f"группы не равны: {c}"

    kw = dict(rate_hz=150.0, seed=7, device=device,
              transient_ms=100.0, measure_ms=300.0)
    both = probe_batched(assets["weights"], assets["n"], dn_idx,
                         [groups[0], groups[1]], **kw)
    one0 = probe_batched(assets["weights"], assets["n"], dn_idx, [groups[0]], **kw)
    assert both.shape == (2, len(dn_idx)), f"форма отклика {both.shape}"
    assert both[0].sum() > 0 and both[1].sum() > 0, "группа не вызвала ни одного спайка"

    # элементы батча не должны быть копией друг друга
    assert not np.array_equal(both[0], both[1]), \
        "элементы батча совпали побитово: они делят один поток шума"

    # та же группа батчем и поодиночке: разный шум, но тот же мозг
    m_b, m_1 = float(both[0].mean()), float(one0[0].mean())
    assert abs(m_b - m_1) < 0.5 * max(m_b, m_1) + 1e-9, (
        f"батч и одиночный прогон разошлись по уровню: {m_b:.3f} против "
        f"{m_1:.3f} — батчевание меняет динамику, а не только шум")
    print(f"  батч[0] {m_b:.2f} Гц, одиночный {m_1:.2f} Гц, "
          f"батч[1] {float(both[1].mean()):.2f} Гц")

    best, sweep = sweep_rate(assets, dn_idx, device, seed=0)
    print(sweep.to_string(index=False))
    sweep.to_csv(out("zone_rate_sweep.csv"), index=False)
    assert sweep["dn_mean_hz"].iloc[-1] > sweep["dn_mean_hz"].iloc[0], \
        "отклик не растёт с частотой — стимуляция не доходит"
    print(f"рабочая точка: {best:.0f} Гц")
    print("самопроверка батчевого зонда: ОК")


if __name__ == "__main__":
    self_check()
