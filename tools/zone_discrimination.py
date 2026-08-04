"""Этап 0: различает ли выход мозга подмножества зрительного входа.

Шлагбаум перед картой. Если стимуляция разных групп проекционных нейронов даёт
нисходящим один и тот же отклик, никакая ретинотопия не поможет: пространственная
информация не доходит до того места, откуда идут команды телу.

Группы здесь ПРОИЗВОЛЬНЫЕ — карта на этом этапе не нужна и намеренно не
используется, чтобы шлагбаум не зависел от её правильности.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flypaths import add_fly_brain_to_path, out  # noqa: E402
from tools.zone_probe import descending_indices  # noqa: E402
from tools.replicate_vision import welch  # noqa: E402
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


def correlations(resp, groups_per_side=4):
    """Внутригрупповые и межгрупповые корреляции откликов.

    resp[side][seed] — матрица (группа, нисходящий). Центрирование по каждому
    нисходящему убирает общую составляющую: вопрос не в том, кто вообще горит
    от зрения, а в том, различается ли КАРТИНА по группам.
    """
    rows = []
    for side, per_seed in resp.items():
        allc = np.concatenate(list(per_seed.values()), axis=0)   # (seed*группа, DN)
        keep = allc.max(axis=0) > 1.0
        assert keep.sum() >= 20, f"{side}: активных нисходящих всего {keep.sum()}"
        mu = allc[:, keep].mean(axis=0)
        cen = {s: m[:, keep] - mu for s, m in per_seed.items()}
        seeds = sorted(cen)
        for g in range(groups_per_side):
            for i, si in enumerate(seeds):
                for sj in seeds[i + 1:]:
                    rows.append({"kind": "within", "side": side, "pair": f"g{g}:{si}-{sj}",
                                 "corr": float(np.corrcoef(cen[si][g], cen[sj][g])[0, 1])})
        for s in seeds:
            for a in range(groups_per_side):
                for b in range(a + 1, groups_per_side):
                    rows.append({"kind": "between", "side": side, "pair": f"s{s}:g{a}-g{b}",
                                 "corr": float(np.corrcoef(cen[s][a], cen[s][b])[0, 1])})
    return pd.DataFrame(rows)


def self_check() -> None:
    """Батч обязан давать элементам независимый шум и тот же мозг.

    Поэлементного равенства с одиночным прогоном здесь НЕ требуется и быть его
    не может: torch.bernoulli на входе формы (batch, n) забирает batch*n
    бросков за шаг, поэтому со второго шага потоки шума расходятся. Измерено:
    отклик одной группы при разном шуме коррелирует на 0.789, двух разных
    групп — на 0.014. Требуется именно это: элементы батча независимы, а одна
    и та же группа в батче и поодиночке даёт согласованный отклик.
    """
    # контроль разбора статистики идёт первым: он не требует мозга, это чистая
    # арифметика numpy, и падать на нём нужно за секунду, а не после ~12 минут
    # мозговой части ниже.
    rng = np.random.default_rng(0)
    base = rng.random((4, 200)) * 50.0
    fake = {"left": {s: base + rng.normal(0, 1.0, base.shape) for s in (0, 1, 2)}}
    df = correlations(fake)
    w = df.loc[df.kind == "within", "corr"].mean()
    b = df.loc[df.kind == "between", "corr"].mean()
    assert w > b + 0.3, f"разбор не отличает группы на синтетике: within {w:.3f}, between {b:.3f}"
    # обратный контроль: если групп нет, разбор не должен их выдумывать
    flat = {"left": {s: np.tile(base.mean(axis=0), (4, 1))
                     + rng.normal(0, 1.0, base.shape) for s in (0, 1, 2)}}
    d2 = correlations(flat)
    assert abs(d2.loc[d2.kind == "within", "corr"].mean()
               - d2.loc[d2.kind == "between", "corr"].mean()) < 0.2, \
        "разбор выдумывает различие там, где групп нет"

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


def main():
    device = "cpu"
    print("=" * 78)
    print(" ЭТАП 0: различает ли выход мозга подмножества зрительного входа")
    print("=" * 78)
    assets = load_brain_assets(device)
    dn_idx, _ = descending_indices(device)
    print(f"нисходящих под наблюдением: {len(dn_idx)}")

    resp = {}
    for side in ("left", "right"):
        src = np.array(assets["lc_l"] if side == "left" else assets["lc_r"])
        groups = split_groups(src, n_groups=4, seed=0)
        print(f"{side}: {len(src)} проекционных -> группы {[len(g) for g in groups]}")
        resp[side] = {}
        for seed in (0, 1, 2):
            t0 = time.perf_counter()
            resp[side][seed] = probe_batched(
                assets["weights"], assets["n"], dn_idx, list(groups),
                rate_hz=WORK_RATE_HZ, seed=1000 + seed, device=device)
            print(f"  seed={seed}: средний отклик "
                  f"{resp[side][seed].mean():.2f} Гц, активных "
                  f"{int((resp[side][seed] > 1.0).any(axis=0).sum())} "
                  f"[{time.perf_counter() - t0:.0f} с]")

    df = correlations(resp)
    df.to_csv(out("zone_discrimination.csv"), index=False)

    w = df.loc[df.kind == "within", "corr"]
    b = df.loc[df.kind == "between", "corr"]
    t, dof = welch(b, w)
    print("\n" + "=" * 78)
    print(" РЕЗУЛЬТАТ")
    print("=" * 78)
    print(f"  внутригрупповая (потолок шума): {w.mean():+.3f} ± {w.std():.3f}  n={len(w)}")
    print(f"  межгрупповая:                   {b.mean():+.3f} ± {b.std():.3f}  n={len(b)}")
    print(f"  Уэлч: t={t:+.2f}, dof={dof:.1f}")
    print(f"  критерий |t| > 3 -> {'ПРОЙДЕН' if abs(t) > 3 else 'НЕ ПРОЙДЕН'}")
    if abs(t) <= 3:
        print("\n  Карта не строится. Записать отрицательный результат с числами")
        print("  в PROJECT_LOG.md и остановиться.")
    print(f"\nсохранено: {out('zone_discrimination.csv')}")


if __name__ == "__main__":
    if "--check" in sys.argv:
        self_check()
    else:
        main()
