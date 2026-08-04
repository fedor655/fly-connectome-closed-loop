# Пространственное зрение на уровне зрительных проекционных нейронов — план реализации

> **Для агентов:** ОБЯЗАТЕЛЬНЫЙ ПОД-SKILL: используйте superpowers:subagent-driven-development (рекомендуется) или superpowers:executing-plans, чтобы выполнять план задача за задачей. Шаги размечены чекбоксами (`- [ ]`).

**Цель:** разные участки поля зрения дают разную стимуляцию разным зрительным проекционным нейронам, и это проверено на выходе мозга и на поведении.

**Архитектура:** положение нейрона в поле зрения выводится не из его собственных координат, а распространением по синапсам от сетчатки, где лист координат проверен и ориентирован по кайме DRA. Полученные положения режутся на 4 полосы по квантилям; яркость полосы омматидиев подаётся своей группе проекционных нейронов. Оптические доли не задействованы: стимуляция идёт прямо в проекционные, как в работающем `closed_loop_vision.py`.

**Стек:** python 3.12 в `.venv`, torch 2.13 (CPU, CUDA на macOS нет), numpy, pandas, mujoco 3.9.0, flygym 2.1.0. Модель мозга — `TorchModel` из `fly-brain/code/run_pytorch.py`.

**Спека:** [docs/superpowers/specs/2026-08-04-vision-spatial-b-design.md](../specs/2026-08-04-vision-spatial-b-design.md)

## Global Constraints

- Ветка `vision-spatial-b`. Перед push — `git fetch` и `rebase`, не force: в `main` пушут два человека.
- Новых зависимостей не добавлять. Доступны: mujoco 3.9.0, flygym 2.1.0, torch, numpy, pandas, scipy, imageio.
- `TorchModel` не переписывать. Предшественник переписал и потерял `wScale=0.275` и `time_factor_mem=0.005`, получив в 1204 раза больше спайков.
- `flypaths` импортируется раньше `flygym` и `mujoco` — он ставит `MUJOCO_GL`.
- Запуск только через `.venv/bin/python`. Интерактивные окна mujoco — через `.venv/bin/mjpython`.
- Координаты FlyWire — воксели FAFB, анизотропные: x и y по 4 нм, z по 40 нм. Любая геометрия считается после умножения на `VOXEL_NM = np.array([4.0, 4.0, 40.0])`.
- Комментарии и вывод по-русски. Числа приводить, не пересказывать словами.
- Тесты по конвенции проекта: `self_check()` под `if __name__ == "__main__"` внутри инструмента плюс один `test_*.py` в корне с голыми `assert`. Ни pytest, ни фикстур — в проекте их нет.
- Кэши в `output/` в git не идут (они производные от аннотаций, которых в git тоже нет).
- Правило проекта: критерий приёмки формулируется ДО прогона. Все пороги ниже уже зафиксированы; менять их по результату нельзя.

---

### Задача 1: батчевый зонд и рабочая точка по частоте

Одиночный прогон мозга на этой машине стоит 270 с на секунду симуляции. Батч на 8 условий даёт около 60 с на условие. Без него этап 0 не влезает в разумное время.

**Файлы:**
- Создать: `tools/zone_discrimination.py`
- Проверка: `self_check()` в том же файле

**Интерфейсы:**
- Использует: `closed_loop_vision.load_brain_assets(device, verbose=...)` → dict с ключами `weights`, `n`, `lc_l`, `lc_r`; `tools.zone_probe.descending_indices(device)` → `(idx: np.ndarray[int64], side: np.ndarray[str])`
- Даёт: `probe_batched(weights, n, dn_idx, stim_sets, rate_hz, seed, device, transient_ms=300.0, measure_ms=500.0) -> np.ndarray` формы `(len(stim_sets), len(dn_idx))`, частоты в Гц; `split_groups(idx, n_groups=4, seed=0) -> list[np.ndarray]`

- [ ] **Шаг 1: написать падающую проверку**

Создать `tools/zone_discrimination.py` и положить в него только проверку:

```python
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
    print("самопроверка батчевого зонда: ОК")


if __name__ == "__main__":
    self_check()
```

- [ ] **Шаг 2: убедиться, что проверка падает**

Запустить:

```bash
.venv/bin/python tools/zone_discrimination.py
```

Ожидается: `NameError: name 'split_groups' is not defined`.

- [ ] **Шаг 3: написать минимальную реализацию**

Вставить перед `self_check()`:

```python
def split_groups(idx, n_groups=4, seed=0):
    """Разбить индексы на равные произвольные группы.

    Произвольность здесь — свойство, а не небрежность: на этапе 0 проверяется
    сама различимость подмножеств, и группы по анатомии её бы подпёрли.
    """
    rng = np.random.default_rng(seed)
    return np.array_split(rng.permutation(np.asarray(idx)), n_groups)


def probe_batched(weights, n, dn_idx, stim_sets, rate_hz, seed, device,
                  transient_ms=300.0, measure_ms=500.0):
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
    for b, sset in enumerate(stim_sets):
        rates[b, torch.tensor(sset, dtype=torch.long, device=device)] = rate_hz
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
```

- [ ] **Шаг 4: запустить проверку**

```bash
.venv/bin/python tools/zone_discrimination.py
```

Ожидается строка с тремя частотами и `самопроверка батчевого зонда: ОК`. Время — около полутора минут.

Если падает ассерт про уровень — батчевание меняет не только шум, и весь этап 0 придётся гонять при batch=1 (в 6 раз дольше, около двух часов). Проверять модель на этот случай не надо: она валидирована, и переписывать её нельзя.

- [ ] **Шаг 5: добавить развёртку по частоте**

Дописать в файл перед `self_check()`:

```python
SWEEP_RATES = (25.0, 50.0, 100.0, 200.0)

# Рабочая точка стимуляции. Значение ниже — только чтобы файл импортировался до
# первой развёртки; шаг 6 заменяет его на измеренное, с приростом отклика в
# комментарии. 100 Гц взято как середина развёртки, а не как результат.
WORK_RATE_HZ = 100.0


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
```

Это требует, чтобы `probe_batched` умел разные частоты по элементам батча. Заменить в `probe_batched` сигнатуру и заполнение частот:

```python
def probe_batched(weights, n, dn_idx, stim_sets, rate_hz, seed, device,
                  transient_ms=300.0, measure_ms=500.0, rates_per_item=None):
```

и

```python
    per_item = rates_per_item if rates_per_item is not None else [rate_hz] * batch
    for b, (sset, hz) in enumerate(zip(stim_sets, per_item)):
        rates[b, torch.tensor(sset, dtype=torch.long, device=device)] = float(hz)
```

- [ ] **Шаг 6: прогнать развёртку и записать рабочую точку**

Дописать в `self_check()` перед `print`:

```python
    best, sweep = sweep_rate(assets, dn_idx, device, seed=0)
    print(sweep.to_string(index=False))
    sweep.to_csv(out("zone_rate_sweep.csv"), index=False)
    assert sweep["dn_mean_hz"].iloc[-1] > sweep["dn_mean_hz"].iloc[0], \
        "отклик не растёт с частотой — стимуляция не доходит"
    print(f"рабочая точка: {best:.0f} Гц")
```

Запустить:

```bash
.venv/bin/python tools/zone_discrimination.py
```

Ожидается: таблица из 4 строк, `рабочая точка: <число> Гц`, файл `output/zone_rate_sweep.csv`. Время — около 6 минут.

Записать выбранную частоту в константу `WORK_RATE_HZ` в начале файла с комментарием, откуда она взялась и какой был прирост на верхней ступени.

- [ ] **Шаг 7: коммит**

```bash
git add tools/zone_discrimination.py output/zone_rate_sweep.csv
git commit -m "Батчевый зонд нисходящих и развёртка по частоте"
```

---

### Задача 2: этап 0 — шлагбаум различимости

**Файлы:**
- Изменить: `tools/zone_discrimination.py` (добавить `main()`)
- Результат: `output/zone_discrimination.csv`

**Интерфейсы:**
- Использует: `probe_batched`, `split_groups`, `sweep_rate` из задачи 1; `tools.replicate_vision.welch(a, b) -> (t, dof)`
- Даёт: `output/zone_discrimination.csv` с колонками `kind` (`within`/`between`), `side`, `pair`, `corr`

**Критерий, зафиксирован до прогона.** Отклик — вектор частот 1291 нисходящего. Первичная величина — корреляция Пирсона векторов, у которых из каждой компоненты вычтено её среднее по всем условиям своей стороны (общая составляющая «любой зрительный вход зажигает вот этих» не несёт информации о группе). Берутся только нисходящие с частотой выше 1 Гц хотя бы в одном условии — иначе тысячи нулей вытянут корреляцию к единице сами по себе.

- внутригрупповая корреляция: одна группа, разные seed — это ИЗМЕРЯЕМЫЙ потолок пуассоновского шума;
- межгрупповая: разные группы, один seed, одна сторона;
- **группы различимы, если межгрупповая ниже внутригрупповой по Уэлчу |t| > 3.**

Если не пройдено — карта не строится, отрицательный результат с числами пишется в `PROJECT_LOG.md`, работа останавливается.

- [ ] **Шаг 1: написать падающую проверку разбора статистики**

Дописать **только это** в `self_check()` файла `tools/zone_discrimination.py`, перед финальным `print`. Саму функцию `correlations` пока не писать — она появится на шаге 3.

```python
    # контроль разбора: четыре группы, различающиеся заведомо, и шум поверх
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
```

Второй контроль обязателен: без него проверка подтверждает только то, что разбор что-то считает, а не то, что он считает правильно.

- [ ] **Шаг 2: убедиться, что проверка падает**

```bash
.venv/bin/python tools/zone_discrimination.py --check
```

Ожидается: `NameError: name 'correlations' is not defined`.

- [ ] **Шаг 3: написать разбор**

Вставить в `tools/zone_discrimination.py` перед `self_check()`:

```python
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
```

- [ ] **Шаг 4: убедиться, что проверка проходит**

```bash
.venv/bin/python tools/zone_discrimination.py --check
```

Ожидается: `самопроверка батчевого зонда: ОК` без падений на синтетических контролях.

- [ ] **Шаг 5: написать основной прогон**

Дописать в конец файла, заменив блок `if __name__`:

```python
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
```

Добавить в импорты `import time` и `from tools.replicate_vision import welch`.

- [ ] **Шаг 6: прогнать этап 0**

```bash
.venv/bin/python tools/zone_discrimination.py
```

Время — около 20 минут. Ожидается таблица корреляций и вердикт по |t|.

**Это шлагбаум.** Если `НЕ ПРОЙДЕН` — остановиться, записать в `PROJECT_LOG.md` числа (внутригрупповая, межгрупповая, t, dof, число активных нисходящих, рабочая частота) и не выполнять задачи 3–7.

- [ ] **Шаг 7: коммит**

```bash
git add tools/zone_discrimination.py output/zone_discrimination.csv
git commit -m "Этап 0: различимость произвольных групп проекционных на выходе мозга"
```

---

### Задача 3: карта поля зрения и split-half контроль оси

**Файлы:**
- Создать: `tools/visual_field_map.py`
- Результат: `output/visual_field_map.npz` (кэш, в git не идёт), `output/visual_field_split_half.csv`

**Интерфейсы:**
- Использует: `flypaths.ANNOTATIONS`, `benchmark.path_comp`, `benchmark.path_con`
- Даёт:
  - `VOXEL_NM: np.ndarray` формы (3,)
  - `sheet_axes(retina: pd.DataFrame) -> (mu: (3,), E: (3,), A: (3,))` — центр листа, единичное направление элевации (дорсально), единичное направление азимута (антериально)
  - `field_positions(ann, con, side) -> (vp, axes, src, vp_ids)` — `vp: dict[root_id -> np.ndarray(2,)]` положения проекционных нейронов в осях (элевация, азимут); `axes: (mu, E, A)`; `src: dict[root_id -> np.ndarray(2,)]` положения ламины и медуллы, из которых считались проекционные (нужны split-half контролю, чтобы не пересчитывать цепочку); `vp_ids: set[int]`
  - `split_half(ann, con, side, seed=0) -> dict` с ключами `r_elev`, `r_azim`, `n`
  - `strips(vals: np.ndarray, n=4) -> np.ndarray[int]`
  - `self_check() -> None`

**Критерий, зафиксирован до прогона.**
- **Азимут проходит при r > 0.5 на обоих полушариях.** Это проверяемая величина.
- **Элевация — положительный контроль метода, обязана дать r > 0.9 на обоих полушариях.** Если не даёт — сломан расчёт, чинить надо расчёт, а не уходить в запасной вариант.
- Прошёл азимут → ось карты `azimuth`. Не прошёл → ось `elevation`, и приёмка этапа 2 меняется на «объект вверху против объекта внизу».

- [ ] **Шаг 1: написать падающую проверку осей**

Создать `tools/visual_field_map.py`:

```python
"""Карта поля зрения: какому проекционному нейрону какой участок подавать.

Из координат самих проекционных нейронов ретинотопию взять нельзя — закрыто в
tools/lc_retinotopy.py (доли дисперсии 0.913/0.078/0.008, почти линия; координаты
это точки у сомы). Из координат сом медуллы — тоже нельзя: сохраняется одна ось
из двух (R2 0.94-0.98 против 0.02-0.24 на пяти типах клеток и двух полушариях).

Поэтому координаты сом промежуточных ступеней не используются вовсе. Положение
распространяется по синапсам от сетчатки, где лист проверен (третья главная ось
0.000-0.005) и ориентирован каймой DRA.

Оптические доли здесь не задействованы: карта решает только, кому что подавать,
а стимуляция идёт прямо в проекционные нейроны.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flypaths import ANNOTATIONS, OUTPUT_DIR, add_fly_brain_to_path, out  # noqa: E402

add_fly_brain_to_path()
from benchmark import path_comp, path_con  # noqa: E402

# Воксели FAFB анизотропны: x и y по 4 нм, z по 40 нм. Без приведения размах по
# z выходит 3949 против 110905 по x, и облако кажется плоским, хотя это единицы.
VOXEL_NM = np.array([4.0, 4.0, 40.0])

# Анатомические реперы (смещение центра популяции от центра мозга, нм):
#   оцеллярные фоторецепторы, нерв OCN:  y = -167303  -> дорсально это -y
#   антеннальный нерв AN:                z = -116658  -> антериально это -z
#   шейный коннектив CV:                 z =  +40200, y = +54128 (зад и низ)
DORSAL_AXIS, DORSAL_SIGN = 1, -1.0
ANTERIOR_AXIS, ANTERIOR_SIGN = 2, -1.0

N_STRIPS = 4
MEDULLA_CLASSES = ("ME", "ME>LO", "ME>LOP", "ME>LO.LOP")
CACHE = OUTPUT_DIR / "visual_field_map.npz"
OMMATIDIA_MAP = (Path(__import__("flygym").__file__).parent /
                 "assets/model/neuromechfly/vision/ommatidia_id_map.npy")

# Ось карты. Значение ниже — только чтобы файл импортировался до split-half
# контроля; шаг 6 задачи 3 заменяет его на выбранное контролем, с числами
# r_elev и r_azim в комментарии. Заранее объявленный запасной вариант —
# "elevation".
MAP_AXIS = "azimuth"


def self_check() -> None:
    """Ломается, если оси листа определены неверно."""
    _, ann, con = load_tables()
    for side in ("left", "right"):
        ret = retina_of(ann, side)
        mu, E, A = sheet_axes(ret)
        assert abs(np.linalg.norm(E) - 1) < 1e-9 and abs(np.linalg.norm(A) - 1) < 1e-9
        assert abs(E @ A) < 1e-9, f"{side}: оси листа не ортогональны"
        assert E[DORSAL_AXIS] * DORSAL_SIGN > 0, \
            f"{side}: элевация по кайме DRA смотрит не туда, куда оцелли"
        assert A[ANTERIOR_AXIS] * ANTERIOR_SIGN > 0, \
            f"{side}: азимут смотрит не в сторону антенн"
        X = ret[["pos_x", "pos_y", "pos_z"]].to_numpy(float) * VOXEL_NM - mu
        dra = ret["cell_sub_class"].to_numpy() == "DRA"
        assert dra.sum() > 50, f"{side}: каймы DRA всего {dra.sum()} нейронов"
        pe = (X @ E)
        pct = float((pe < pe[dra].mean()).mean() * 100)
        assert pct > 75, f"{side}: кайма DRA не у дорсального края, процентиль {pct:.0f}"
        print(f"  {side}: DRA на {pct:.0f}-м процентиле по элевации")
    print("самопроверка осей листа: ОК")


if __name__ == "__main__":
    self_check()
```

- [ ] **Шаг 2: убедиться, что проверка падает**

```bash
.venv/bin/python tools/visual_field_map.py
```

Ожидается: `NameError: name 'load_tables' is not defined`.

- [ ] **Шаг 3: реализовать оси**

Вставить перед `self_check()`:

```python
def load_tables():
    """Аннотации и связи, отфильтрованные до нейронов нашей модели."""
    comp = pd.read_csv(path_comp, index_col=0)
    flyid2i = {int(j): i for i, j in enumerate(comp.index)}
    ann = pd.read_csv(ANNOTATIONS, sep="\t", low_memory=False)
    ann["root_id"] = pd.to_numeric(ann["root_id"], errors="coerce")
    ann = ann.dropna(subset=["root_id"])
    ann["root_id"] = ann["root_id"].astype("int64")
    ann = ann[ann["root_id"].isin(flyid2i)].drop_duplicates("root_id")
    con = pd.read_parquet(path_con,
                          columns=["Presynaptic_ID", "Postsynaptic_ID", "Connectivity"])
    return flyid2i, ann, con[con["Connectivity"] > 0]


def retina_of(ann, side):
    return ann[(ann["side"] == side) & (ann["super_class"] == "sensory")
               & (ann["cell_class"] == "visual") & ann["pos_x"].notna()]


def sheet_axes(retina):
    """Оси листа сетчатки, привязанные к анатомии, а не к номеру компоненты.

    Брать первую и вторую главные компоненты нельзя: у двух глаз они нумеруются
    по-разному, и «ось 1» слева и справа означает разное (наблюдалось: контроль
    дал 0.912 против 0.585). Элевация определяется каймой DRA — dorsal rim area
    лежит у дорсального края глаза по построению. Азимут — перпендикуляр к ней
    в плоскости листа, знак по антеннальному нерву.
    """
    X = retina[["pos_x", "pos_y", "pos_z"]].to_numpy(float) * VOXEL_NM
    mu = X.mean(axis=0)
    P = np.linalg.svd(X - mu, full_matrices=False)[2][:2]     # плоскость листа
    dra = retina["cell_sub_class"].to_numpy() == "DRA"
    d = (X[dra].mean(axis=0) - mu) @ P.T
    e2 = d / np.linalg.norm(d)                                 # к кайме = дорсально
    a2 = np.array([-e2[1], e2[0]])                             # поворот на 90°
    E, A = e2 @ P, a2 @ P
    E /= np.linalg.norm(E)
    A /= np.linalg.norm(A)
    if A[ANTERIOR_AXIS] * ANTERIOR_SIGN < 0:
        A = -A                                                 # знак азимута по антеннам
    return mu, E, A
```

Знак элевации не переворачивается намеренно: он задан каймой, и совпадение с оцеллями — независимая проверка в `self_check()`, а не подгонка.

- [ ] **Шаг 4: убедиться, что проверка проходит**

```bash
.venv/bin/python tools/visual_field_map.py
```

Ожидается два вывода вида `left: DRA на 87-м процентиле по элевации` и `самопроверка осей листа: ОК`. Время — около 20 секунд.

- [ ] **Шаг 5: добавить цепочку положений и split-half контроль**

Вставить перед `self_check()`:

```python
def _wmean(edges, srcpos):
    """Взвешенное среднее положений пресинаптических партнёров."""
    p = np.array([srcpos[i] for i in edges["Presynaptic_ID"]])
    w = edges["Connectivity"].to_numpy(float)
    d = pd.DataFrame({"post": edges["Postsynaptic_ID"].to_numpy(),
                      "e": p[:, 0] * w, "a": p[:, 1] * w, "w": w}).groupby("post").sum()
    return {int(k): np.array([r.e / r.w, r.a / r.w]) for k, r in d.iterrows()}


def _edges_to(con, targets, srcpos):
    tg = set(int(x) for x in targets)
    return con[con["Postsynaptic_ID"].isin(tg) & con["Presynaptic_ID"].isin(srcpos)]


def field_positions(ann, con, side):
    """Положение каждого проекционного нейрона в поле зрения.

    Цепочка: сетчатка (координаты) -> ламина -> медулла -> проекционные.
    Координаты сом ламины и медуллы не используются: из них восстанавливается
    только одна ось из двух.
    """
    A_ = ann[ann["side"] == side]
    ret = retina_of(ann, side)
    mu, E, Az = sheet_axes(ret)
    X = ret[["pos_x", "pos_y", "pos_z"]].to_numpy(float) * VOXEL_NM - mu
    pos = {int(r): np.array([x @ E, x @ Az]) for r, x in zip(ret["root_id"], X)}

    lam = _wmean(_edges_to(con, A_.loc[A_["cell_class"] == "LA>ME", "root_id"], pos), pos)
    src1 = {**pos, **lam}
    med = _wmean(_edges_to(con, A_.loc[A_["cell_class"].isin(MEDULLA_CLASSES),
                                       "root_id"], src1), src1)
    src2 = {**med, **lam}
    vp_ids = A_.loc[A_["super_class"] == "visual_projection", "root_id"]
    vp = _wmean(_edges_to(con, vp_ids, src2), src2)
    return vp, (mu, E, Az), src2, set(int(x) for x in vp_ids)


def split_half(ann, con, side, seed=0):
    """Положение по случайной половине входов против второй половины.

    Если ось несёт сигнал, две независимые оценки совпадут. Если ось — шум,
    половины разойдутся. Элевация здесь положительный контроль метода: она
    обязана дать r > 0.9, иначе сломан расчёт, а не ось.
    """
    _, _, src, vp_ids = field_positions(ann, con, side)
    e = _edges_to(con, vp_ids, src)
    rng = np.random.default_rng(seed)
    m = rng.random(len(e)) < 0.5
    a, b = _wmean(e[m], src), _wmean(e[~m], src)
    both = sorted(set(a) & set(b))
    A_ = np.array([a[i] for i in both])
    B_ = np.array([b[i] for i in both])
    return {"side": side, "n": len(both),
            "r_elev": float(np.corrcoef(A_[:, 0], B_[:, 0])[0, 1]),
            "r_azim": float(np.corrcoef(A_[:, 1], B_[:, 1])[0, 1])}


def strips(vals, n=N_STRIPS):
    """Номер полосы по квантилям: полосы равны по населению, а не по координате.

    Покрытие листа неравномерное (отношение максимума к медиане расстояния до
    соседа 7-12), поэтому резать по координате нельзя — крайние полосы выйдут
    полупустыми.
    """
    vals = np.asarray(vals, float)
    q = np.quantile(vals, np.linspace(0.0, 1.0, n + 1)[1:-1])
    return np.searchsorted(q, vals).astype(int)
```

- [ ] **Шаг 6: прогнать контроль и зафиксировать ось**

Дописать в `self_check()` перед последним `print`:

```python
    rows = [split_half(ann, con, s, seed=0) for s in ("left", "right")]
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(out("visual_field_split_half.csv"), index=False)
    assert df["r_elev"].min() > 0.9, (
        f"положительный контроль провален: элевация {df['r_elev'].min():.3f} < 0.9. "
        "Сломан расчёт, а не ось — чинить расчёт, в запасной вариант не уходить")
    axis = "azimuth" if df["r_azim"].min() > 0.5 else "elevation"
    print(f"ось карты: {axis} (азимут {df['r_azim'].min():.3f}, порог 0.5)")
```

Запустить:

```bash
.venv/bin/python tools/visual_field_map.py
```

Ожидается таблица из двух строк с `r_elev` и `r_azim` и строка `ось карты: ...`. Время — около минуты.

Записать полученную ось в константу `MAP_AXIS` в начале файла с числами контроля в комментарии.

- [ ] **Шаг 7: коммит**

```bash
git add tools/visual_field_map.py output/visual_field_split_half.csv
git commit -m "Карта поля зрения: оси по кайме DRA, положения по цепочке синапсов, split-half контроль"
```

---

### Задача 4: полосы омматидиев и соответствие, измеренное на теле

Карта даёт полосы со стороны коннектома. Со стороны симулятора надо знать, какая ось картинки омматидиев отвечает за азимут и где у неё перёд. Это измеряется, а не постулируется.

**Файлы:**
- Изменить: `tools/visual_field_map.py`
- Результат: `output/visual_field_map.npz`

**Интерфейсы:**
- Использует: `flyreplay.build_scene(pillar_xy, geom_pos=None) -> Simulation`
- Даёт:
  - `measure_ommatidia_axis() -> dict` с ключами `axis` (0 — строки, 1 — столбцы), `flip` (bool), `contrast` (float)
  - `ommatidia_strips(axis, flip, n=N_STRIPS) -> np.ndarray` формы (721,)
  - `load_or_build() -> dict` с ключами `axis`, `ommatidia`, `left_idx`, `left_strip`, `right_idx`, `right_strip`

- [ ] **Шаг 1: написать падающую проверку**

Дописать в `self_check()` файла `tools/visual_field_map.py`:

```python
    m = load_or_build()
    om = m["ommatidia"]
    assert om.shape == (721,), f"омматидиев должно быть 721, получено {om.shape}"
    c = np.bincount(om, minlength=N_STRIPS)
    assert c.min() * 2 >= c.max(), f"полосы поля зрения слишком неравны: {c.tolist()}"
    for side in ("left", "right"):
        idx, st = m[f"{side}_idx"], m[f"{side}_strip"]
        assert len(idx) == len(st) and len(idx) == len(set(idx.tolist()))
        cs = np.bincount(st, minlength=N_STRIPS)
        assert cs.min() * 2 >= cs.max(), f"{side}: полосы листа неравны: {cs.tolist()}"
        assert cs.sum() > 3800, f"{side}: покрыто всего {cs.sum()} проекционных"
    assert not (set(m["left_idx"].tolist()) & set(m["right_idx"].tolist())), \
        "левые и правые проекционные пересеклись"
    print(f"карта: ось {m['axis']}, полос {N_STRIPS}, "
          f"проекционных {len(m['left_idx'])}/{len(m['right_idx'])}")
```

- [ ] **Шаг 2: убедиться, что проверка падает**

```bash
.venv/bin/python tools/visual_field_map.py
```

Ожидается: `NameError: name 'load_or_build' is not defined`.

- [ ] **Шаг 3: реализовать измерение ориентации на теле**

Вставить перед `self_check()`:

```python
def measure_ommatidia_axis():
    """Какая ось картинки омматидиев отвечает за азимут и где у неё перёд.

    Постулировать нельзя: нумерация омматидиев в flygym — это пиксели
    отрендеренного глаза, и связь их осей с полем зрения нигде не объявлена.
    Меряем: ставим столб прямо по курсу и сбоку, смотрим, какие омматидии
    темнеют, и берём ту ось, вдоль которой центры тяжести затемнения разошлись
    сильнее. Знак — по тому, куда сместился «перёд».
    """
    from flyreplay import build_scene

    def darkening(xy):
        sim = build_scene(xy)
        sim.warmup(0.05)
        name = next(iter(sim.world.fly_lookup))
        v = sim.get_ommatidia_readouts(name).sum(axis=2).mean(axis=0)   # (721,)
        sim.close()
        return v

    empty = darkening((500.0, 500.0))       # FAR_AWAY: столб унесён
    front = empty - darkening((6.0, 0.0))
    sidew = empty - darkening((0.0, 6.0))

    m = np.load(OMMATIDIA_MAP)
    ids = np.unique(m); ids = ids[ids > 0]
    flat = m.ravel().astype(np.int64)
    yy, xx = np.divmod(np.arange(flat.size), m.shape[1])
    cnt = np.bincount(flat, minlength=722).astype(float)
    cy = np.bincount(flat, weights=yy, minlength=722)[ids] / cnt[ids]
    cx = np.bincount(flat, weights=xx, minlength=722)[ids] / cnt[ids]
    cent = np.stack([cy, cx], axis=1)

    def com(d):
        w = np.clip(d, 0, None)
        return (cent * w[:, None]).sum(axis=0) / max(w.sum(), 1e-9), w.sum()

    cf, wf = com(front)
    cs, ws = com(sidew)
    assert wf > 0 and ws > 0, "ни одна сцена не затемнила глаз — столб не виден"
    shift = cf - cs
    axis = int(np.argmax(np.abs(shift) / cent.std(axis=0)))
    flip = bool(shift[axis] < 0)      # перёд должен получить БОЛЬШИЙ номер полосы
    return {"axis": axis, "flip": flip,
            "contrast": float(abs(shift[axis]) / cent[:, axis].std()),
            "weight_front": float(wf), "weight_side": float(ws)}


def ommatidia_strips(axis, flip, n=N_STRIPS):
    """Номер полосы для каждого из 721 омматидия.

    В карте flygym id=0 это фон, реальные омматидии пронумерованы 1..721;
    показание с индексом i соответствует id i+1 — проверяется ассертом.
    """
    m = np.load(OMMATIDIA_MAP)
    ids = np.unique(m); ids = ids[ids > 0]
    assert ids.min() == 1 and ids.max() == 721 and len(ids) == 721, \
        f"неожиданная нумерация омматидиев: {ids.min()}..{ids.max()}, {len(ids)} шт"
    flat = m.ravel().astype(np.int64)
    yy, xx = np.divmod(np.arange(flat.size), m.shape[1])
    cnt = np.bincount(flat, minlength=722).astype(float)
    c = [np.bincount(flat, weights=yy, minlength=722)[ids] / cnt[ids],
         np.bincount(flat, weights=xx, minlength=722)[ids] / cnt[ids]][axis]
    return strips(-c if flip else c, n)


def load_or_build():
    """Карта целиком. Кэш — производное от аннотаций, в git не идёт."""
    if CACHE.exists():
        z = np.load(CACHE)
        return {k: (str(z[k]) if k == "axis" else z[k]) for k in z.files}
    flyid2i, ann, con = load_tables()      # один раз: parquet на 15 млн строк
    o = measure_ommatidia_axis()
    res = {"axis": MAP_AXIS, "ommatidia": ommatidia_strips(o["axis"], o["flip"]),
           "om_axis": np.array(o["axis"]), "om_flip": np.array(o["flip"]),
           "om_contrast": np.array(o["contrast"])}
    col = 0 if MAP_AXIS == "elevation" else 1
    for side in ("left", "right"):
        vp, _, _, _ = field_positions(ann, con, side)
        keys = np.array(sorted(vp))
        vals = np.array([vp[k][col] for k in keys])
        res[f"{side}_idx"] = np.array([flyid2i[int(k)] for k in keys], dtype=np.int64)
        res[f"{side}_strip"] = strips(vals)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, **res)
    return res
```

`OMMATIDIA_MAP` уже объявлена в задаче 3 рядом с `CACHE` — второй раз добавлять не нужно.

В коде выше `load_tables()` вызывается в `load_or_build` дважды, и каждый вызов читает parquet на 15 млн строк. Исправить сразу: получить `flyid2i, ann, con = load_tables()` один раз в начале функции и переиспользовать.

- [ ] **Шаг 4: запустить проверку**

```bash
.venv/bin/python tools/visual_field_map.py
```

Ожидается строка вида `карта: ось azimuth, полос 4, проекционных 3973/4007` и `самопроверка осей листа: ОК`. Время — около полутора минут при первом построении, секунды из кэша.

- [ ] **Шаг 5: проверить, что измерение ориентации не выродилось**

Дописать в `self_check()`:

```python
    assert float(m["om_contrast"]) > 0.3, (
        f"столб впереди и сбоку затемняют почти одно и то же место сетчатки "
        f"(контраст {float(m['om_contrast']):.2f}) — ориентация не определена")
    print(f"  ориентация омматидиев: ось {int(m['om_axis'])}, "
          f"переворот {bool(m['om_flip'])}, контраст {float(m['om_contrast']):.2f}")
```

Запустить, удалив кэш, чтобы измерение действительно выполнилось:

```bash
rm -f output/visual_field_map.npz && .venv/bin/python tools/visual_field_map.py
```

- [ ] **Шаг 6: коммит**

```bash
git add tools/visual_field_map.py
git commit -m "Полосы омматидиев и ориентация, измеренная на теле"
```

---

### Задача 5: пространственная подача в замкнутом контуре

**Файлы:**
- Изменить: `closed_loop_vision.py` — строки 210–213 (`LOG_COLUMNS`), 292–307 (`step_brain`), 316–321 (калибровка яркости), 344–376 (основной цикл), 425–453 (`main`)

**Интерфейсы:**
- Использует: `tools.visual_field_map.load_or_build()`
- Даёт: `run_trial(..., spatial=False)`; при `spatial=True` в CSV появляются колонки `dark_l0..dark_l3`, `dark_r0..dark_r3`, `rate_l0..rate_l3`, `rate_r0..rate_r3`; колонки `dark_left`, `dark_right`, `lc_rate_left_hz`, `lc_rate_right_hz` остаются и равны среднему по полосам, чтобы `tools/replicate_vision.py` и `tools/analyze_replication.py` продолжали работать без правок

Ключевое решение: **скалярный режим — это частный случай пространственного с одной полосой.** Тогда в цикле нет ветвления, и контроль «скалярный вход на тех же сценах» гоняется тем же кодом, а не параллельной веткой, которая могла бы разойтись.

- [ ] **Шаг 1: написать падающую проверку**

Создать `test_spatial.py` в корне:

```python
"""Проверка карты поля зрения и пространственной подачи.

Ломается, если карта построена неверно или если подача перестала различать
участки поля зрения. Мозг не запускается: сцена, глаза, арифметика подачи —
секунды, а не минуты.

    .venv/bin/python test_spatial.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flyreplay import build_scene  # noqa: E402
from tools.visual_field_map import N_STRIPS, load_or_build  # noqa: E402
from closed_loop_vision import strip_intensity  # noqa: E402


def eye_profile(pillar_xy, om):
    sim = build_scene(pillar_xy)
    sim.warmup(0.05)
    name = next(iter(sim.world.fly_lookup))
    raw = sim.get_ommatidia_readouts(name).sum(axis=2)      # (2, 721)
    prof = strip_intensity(raw, om, N_STRIPS)
    sim.close()
    return raw, prof


def main():
    m = load_or_build()
    om = m["ommatidia"]

    raw_f, prof_f = eye_profile((6.0, 0.0), om)             # столб по курсу
    raw_s, prof_s = eye_profile((0.0, 6.0), om)             # столб слева
    assert prof_f.shape == (2, N_STRIPS), f"профиль {prof_f.shape}"

    # 1. скалярный режим — частный случай: одна полоса даёт ровно старую величину
    one = strip_intensity(raw_f, np.zeros(721, int), 1)
    assert np.allclose(one[:, 0], raw_f.mean(axis=1)), \
        "одна полоса перестала совпадать со средним по глазу"

    # 2. разные сцены дают разные ПРОФИЛИ, а не только разную сумму
    df = prof_f - prof_f.mean(axis=1, keepdims=True)
    ds = prof_s - prof_s.mean(axis=1, keepdims=True)
    spread = float(np.abs(df - ds).max())
    assert spread > 1e-3, (
        f"профили сцен различаются только уровнем (максимум разности форм "
        f"{spread:.2e}) — пространственная подача ничего не добавляет")
    print(f"  формы профилей разошлись на {spread:.4f}")

    # 3. контроль на самопроверку: перемешанная карта обязана дать другое
    shuf = np.random.default_rng(0).permutation(om)
    alt = strip_intensity(raw_f, shuf, N_STRIPS)
    assert not np.allclose(alt, prof_f), \
        "перемешанная карта дала тот же профиль: проверка ничего не проверяет"

    # 4. карта покрывает обе стороны и полосы населены
    for side in ("left", "right"):
        c = np.bincount(m[f"{side}_strip"], minlength=N_STRIPS)
        assert c.min() * 2 >= c.max(), f"{side}: полосы неравны: {c.tolist()}"
        assert c.sum() > 3800, f"{side}: покрыто {c.sum()} проекционных"
        print(f"  {side}: полосы {c.tolist()}")

    print("OK: карта и подача целы")


if __name__ == "__main__":
    main()
```

- [ ] **Шаг 2: убедиться, что проверка падает**

```bash
.venv/bin/python test_spatial.py
```

Ожидается: `ImportError: cannot import name 'strip_intensity' from 'closed_loop_vision'`.

- [ ] **Шаг 3: добавить `strip_intensity` в `closed_loop_vision.py`**

Вставить после `LOG_COLUMNS` (строка 213):

```python
def strip_intensity(raw, om_strip, n_strips):
    """Средняя яркость по полосам поля зрения: (2, 721) -> (2, n_strips).

    При n_strips=1 это ровно прежнее .mean(axis=1) — скалярный режим остаётся
    частным случаем пространственного, а не отдельной веткой кода, которая
    могла бы с ним разойтись.
    """
    return np.stack([[raw[k][om_strip == s].mean() for s in range(n_strips)]
                     for k in (0, 1)])
```

- [ ] **Шаг 4: убедиться, что проверка проходит до пункта 2**

```bash
.venv/bin/python test_spatial.py
```

Ожидается прохождение пунктов 1–4 и `OK: карта и подача целы`. Время — около 40 секунд.

- [ ] **Шаг 5: коммит проверки**

```bash
git add test_spatial.py closed_loop_vision.py
git commit -m "Проверка карты и пространственной подачи"
```

- [ ] **Шаг 6: развести подачу по полосам в `run_trial`**

В сигнатуру `run_trial` (строка 216) добавить параметр `spatial=False`.

Сразу после распаковки `lc_l, lc_r` (строка 227) вставить:

```python
    # Скалярный режим — одна полоса на глаз. Один и тот же код обслуживает оба
    # режима, поэтому контроль «скалярный вход на тех же сценах» не может
    # разойтись с основным по реализации.
    if spatial:
        m = load_or_build()
        n_strips = int(m["ommatidia"].max()) + 1
        om_strip = m["ommatidia"]
        grp_l = [m["left_idx"][m["left_strip"] == s] for s in range(n_strips)]
        grp_r = [m["right_idx"][m["right_strip"] == s] for s in range(n_strips)]
    else:
        n_strips = 1
        om_strip = np.zeros(721, dtype=int)
        grp_l, grp_r = [np.asarray(lc_l)], [np.asarray(lc_r)]
```

и импорт наверху файла, рядом с `from flyreplay import ...`:

```python
from tools.visual_field_map import load_or_build  # noqa: E402
```

Заменить тело `step_brain` (строки 294–296):

```python
        rates.zero_()
        rates[:, lc_l] = rate_l
        rates[:, lc_r] = rate_r
```

на:

```python
        rates.zero_()
        for s in range(n_strips):
            rates[:, grp_l[s]] = float(rate_l[s])
            rates[:, grp_r[s]] = float(rate_r[s])
```

и сигнатуру `def step_brain(rate_l, rate_r)` оставить прежней — меняется только тип аргументов на массивы длины `n_strips`.

- [ ] **Шаг 7: развести калибровку и основной цикл**

Заменить строку 320:

```python
            base_int.append(sim.get_ommatidia_readouts(fly.name).sum(axis=2).mean(axis=1))
```

на:

```python
            base_int.append(strip_intensity(
                sim.get_ommatidia_readouts(fly.name).sum(axis=2), om_strip, n_strips))
```

`baseline` станет формы `(2, n_strips)` — строка 321 не меняется.

Заменить строку 327:

```python
            a, b = step_brain(lc_base, lc_base)
```

на:

```python
            flat = np.full(n_strips, lc_base)
            a, b = step_brain(flat, flat)
```

Заменить строки 345 и 353–355:

```python
            inten_raw = sim.get_ommatidia_readouts(fly.name).sum(axis=2).mean(axis=1)
```

на:

```python
            inten_raw = strip_intensity(
                sim.get_ommatidia_readouts(fly.name).sum(axis=2), om_strip, n_strips)
```

и

```python
            rate_l = lc_base + lc_span * float(dark[0])
            rate_r = lc_base + lc_span * float(dark[1])
```

на:

```python
            rate_l = lc_base + lc_span * dark[0]
            rate_r = lc_base + lc_span * dark[1]
```

Строки 350–351 (`rel`, `dark`) не меняются: numpy посчитает их поэлементно на форме `(2, n_strips)`.

Заменить формирование строки лога (строки 372–376):

```python
            rows.append([cycle, cycle * SYNC_MS / 1000.0,
                         float(eye_filt[0]), float(eye_filt[1]),
                         float(dark[0]), float(dark[1]), rate_l, rate_r,
                         hz_l, hz_r, cmd_l, cmd_r,
                         float(pos[0]), float(pos[1]), heading, dist])
```

на:

```python
            # Сводные колонки (среднее по полосам) остаются на прежних местах:
            # tools/replicate_vision.py и tools/analyze_replication.py читают
            # именно их, и переписывать разбор ради нового режима незачем.
            row = [cycle, cycle * SYNC_MS / 1000.0,
                   float(eye_filt[0].mean()), float(eye_filt[1].mean()),
                   float(dark[0].mean()), float(dark[1].mean()),
                   float(rate_l.mean()), float(rate_r.mean()),
                   hz_l, hz_r, cmd_l, cmd_r,
                   float(pos[0]), float(pos[1]), heading, dist]
            if n_strips > 1:
                row += [float(x) for x in dark[0]] + [float(x) for x in dark[1]]
                row += [float(x) for x in rate_l] + [float(x) for x in rate_r]
            rows.append(row)
```

и построение DataFrame (строка 379):

```python
    df = pd.DataFrame(rows, columns=LOG_COLUMNS)
```

на:

```python
    cols = list(LOG_COLUMNS)
    if n_strips > 1:
        cols += [f"dark_l{s}" for s in range(n_strips)]
        cols += [f"dark_r{s}" for s in range(n_strips)]
        cols += [f"rate_l{s}" for s in range(n_strips)]
        cols += [f"rate_r{s}" for s in range(n_strips)]
    df = pd.DataFrame(rows, columns=cols)
```

Добавить в `summary` (после строки 406) строку:

```python
        "spatial": bool(spatial), "n_strips": n_strips,
```

- [ ] **Шаг 8: добавить флаг в CLI**

В `main()` после строки 450 добавить:

```python
    ap.add_argument("--spatial", action="store_true",
                    help="подавать яркость по полосам поля зрения, а не одним "
                         "числом на глаз")
```

и в вызов `run_trial` (строка 467) добавить `spatial=args.spatial,`.

- [ ] **Шаг 9: проверить оба режима коротким прогоном**

```bash
.venv/bin/python closed_loop_vision.py --cycles 6 --tag smoke_scalar
```

```bash
.venv/bin/python closed_loop_vision.py --cycles 6 --spatial --tag smoke_spatial
```

Ожидается: оба завершаются без ошибок; в `output/closed_loop_vision_smoke_spatial.csv` есть колонки `dark_l0..dark_l3`, в скалярном их нет. Проверить, что колонка `dark_left` присутствует в обоих. Время — по 3–4 минуты каждый (калибровка мозга 3000 мс плюс 6 циклов).

- [ ] **Шаг 10: убедиться, что полосы действительно различаются в прогоне**

```bash
.venv/bin/python -c "
import pandas as pd
d = pd.read_csv('output/closed_loop_vision_smoke_spatial.csv')
c = [f'dark_l{s}' for s in range(4)]
print(d[c].describe().loc[['mean','std','max']].round(4))
print('разброс между полосами по циклам:', float(d[c].std(axis=1).mean()).__round__(5))
"
```

Ожидается ненулевой разброс между полосами. Ноль означает, что карта раздала всем одно и то же и подача не работает.

- [ ] **Шаг 11: коммит**

```bash
git add closed_loop_vision.py
git commit -m "Пространственная подача: яркость полосы своей группе проекционных"
```

---

### Задача 6: сцены для этапа 2

**Файлы:**
- Изменить: `flyreplay.py` — класс `PillarWorld` (строки 33–44)
- Изменить: `test_spatial.py` — добавить проверку новых сцен

**Интерфейсы:**
- Даёт: `PillarWorld(x, y, z=None, h=PILLAR_H)`; при `z=None` столб стоит на грунте, как раньше

Обратная совместимость обязательна: `flyreplay.build_scene`, `closed_loop_vision.run_trial` и `test_replay.py` вызывают `PillarWorld(x, y)` двумя аргументами.

- [ ] **Шаг 1: написать падающую проверку**

Дописать в `test_spatial.py` перед `print("OK: ...")`:

```python
    # 5. сцены этапа 2: две позиции объекта вдоль оси карты
    from flyreplay import PillarWorld
    import inspect
    p = inspect.signature(PillarWorld.__init__).parameters
    assert "z" in p and "h" in p, "PillarWorld не принимает высоту"
    assert p["z"].default is None and p["h"].default is not None, \
        "старый вызов PillarWorld(x, y) должен работать без изменений"
    raw_up, prof_up = eye_profile_z((4.0, 0.0), 4.0, 1.5, om)
    raw_dn, prof_dn = eye_profile_z((4.0, 0.0), 0.75, 1.5, om)
    du = prof_up - prof_up.mean(axis=1, keepdims=True)
    dd = prof_dn - prof_dn.mean(axis=1, keepdims=True)
    assert float(np.abs(du - dd).max()) > 1e-3, \
        "объект вверху и внизу дают одинаковый профиль полос"
    print(f"  верх против низа: формы разошлись на {float(np.abs(du - dd).max()):.4f}")
```

и функцию рядом с `eye_profile`:

```python
def eye_profile_z(xy, z, h, om):
    """То же, что eye_profile, но объект поднят: сцены запасного варианта."""
    from flygym.simulation import Simulation
    from flygym.utils.math import Rotation3D
    from flygym_demo.complex_terrain import make_locomotion_fly
    from flyreplay import PillarWorld

    fly = make_locomotion_fly()
    fly.add_vision()
    world = PillarWorld(xy[0], xy[1], z=z, h=h)
    world.add_fly(fly, spawn_position=[0.0, 0.0, 0.5],
                  spawn_rotation=Rotation3D("quat", [1, 0, 0, 0]),
                  add_ground_contact_sensors=True)
    sim = Simulation(world)
    sim.reset()
    sim.warmup(0.05)
    raw = sim.get_ommatidia_readouts(fly.name).sum(axis=2)
    prof = strip_intensity(raw, om, N_STRIPS)
    sim.close()
    return raw, prof
```

- [ ] **Шаг 2: убедиться, что проверка падает**

```bash
.venv/bin/python test_spatial.py
```

Ожидается: `AssertionError: PillarWorld не принимает высоту`.

- [ ] **Шаг 3: расширить `PillarWorld`**

Заменить в `flyreplay.py` строки 33–44:

```python
class PillarWorld(FlatGroundWorld):
    """Плоский грунт и один тёмный столб.

    z и h нужны сценам этапа 2 запасного варианта: объект вверху против объекта
    внизу проверяет ось элевации. По умолчанию столб стоит на грунте во всю
    высоту — ровно как раньше, поэтому вызов PillarWorld(x, y) не меняется.
    """

    def __init__(self, x: float, y: float,
                 z: float | None = None, h: float = PILLAR_H) -> None:
        super().__init__(name="pillar_world", half_size=300)
        self.mjcf_root.worldbody.add_geom(
            type=GEOM_TYPES["cylinder"], name="pillar",
            size=[PILLAR_R, h / 2, 0.0],
            pos=[x, y, h / 2 if z is None else z],
            rgba=[0.05, 0.05, 0.05, 1.0],
            contype=0, conaffinity=0,
        )
```

- [ ] **Шаг 4: запустить обе проверки**

```bash
.venv/bin/python test_spatial.py
```

```bash
.venv/bin/python test_replay.py
```

Обе должны пройти. Вторая — доказательство, что старый вызов `PillarWorld(x, y)` не сломан.

- [ ] **Шаг 5: коммит**

```bash
git add flyreplay.py test_spatial.py
git commit -m "Сцены с поднятым объектом для оси элевации"
```

---

### Задача 7: этап 2 — приёмка на поведении

**Файлы:**
- Создать: `tools/replicate_spatial.py`
- Результат: `output/spatial_replication.csv`, `output/spatial_rep_*.csv`

**Интерфейсы:**
- Использует: `closed_loop_vision.load_brain_assets`, `closed_loop_vision.run_trial(..., spatial=...)`, `tools.replicate_vision.welch`, `tools.visual_field_map.load_or_build`
- Даёт: `output/spatial_replication.csv` с колонками сводки `run_trial` плюс `condition`, `spatial`

Сцены выбираются по оси, зафиксированной в задаче 3:

| Ось карты | Сцена A | Сцена Б | Контроль |
|---|---|---|---|
| `azimuth` | столб по курсу, `pillar_x=10.0, pillar_y=0.0` | столб сбоку, `pillar_x=0.0, pillar_y=10.0` | `no_pillar=True` |
| `elevation` | объект вверху, `pillar_x=4.0, pillar_y=0.0, z=4.0, h=1.5` | объект внизу, те же x и y, `z=0.75, h=1.5` | `no_pillar=True` |

Обе сцены основной пары равноудалены от мухи — меняется только положение вдоль оси карты, не расстояние. Иначе разделение объяснялось бы разной яркостью, а не разным участком поля зрения.

**Критерий, зафиксирован до прогона. Оба условия обязательны:**

1. Поворот курса в сцене A против сцены Б различается по Уэлчу **|t| > 3** при пространственном входе, и того же разделения НЕТ при скалярном входе на тех же сценах.
2. Сцены зажигают **разные множества** проекционных нейронов: множество определяется как объединение групп полос, у которых `dark` превысил медиану своего прогона; сравниваются множества индексов, а не средние. Пересечение по Жаккару должно быть заметно меньше 1.

Опорные числа подтверждённого избегания: без столба +1.0 ± 9.7°, столб слева −53.6 ± 18.3°, столб справа +56.6 ± 15.5°, слева против справа t = −13.0.

- [ ] **Шаг 1: написать серию**

Создать `tools/replicate_spatial.py`:

```python
"""Этап 2: различает ли муха участки поля зрения на поведении.

Три сцены на два режима входа по 8 seed = 48 прогонов. Скалярный вход — это не
запасной путь, а обязательный контроль ценности карты: если он двигает муху не
хуже пространственного, карта ничего не добавила, и это записывается прямым
текстом.

Батча здесь намеренно нет. Батчатся только разомкнутые прогоны, где тела нет и
ошибка видна сразу; в петле с телом тонкая ошибка калибровки спрячется, а
проект на таких уже спотыкался дважды.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flypaths import out  # noqa: E402
from closed_loop_vision import load_brain_assets, run_trial  # noqa: E402
from tools.replicate_vision import welch  # noqa: E402
from tools.visual_field_map import load_or_build  # noqa: E402

# Ключ (латиницей) идёт в имена файлов, подпись — в вывод. Обе сцены основной
# пары равноудалены от мухи: меняется положение вдоль оси карты, не расстояние.
# Иначе разделение объяснялось бы разной яркостью, а не разным участком поля.
SCENES = {
    "azimuth": [
        ("A", "A: объект по курсу",
         dict(no_pillar=False, pillar_x=10.0, pillar_y=0.0)),
        ("B", "Б: объект сбоку",
         dict(no_pillar=False, pillar_x=0.0, pillar_y=10.0)),
        ("ctrl", "контроль: пусто", dict(no_pillar=True)),
    ],
    "elevation": [
        ("A", "A: объект вверху",
         dict(no_pillar=False, pillar_x=4.0, pillar_y=0.0,
              pillar_z=4.0, pillar_h=1.5)),
        ("B", "Б: объект внизу",
         dict(no_pillar=False, pillar_x=4.0, pillar_y=0.0,
              pillar_z=0.75, pillar_h=1.5)),
        ("ctrl", "контроль: пусто", dict(no_pillar=True)),
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--cycles", type=int, default=100)
    ap.add_argument("--tag", type=str, default="spatial")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    axis = str(load_or_build()["axis"])
    scenes = SCENES[axis]
    print("=" * 78)
    print(f" ЭТАП 2: ось карты «{axis}», сцен {len(scenes)}, "
          f"режимов входа 2, seed {args.seeds}")
    print("=" * 78)
    print(f"всего прогонов: {2 * len(scenes) * args.seeds}")

    assets = load_brain_assets(device)
    rows, done = [], 0
    total = 2 * len(scenes) * args.seeds
    t_all = time.perf_counter()

    for spatial in (True, False):
        mode = "spatial" if spatial else "scalar"
        for key, label, kw in scenes:
            for seed in range(args.seeds):
                t0 = time.perf_counter()
                df, s = run_trial(assets, device, cycles=args.cycles, seed=seed,
                                  spatial=spatial, verbose=False, **kw)
                done += 1
                s["condition"] = label
                s["scene"] = key
                s["input"] = "пространственный" if spatial else "скалярный"
                rows.append(s)
                print(f"  [{done:2d}/{total}] {mode:8s} {label:<20s} seed={seed} "
                      f"поворот {s['turn_deg']:+7.1f}°  путь {s['path_mm']:5.2f} мм "
                      f"[{time.perf_counter() - t0:.0f} с]")
                pd.DataFrame(rows).to_csv(out(f"{args.tag}_replication.csv"), index=False)
                df.round(4).to_csv(
                    out(f"{args.tag}_rep_{mode}_{key}_s{seed}.csv"), index=False)

    print(f"\nвсего {time.perf_counter() - t_all:.0f} с")
    res = pd.DataFrame(rows)

    print("\n" + "=" * 78)
    print(" СВОДКА (среднее ± ст.откл. по seed)")
    print("=" * 78)
    for inp in ("пространственный", "скалярный"):
        print(f"\n  вход: {inp}")
        for _, label, _ in scenes:
            g = res[(res["condition"] == label) & (res["input"] == inp)]
            print(f"    {label:<20s} поворот {g['turn_deg'].mean():>+7.1f} ± "
                  f"{g['turn_deg'].std():<6.1f}  путь {g['path_mm'].mean():>5.2f}")

    print("\n" + "=" * 78)
    print(" КРИТЕРИЙ 1: A против Б, и есть ли разделение у скалярного контроля")
    print("=" * 78)
    verdict = {}
    for inp in ("пространственный", "скалярный"):
        a = res[(res["scene"] == "A") & (res["input"] == inp)]["turn_deg"]
        b = res[(res["scene"] == "B") & (res["input"] == inp)]["turn_deg"]
        t, dof = welch(a, b)
        verdict[inp] = abs(t)
        print(f"  {inp:<18s} {a.mean():+7.1f} против {b.mean():+7.1f}  "
              f"t={t:+6.2f}  dof={dof:5.1f}  -> "
              f"{'разделяет' if abs(t) > 3 else 'не разделяет'}")
    ok1 = verdict["пространственный"] > 3 and verdict["скалярный"] <= 3
    print(f"\n  критерий 1 -> {'ПРОЙДЕН' if ok1 else 'НЕ ПРОЙДЕН'}")
    if verdict["скалярный"] > 3:
        print("  Скалярный вход разделяет сцены не хуже: карта ничего не добавила.")

    print(f"\nсохранено: {out(f'{args.tag}_replication.csv')}")


if __name__ == "__main__":
    main()
```

Сцены оси `elevation` передают `pillar_z` и `pillar_h`, а `run_trial` их пока не принимает. Правка в `closed_loop_vision.py`: в сигнатуру `run_trial` добавить `pillar_z=None, pillar_h=None`, а строку 244 `world = PillarWorld(px, py)` заменить на:

```python
    world = (PillarWorld(px, py) if pillar_z is None
             else PillarWorld(px, py, z=pillar_z, h=pillar_h))
```

Делать это надо независимо от того, какая ось выбрана: при оси `azimuth` параметры просто остаются `None`, а без них серия упадёт на `TypeError` при первом же прогоне запасного варианта.

- [ ] **Шаг 2: короткий пробный прогон**

```bash
.venv/bin/python tools/replicate_spatial.py --seeds 1 --cycles 8 --tag spatial_smoke
```

Ожидается 6 прогонов без ошибок и файл `output/spatial_smoke_replication.csv` с колонками `condition`, `input`, `turn_deg`. Время — около 25 минут.

- [ ] **Шаг 3: коммит серии**

```bash
git add tools/replicate_spatial.py closed_loop_vision.py
git commit -m "Серия этапа 2: три сцены на два режима входа"
```

- [ ] **Шаг 4: добавить критерий 2 — сравнение множеств**

Дописать в `main()` перед финальным `print("сохранено")`:

```python
    print("\n" + "=" * 78)
    print(" КРИТЕРИЙ 2: разные ли МНОЖЕСТВА проекционных нейронов")
    print("=" * 78)
    m = load_or_build()
    n_strips = int(m["ommatidia"].max()) + 1

    def lit_set(scene_key, seed):
        """Какие проекционные нейроны получили сверхмедианный вход за прогон."""
        d = pd.read_csv(out(f"{args.tag}_rep_spatial_{scene_key}_s{seed}.csv"))
        idx = set()
        for side, key in (("left", "l"), ("right", "r")):
            cols = [f"dark_{key}{s}" for s in range(n_strips)]
            mean = d[cols].mean(axis=0).to_numpy()
            for s in np.where(mean > np.median(mean))[0]:
                idx |= set(m[f"{side}_idx"][m[f"{side}_strip"] == s].tolist())
        return idx

    jac = []
    for seed in range(args.seeds):
        sa, sb = lit_set("A", seed), lit_set("B", seed)
        inter, union = len(sa & sb), len(sa | sb)
        jac.append(inter / max(union, 1))
    jac = np.array(jac)
    print(f"  Жаккар A против Б по {args.seeds} seed: "
          f"{jac.mean():.3f} ± {jac.std():.3f}, максимум {jac.max():.3f}")
    ok2 = jac.mean() < 0.9
    print(f"  критерий 2 -> {'ПРОЙДЕН' if ok2 else 'НЕ ПРОЙДЕН'}")

    # Обратная причинность. Связь «темнота -> поворот» значима ТОЛЬКО в
    # контроле без объекта: муха кренится при повороте, крен меняет яркость.
    # Признак артефакта — периодика по сдвигам вместо одного максимума.
    print("\n" + "=" * 78)
    print(" КОНТРОЛЬ ОБРАТНОЙ ПРИЧИННОСТИ: профиль по сдвигам")
    print("=" * 78)
    lags = (0, 1, 2, 3, 5, 8)
    print(f"  {'сцена':<20s}" + "".join(f"{l:>8d}" for l in lags))
    for key, label, _ in scenes:
        prof = []
        for lag in lags:
            rr = []
            for seed in range(args.seeds):
                d = pd.read_csv(out(f"{args.tag}_rep_spatial_{key}_s{seed}.csv"))
                c = (d["cmd_left"] - d["cmd_right"]).to_numpy()
                h = np.diff(d["heading_deg"].to_numpy(), prepend=d["heading_deg"].iloc[0])
                a, b = (c, h) if lag == 0 else (c[:-lag], h[lag:])
                if a.std() > 1e-9 and b.std() > 1e-9:
                    rr.append(float(np.corrcoef(a, b)[0, 1]))
            prof.append(np.mean(rr) if rr else float("nan"))
        print(f"  {label:<20s}" + "".join(f"{v:>+8.3f}" for v in prof))
        if np.sign(prof[0]) != np.sign(prof[1]) or np.sign(prof[1]) != np.sign(prof[2]):
            print(f"    знак прыгает по сдвигам -> артефакт качки, не причинность")

    print(f"\n  ИТОГ ЭТАПА 2: "
          f"{'ПРОЙДЕН' if (ok1 and ok2) else 'НЕ ПРОЙДЕН'}")
```

Профиль по сдвигам не входит в критерий приёмки — он ловит истолкование. Подтверждённое избегание давало чёткий максимум на сдвиге 2 цикла (30 мс) с одинаковым знаком во всех сценах; чередование знака означает, что меряется крен при повороте, а не влияние зрения.

- [ ] **Шаг 5: прогнать полную серию**

```bash
.venv/bin/python tools/replicate_spatial.py --seeds 8 --cycles 100 --tag spatial
```

Время — 5–6 часов. Запускать фоном.

- [ ] **Шаг 6: записать результат в журнал**

Дописать в `PROJECT_LOG.md` новый раздел с датой. Обязательно привести числами:

- рабочая частота стимуляции и прирост отклика на верхней ступени развёртки;
- этап 0: внутригрупповая и межгрупповая корреляции, t, dof;
- split-half контроль: `r_elev` и `r_azim` на обеих сторонах, выбранная ось;
- ориентация омматидиев: ось, переворот, контраст;
- этап 2: поворот по каждой сцене для обоих режимов входа, t для A против Б в обоих режимах, Жаккар;
- вердикт по обоим критериям, включая случай, когда скалярный контроль разделяет сцены не хуже пространственного.

Отрицательный результат записывается так же подробно, как положительный. Опровергнутое из журнала не удаляется.

Обновить таблицу «План работ и критерии приёмки» в конце `PROJECT_LOG.md`: пункт 13 («Внутриглазная ретинотопия», закрыт отрицательно) дополнить ссылкой на новый раздел, добавить строку про пространственный вход с полученным статусом.

- [ ] **Шаг 7: коммит и push**

```bash
git add PROJECT_LOG.md output/spatial_replication.csv tools/replicate_spatial.py
git commit -m "Этап 2: результат пространственного зрения с числами"
```

```bash
git fetch && git rebase origin/main
```

```bash
git push -u origin vision-spatial-b
```

Не `force`: в `main` пушут два человека.

---

## Что делать, если этап падает

| Где упало | Что это значит | Что делать |
|---|---|---|
| Развёртка по частоте: отклик не растёт | стимуляция не доходит до нисходящих | проверить, что индексы проекционных взяты из `assets["lc_l"]`, а не из аннотаций напрямую; это разные нумерации |
| Развёртка: полка между 100 и 200 Гц | сеть в насыщении | взять рабочую точку ниже, повторить развёртку с (10, 25, 50, 100) |
| Этап 0: \|t\| ≤ 3 | выход мозга не различает подмножества входа | остановиться, записать отрицательный результат, задачи 3–7 не выполнять |
| Split-half: `r_elev` ≤ 0.9 | сломан расчёт, а не ось | проверить `VOXEL_NM`, проверить что `sheet_axes` берёт DRA своей стороны, не обеих; в запасной вариант не уходить |
| Split-half: `r_azim` ≤ 0.5 | азимут не восстанавливается | перейти на ось `elevation`, сцены этапа 2 меняются на «вверху/внизу» — это заранее объявленный запасной вариант, а не провал |
| `om_contrast` ≤ 0.3 | столб впереди и сбоку затемняют одно место | увеличить контраст сцены: приблизить столб (4.0 вместо 6.0) и проверить `tools/vision_smoke_check.py`, видит ли тело объект вообще |
| Этап 2: скалярный контроль разделяет сцены не хуже | карта ничего не добавила | записать это прямым текстом в журнал; результат отрицательный и подлежит публикации наравне с положительным |
