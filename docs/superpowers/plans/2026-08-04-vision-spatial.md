# Пространственный зрительный вход: план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Довести до мозга мухи пространственный зрительный вход через зонную карту фоторецепторов и проверить, доходит ли пространственная информация до нисходящих нейронов и до поведения.

**Architecture:** Лист фоторецепторов R1-6 каждой стороны режется на 4 зоны по двум главным осям; омматидии flygym режутся на те же 4 зоны; стимуляция идёт зона в зону. Этап 1 без тела меряет различимость зон по отклику 1291 нисходящего нейрона против потолка пуассоновского шума. Этап 2 строится только если этап 1 пройден: неподвижная муха при базовой команде 0 и приближающийся объект.

**Tech Stack:** Python 3.14 из `.venv`, torch (LIF-модель коннектома), pandas, numpy, flygym/MuJoCo. Новых зависимостей не добавляется.

## Global Constraints

- Запуск только через `.venv/bin/python`. Системный python не годится: flygym требует CPython 3.12–3.14.
- `MUJOCO_GL` выставляет `flypaths.py` по платформе. Не трогать, не переопределять, не импортировать `flygym`/`mujoco` раньше `flypaths`.
- Всё производное от аннотаций FlyWire в git не идёт: репозиторий `flyconnectome/flywire_annotations` опубликован без лицензии. Кэш карты зон — в `output/`, с добавлением в `.gitignore`.
- Порог значимости: по Уэлчу |t| > 3. Для этапа 2 дополнительно — эффект виден на каждом seed отдельно.
- Утверждения вида «объект слева» из зонной карты не делаются: ориентация листа относительно поля зрения неизвестна.
- Отбор нейронов производится измерением отклика, а не по весам связей. В проекте дважды выяснялось, что анатомический вес не равен функциональному влиянию: Sugar GRNs до P9 и DNg108 от восходящих — оба с формальным путём и нулевым откликом.
- Отрицательный результат на любом этапе записывается в `PROJECT_LOG.md` и работа останавливается. Это не провал плана, а его штатный исход.
- Ветка `vision-spatial`, базовый коммит `f37bbda`.

---

### Task 1: Зонная карта фоторецепторов и омматидиев

**Files:**
- Create: `tools/retina_zone_map.py`
- Modify: `.gitignore` (добавить `output/retina_zone_map.npz`)

**Interfaces:**
- Consumes: `flypaths.ANNOTATIONS`, `flypaths.OUTPUT_DIR`, `benchmark.path_comp`
- Produces:
  - `photoreceptor_zones(n_split=2) -> dict[str, tuple[np.ndarray, np.ndarray]]` — ключи `"left"`/`"right"`, значения `(индексы нейронов в матрице весов, номер зоны 0..3 для каждого)`
  - `ommatidia_zones(n_split=2) -> np.ndarray` — длина 721, номер зоны 0..3 для каждого омматидия
  - `load_or_build(n_split=2) -> dict` — с кэшем в `output/retina_zone_map.npz`
  - Константа `N_ZONES = 4`

- [ ] **Step 1: Написать падающую самопроверку**

Создать `tools/retina_zone_map.py` только с функцией самопроверки, без реализации:

```python
def self_check() -> None:
    """Проверки, которые ломаются, если карта построена неверно."""
    zones = load_or_build()

    om = zones["ommatidia"]
    assert om.shape == (721,), f"омматидиев должно быть 721, получено {om.shape}"
    assert set(np.unique(om)) == set(range(N_ZONES)), "не все зоны заполнены"
    counts = np.bincount(om, minlength=N_ZONES)
    assert counts.min() * 2 >= counts.max(), (
        f"зоны поля зрения слишком неравны: {counts.tolist()}")

    for side in ("left", "right"):
        idx, zn = zones[side]
        assert len(idx) == len(zn), f"{side}: длины не совпали"
        assert len(idx) == len(set(idx.tolist())), f"{side}: индексы повторяются"
        assert set(np.unique(zn)) == set(range(N_ZONES)), f"{side}: не все зоны заполнены"
        c = np.bincount(zn, minlength=N_ZONES)
        assert c.min() * 2 >= c.max(), f"{side}: зоны листа слишком неравны: {c.tolist()}"

    assert not (set(zones["left"][0].tolist()) & set(zones["right"][0].tolist())), \
        "левые и правые фоторецепторы пересеклись"
    print("самопроверка карты зон: ОК")


if __name__ == "__main__":
    self_check()
```

- [ ] **Step 2: Убедиться, что самопроверка падает**

Run: `.venv/bin/python tools/retina_zone_map.py`
Expected: FAIL с `NameError: name 'load_or_build' is not defined`

- [ ] **Step 3: Реализовать карту**

Дописать выше самопроверки:

```python
"""Зонная карта: какой участок поля зрения на какие фоторецепторы идёт.

Сейчас весь глаз сводится к одному числу и все зрительные нейроны своей
стороны получают одинаковую стимуляцию. Здесь строится грубая пространственная
карта: лист R1-6 режется на 4 зоны, поле зрения flygym — на те же 4 зоны.

Ориентация листа относительно поля зрения из координат НЕ выводится, поэтому
соответствие зон произвольно и зафиксировано один раз. Утверждать по этой карте
«объект слева» нельзя. Для различимости зон и для приближения объекта
ориентация не нужна: оба свойства не зависят от поворота карты.

Ретинотопия по LC закрыта отрицательно (tools/lc_retinotopy.py): их координаты
почти вырождены в линию. На входе картина другая — у R1-6 третья главная
компонента 0.000, лист плоский.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flypaths import ANNOTATIONS, OUTPUT_DIR, add_fly_brain_to_path  # noqa: E402

add_fly_brain_to_path()
from benchmark import path_comp  # noqa: E402

N_ZONES = 4
CACHE = OUTPUT_DIR / "retina_zone_map.npz"
OMMATIDIA_MAP = (Path(__import__("flygym").__file__).parent /
                 "assets/model/neuromechfly/vision/ommatidia_id_map.npy")


def _split_2d(P: np.ndarray) -> np.ndarray:
    """Разрезать облако точек на 4 по медиане вдоль каждой из двух осей.

    Медиана, а не среднее: она делит пополам по числу точек, поэтому зоны
    выходят равными по населению даже при неравномерном покрытии листа
    (у R1-6 отношение максимума к медиане расстояния до соседа 7-12).
    """
    a = (P[:, 0] > np.median(P[:, 0])).astype(int)
    b = (P[:, 1] > np.median(P[:, 1])).astype(int)
    return a * 2 + b


def photoreceptor_zones(n_split: int = 2) -> dict:
    comp = pd.read_csv(path_comp, index_col=0)
    flyid2i = {int(j): i for i, j in enumerate(comp.index)}

    ann = pd.read_csv(ANNOTATIONS, sep="\t", low_memory=False)
    ann["root_id"] = pd.to_numeric(ann["root_id"], errors="coerce")
    ann = ann.dropna(subset=["root_id"])
    ann["root_id"] = ann["root_id"].astype("int64")
    ann = ann[ann["root_id"].isin(flyid2i.keys()) & ann["pos_x"].notna()]

    out = {}
    for side in ("left", "right"):
        s = ann[(ann["cell_type"] == "R1-6") & (ann["side"] == side)]
        idx = np.array([flyid2i[i] for i in s["root_id"]], dtype=np.int64)
        X = s[["pos_x", "pos_y", "pos_z"]].to_numpy(float)
        X = X - X.mean(axis=0)
        # Две первые главные оси задают плоскость листа. Третья у R1-6 забирает
        # 0.000-0.001 дисперсии, то есть лист действительно плоский.
        Vt = np.linalg.svd(X, full_matrices=False)[2]
        out[side] = (idx, _split_2d(X @ Vt[:2].T))
    return out


def ommatidia_zones(n_split: int = 2) -> np.ndarray:
    """Номер зоны для каждого из 721 омматидия.

    В карте flygym id=0 это фон (60112 пикселей), реальные омматидии
    пронумерованы 1..721. Показание с индексом i соответствует id i+1 —
    это утверждение проверяется ассертом ниже.
    """
    m = np.load(OMMATIDIA_MAP)
    ids = np.unique(m)
    ids = ids[ids > 0]
    assert ids.min() == 1 and ids.max() == 721 and len(ids) == 721, \
        f"неожиданная нумерация омматидиев: {ids.min()}..{ids.max()}, {len(ids)} шт"

    flat = m.ravel().astype(np.int64)
    yy, xx = np.divmod(np.arange(flat.size), m.shape[1])
    cnt = np.bincount(flat, minlength=722).astype(float)
    cy = np.bincount(flat, weights=yy, minlength=722)[ids] / cnt[ids]
    cx = np.bincount(flat, weights=xx, minlength=722)[ids] / cnt[ids]
    return _split_2d(np.stack([cx, cy], axis=1))


def load_or_build(n_split: int = 2) -> dict:
    if CACHE.exists():
        z = np.load(CACHE)
        return {"ommatidia": z["ommatidia"],
                "left": (z["left_idx"], z["left_zone"]),
                "right": (z["right_idx"], z["right_zone"])}
    pr = photoreceptor_zones(n_split)
    om = ommatidia_zones(n_split)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, ommatidia=om,
             left_idx=pr["left"][0], left_zone=pr["left"][1],
             right_idx=pr["right"][0], right_zone=pr["right"][1])
    return {"ommatidia": om, "left": pr["left"], "right": pr["right"]}
```

- [ ] **Step 4: Убедиться, что самопроверка проходит**

Run: `.venv/bin/python tools/retina_zone_map.py`
Expected: PASS, печатает `самопроверка карты зон: ОК`

- [ ] **Step 5: Убедиться, что кэш работает и повторный запуск ничего не пересобирает**

Run: `.venv/bin/python tools/retina_zone_map.py && ls -l output/retina_zone_map.npz && .venv/bin/python tools/retina_zone_map.py`
Expected: оба запуска печатают `ОК`, файл существует, второй запуск заметно быстрее

- [ ] **Step 6: Закоммитить**

```bash
printf '\n# Производное от аннотаций FlyWire\noutput/retina_zone_map.npz\n' >> .gitignore
git add tools/retina_zone_map.py .gitignore
git commit -m "Зонная карта фоторецепторов: лист R1-6 и поле зрения режутся на 4 зоны"
```

---

### Task 2: Рабочая частота стимуляции зоны

**Files:**
- Create: `tools/zone_probe.py`

**Interfaces:**
- Consumes: `retina_zone_map.load_or_build`, `closed_loop_vision.load_brain_assets`, `run_pytorch as rp`
- Produces:
  - `probe_zone(weights, n, dn_idx, stim_idx, rate_hz, seed, device, transient_ms=300.0, measure_ms=500.0) -> np.ndarray` — вектор частот нисходящих, длина `len(dn_idx)`
  - `descending_indices(device) -> tuple[np.ndarray, np.ndarray]` — `(индексы нисходящих, сторона каждого)`

**Зачем отдельная задача:** широкий вход уже уводил сеть в насыщение — при переходе с 87 нейронов LC9 на ~4000 зрительных проекционных частоты пришлось снижать вчетверо. Зона это ~1000 фоторецепторов, и рабочую точку надо померить, а не угадать.

- [ ] **Step 1: Написать падающую проверку насыщения**

Создать `tools/zone_probe.py` с проверкой, без реализации:

```python
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
```

- [ ] **Step 2: Убедиться, что проверка падает**

Run: `.venv/bin/python tools/zone_probe.py`
Expected: FAIL с `NameError: name 'load_brain_assets' is not defined`

- [ ] **Step 3: Реализовать стимуляцию зоны**

Дописать выше самопроверки. Схема повторяет `_select_population` из `closed_loop_vision.py`: своя модель на каждый набор стимулируемых, переходный процесс отбрасывается, спайки нисходящих копятся и делятся на время измерения.

```python
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
```

- [ ] **Step 4: Прогнать проверку и записать рабочую точку**

Run: `.venv/bin/python tools/zone_probe.py 2>&1 | tee output/zone_rate_sweep.log`
Expected: PASS. Пять строк с растущим откликом, последний прирост больше 2%.

Если ассерт про полку падает — это не поломка кода, а результат: рабочую точку взять на той частоте, где прирост ещё есть, и записать её в `RATE_HZ` в Task 3. Записать выбранное значение и причину в `output/zone_rate_sweep.log`.

- [ ] **Step 5: Закоммитить**

```bash
git add tools/zone_probe.py output/zone_rate_sweep.log
git commit -m "Стимуляция зоны и развёртка по частоте: рабочая точка вне насыщения"
```

---

### Task 3: Шлагбаум — различимы ли зоны

**Files:**
- Create: `tools/zone_discrimination.py`

**Interfaces:**
- Consumes: `zone_probe.probe_zone`, `zone_probe.descending_indices`, `retina_zone_map.load_or_build`
- Produces: `output/zone_discrimination.csv` с колонками `pair_kind,zone_a,zone_b,seed_a,seed_b,corr`; вердикт в stdout

**Критерий зафиксирован до прогона:** межзонная корреляция должна быть ниже внутризонной с разделением по Уэлчу |t| > 3. Внутризонная корреляция — потолок, задаваемый пуассоновским шумом: это одна и та же зона на разных seed, различаться там нечему.

- [ ] **Step 1: Написать тест на арифметику вердикта**

Логика вердикта проверяется на синтетике, без прогона мозга — иначе ошибку в формуле не отличить от отсутствия эффекта.

```python
def self_check() -> None:
    """Вердикт на заведомо различимых и заведомо неразличимых данных."""
    rng = np.random.default_rng(0)

    # Зоны различимы: у каждой свой профиль, шум мелкий.
    base = rng.normal(size=(N_ZONES, 50))
    resp = {(z, s): base[z] + 0.05 * rng.normal(size=50)
            for z in range(N_ZONES) for s in range(3)}
    within, between, t = compare(resp)
    assert np.mean(within) > np.mean(between), "различимый случай перепутан"
    assert t > 3.0, f"различимый случай не прошёл порог: t={t:.2f}"

    # Зоны неразличимы: общий профиль, отличается только шум.
    one = rng.normal(size=50)
    resp = {(z, s): one + 0.05 * rng.normal(size=50)
            for z in range(N_ZONES) for s in range(3)}
    within, between, t = compare(resp)
    assert t < 3.0, f"неразличимый случай ошибочно прошёл: t={t:.2f}"

    print("самопроверка вердикта: ОК")
```

- [ ] **Step 2: Убедиться, что тест падает**

Run: `.venv/bin/python tools/zone_discrimination.py --self-check`
Expected: FAIL с `NameError: name 'compare' is not defined`

- [ ] **Step 3: Реализовать сравнение и вердикт**

```python
def compare(resp: dict):
    """Корреляции внутри зоны и между зонами, плюс t по Уэлчу.

    Внутризонная пара — одна зона, разные seed: это потолок, выше которого
    ничего быть не может, потому что различаться там нечему кроме шума счёта.
    """
    keys = sorted(resp)
    within, between = [], []
    for i, (za, sa) in enumerate(keys):
        for zb, sb in keys[i + 1:]:
            c = float(np.corrcoef(resp[(za, sa)], resp[(zb, sb)])[0, 1])
            (within if za == zb else between).append(c)
    within, between = np.array(within), np.array(between)
    # Уэлч: дисперсии у двух выборок разные, объединять их нельзя.
    se = np.sqrt(within.var(ddof=1) / len(within) + between.var(ddof=1) / len(between))
    t = (within.mean() - between.mean()) / max(se, 1e-12)
    return within, between, t
```

- [ ] **Step 4: Убедиться, что тест проходит**

Run: `.venv/bin/python tools/zone_discrimination.py --self-check`
Expected: PASS, печатает `самопроверка вердикта: ОК`

- [ ] **Step 5: Реализовать прогон и запуск**

Шапка файла, до всего остального:

```python
"""Шлагбаум: различает ли мозг, какой участок поля зрения засвечен."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flypaths import out  # noqa: E402
from tools.retina_zone_map import load_or_build, N_ZONES  # noqa: E402
from tools.zone_probe import probe_zone, descending_indices  # noqa: E402
from closed_loop_vision import load_brain_assets  # noqa: E402
```

Дальше сам прогон:

```python
# Частота берётся из output/zone_rate_sweep.log — последняя, на которой прирост
# отклика ещё больше 2%. Ниже насыщения: в полке пространственная структура
# не проходит, все зоны дают одинаковый ответ независимо от того, что подано.
RATE_HZ = 100.0
SEEDS = (0, 1, 2)


def main():
    if "--self-check" in sys.argv:
        self_check()
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assets = load_brain_assets(device)
    dn_idx, _ = descending_indices(device)
    zones = load_or_build()
    idx, zn = zones["left"]          # шлагбаум ставим на одном глазу

    resp = {}
    for z in range(N_ZONES):
        stim = idx[zn == z].tolist()
        for s in SEEDS:
            print(f"  зона {z}, seed {s}: {len(stim)} фоторецепторов...", flush=True)
            resp[(z, s)] = probe_zone(assets["weights"], assets["n"], dn_idx,
                                      stim, RATE_HZ, seed=s, device=device)

    within, between, t = compare(resp)
    rows = []
    keys = sorted(resp)
    for i, (za, sa) in enumerate(keys):
        for zb, sb in keys[i + 1:]:
            rows.append({"pair_kind": "within" if za == zb else "between",
                         "zone_a": za, "zone_b": zb, "seed_a": sa, "seed_b": sb,
                         "corr": float(np.corrcoef(resp[(za, sa)], resp[(zb, sb)])[0, 1])})
    pd.DataFrame(rows).to_csv(out("zone_discrimination.csv"), index=False)

    print("=" * 70)
    print(f"  внутризонная корреляция (потолок шума): {within.mean():.4f} ± {within.std(ddof=1):.4f}")
    print(f"  межзонная корреляция:                   {between.mean():.4f} ± {between.std(ddof=1):.4f}")
    print(f"  Уэлч t = {t:.2f}, порог 3.0")
    print("  ВЕРДИКТ:", "зоны различимы, этап 2 разрешён" if t > 3.0
          else "зоны НЕразличимы, этап 2 не строится")
```

- [ ] **Step 6: Прогнать шлагбаум**

Run: `.venv/bin/python tools/zone_discrimination.py 2>&1 | tee output/zone_discrimination.log`
Expected: 12 прогонов мозга, порядка 15 минут, в конце вердикт.

- [ ] **Step 7: Закоммитить результат независимо от знака**

```bash
git add tools/zone_discrimination.py output/zone_discrimination.csv output/zone_discrimination.log
git commit -m "Шлагбаум различимости зон: <вердикт и числа в теле коммита>"
```

**Если вердикт отрицательный:** записать результат в `PROJECT_LOG.md` — числа, критерий, почему остановились — и остановиться. Задачи 4–6 не выполняются.

---

### Task 4: Приближающийся объект в сцене

**Files:**
- Create: `closed_loop_looming.py`

**Interfaces:**
- Consumes: `closed_loop_vision.PillarWorld`, `flygym` мир и симуляция
- Produces: `move_pillar(sim, x, y) -> None`, `pillar_geom_id(sim) -> int`

**Главный риск этапа:** зрение должно видеть новую позицию объекта, а не закэшированную сцену. Проверяется первым шагом, до всего остального.

- [ ] **Step 1: Написать проверку, что сцена обновляется**

```python
def self_check() -> None:
    """Сдвинуть объект и убедиться, что показания глаз изменились.

    Если сцена закэширована, вся вторая половина работы бессмысленна,
    поэтому проверка идёт до всего остального.
    """
    sim, fly = build_standing_fly(pillar_x=30.0, pillar_y=0.0)
    far = sim.get_ommatidia_readouts(fly.name).sum(axis=2).mean(axis=1).copy()
    move_pillar(sim, 3.0, 0.0)
    for _ in range(10):
        sim.step()
    near = sim.get_ommatidia_readouts(fly.name).sum(axis=2).mean(axis=1)
    delta = float(np.abs(near - far).max())
    print(f"  яркость до сдвига {far}, после {near}, изменение {delta:.4f}")
    assert delta > 0.01, (
        "сцена не обновилась: сдвиг объекта не изменил показания глаз. "
        "Запасной вариант — растить радиус вместо сдвига")
    assert near.mean() < far.mean(), "объект приблизился, а глаза посветлели"
    print("самопроверка сцены: ОК")
```

- [ ] **Step 2: Убедиться, что проверка падает**

Run: `.venv/bin/python closed_loop_looming.py --self-check`
Expected: FAIL с `NameError: name 'build_standing_fly' is not defined`

- [ ] **Step 3: Реализовать сцену и сдвиг**

```python
"""Этап 2: неподвижная муха и приближающийся объект."""
import argparse
import sys
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flypaths import out  # noqa: E402  импорт до flygym: он ставит MUJOCO_GL
from closed_loop_vision import PillarWorld  # noqa: E402

from flygym.simulation import Simulation  # noqa: E402
from flygym.utils.math import Rotation3D  # noqa: E402
from flygym_demo.complex_terrain import make_locomotion_fly  # noqa: E402


def pillar_geom_id(sim) -> int:
    """id геометрии столба. Имя может быть с префиксом от привязки к миру."""
    for gid in range(sim.mj_model.ngeom):
        name = mujoco.mj_id2name(sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if name and name.endswith("pillar"):
            return gid
    raise RuntimeError("геометрия pillar не найдена в модели")


def move_pillar(sim, x: float, y: float) -> None:
    """Передвинуть столб. Он статический, степеней свободы нет, поэтому
    позиция меняется в модели, а не в состоянии."""
    gid = pillar_geom_id(sim)
    sim.mj_model.geom_pos[gid][:2] = (x, y)


def build_standing_fly(pillar_x: float, pillar_y: float):
    fly = make_locomotion_fly()
    fly.add_vision()
    world = PillarWorld(pillar_x, pillar_y)
    world.add_fly(fly, spawn_position=[0.0, 0.0, 0.5],
                  spawn_rotation=Rotation3D("quat", [1, 0, 0, 0]),
                  add_ground_contact_sensors=True)
    sim = Simulation(world)
    sim.reset()
    sim.warmup(0.05)
    return sim, fly
```

- [ ] **Step 4: Убедиться, что проверка проходит**

Run: `.venv/bin/python closed_loop_looming.py --self-check`
Expected: PASS, печатает изменение яркости и `самопроверка сцены: ОК`

Если ассерт про обновление сцены падает — перейти на запасной вариант: вместо `geom_pos` менять `sim.mj_model.geom_size[gid][0]`, растя радиус. Это хуже по честности (объект не приближается, а раздувается), но эксперимент не блокирует. Причину и выбор записать в шапку файла.

- [ ] **Step 5: Закоммитить**

```bash
git add closed_loop_looming.py
git commit -m "Сцена с приближающимся объектом; проверено, что зрение видит сдвиг"
```

---

### Task 5: Опыт целиком, четыре условия по 8 seed

**Files:**
- Modify: `closed_loop_looming.py`

**Interfaces:**
- Consumes: `closed_loop_vision.load_brain_assets`, `retina_zone_map.load_or_build`, `move_pillar`, `build_standing_fly`
- Produces: `output/looming_<условие>_s<seed>.csv` на каждый прогон и сводный `output/looming_summary.csv` с колонками `condition,seed,path_mm,heading_deg,cmd_left_mean,cmd_right_mean,moved`

**Ход опыта:** первые 20 циклов объект стоит далеко и в поле зрения не попадает — это проверка, что муха действительно неподвижна. С 21-го цикла он приближается. Измерения берутся с момента начала движения, окно 30 циклов.

**Базовая команда 0:** пустое поле даёт команду 0, муха стоит. Нулевая гипотеза становится «не сдвинулась вообще».

- [ ] **Step 1: Написать проверку, что муха без стимула стоит**

```python
def check_standing() -> None:
    """Без объекта муха обязана остаться на месте.

    Если она уходит сама, нулевая гипотеза разрушена и весь опыт бессмыслен.
    Порог 0.5 мм взят с запасом: в tools/body_walk_check.py стоящая муха
    за сопоставимое окно проходит 0.022 мм.
    """
    summary = run_looming(condition="none", seed=0, cycles=50, verbose=False)
    print(f"  без объекта путь {summary['path_mm']:.3f} мм")
    assert summary["path_mm"] < 0.5, (
        f"муха ушла без стимула на {summary['path_mm']:.3f} мм — "
        "базовая команда не ноль либо контур не в покое")
    print("самопроверка покоя: ОК")
```

- [ ] **Step 2: Убедиться, что проверка падает**

Run: `.venv/bin/python closed_loop_looming.py --check-standing`
Expected: FAIL с `NameError: name 'run_looming' is not defined`

- [ ] **Step 3: Реализовать прогон**

Сначала дополнить шапку `closed_loop_looming.py` тем, чего в ней после Task 4 не было:

```python
import pandas as pd
import torch

from flypaths import add_fly_brain_to_path
from tools.retina_zone_map import load_or_build, N_ZONES
from closed_loop_vision import (load_brain_assets, DT_BRAIN_MS, SYNC_MS,
                                DARK_GAIN, TAU_CMD_MS)

add_fly_brain_to_path()
import run_pytorch as rp  # noqa: E402

from flygym_demo.complex_terrain import (  # noqa: E402
    HybridControllerObservation,
    HybridTurningController,
    PreprogrammedSteps,
    apply_locomotion_action,
)
```

Четыре условия задаются траекторией объекта и способом кодирования картинки:

```python
FAR = 500.0
START_X, END_X = 30.0, 3.0
ONSET, WINDOW = 20, 30
CONDITIONS = ("approach", "none", "recede", "approach_scalar")


def pillar_track(condition: str, cycle: int) -> tuple[float, float]:
    """Где столб на данном цикле."""
    if condition == "none":
        return FAR, FAR
    if cycle < ONSET:
        return (START_X, 0.0) if condition != "recede" else (END_X, 0.0)
    frac = min((cycle - ONSET) / WINDOW, 1.0)
    if condition == "recede":
        return END_X + (START_X - END_X) * frac, 0.0
    return START_X + (END_X - START_X) * frac, 0.0


def encode(readouts, baseline, om_zone, condition):
    """Картинка -> темнота по зонам. Возвращает массив (N_ZONES, 2 глаза).

    baseline — поомматидиевая яркость пустого поля, форма (2, 721). Хранится
    поомматидиево, а не одним числом на глаз: зоны надо сравнивать каждую со
    своей базой, иначе неравномерность освещения поля притворится объектом.

    Контрольный вход approach_scalar: одна темнота на глаз, размноженная по
    зонам, — ровно то, что было до этой работы. Если он двигает муху не хуже,
    пространственная карта не добавила ничего.
    """
    per_om = readouts.sum(axis=2)                      # (2, 721)
    if condition == "approach_scalar":
        inten = per_om.mean(axis=1)                    # (2,)
        base = baseline.mean(axis=1)
        dark = np.clip((base - inten) / np.maximum(base, 1e-6) * DARK_GAIN, 0, 1)
        return np.repeat(dark[None, :], N_ZONES, axis=0)
    res = np.zeros((N_ZONES, 2))
    for z in range(N_ZONES):
        sel = om_zone == z
        inten = per_om[:, sel].mean(axis=1)
        base = baseline[:, sel].mean(axis=1)
        res[z] = np.clip((base - inten) / np.maximum(base, 1e-6) * DARK_GAIN, 0, 1)
    return res
```

Команда считается иначе, чем в `closed_loop_vision.py`: там пустое поле
приравнено к команде 0.65, здесь — к нулю, иначе муха не будет стоять.

```python
RATE_BASE, RATE_SPAN = 40.0, 120.0   # рабочая точка из output/zone_rate_sweep.log
CMD_SPAN = 0.5                       # подъём отклика на 50% над базой = команда 1.0


def zone_rates(dark, zones, n, device):
    """Темнота по зонам -> вектор частот стимуляции длиной n."""
    rates = torch.zeros(1, n, device=device)
    for side, eye in (("left", 0), ("right", 1)):
        idx, zn = zones[side]
        for z in range(N_ZONES):
            sel = idx[zn == z]
            rates[:, sel.tolist()] = RATE_BASE + RATE_SPAN * float(dark[z, eye])
    return rates


def run_looming(condition, seed, cycles=60, assets=None, device="cpu",
                video_path=None, verbose=True):
    """Один прогон. Возвращает словарь сводки."""
    zones = load_or_build()
    om_zone = zones["ommatidia"]
    if assets is None:
        assets = load_brain_assets(device, verbose=verbose)
    n, read_l, read_r = assets["n"], assets["read_l"], assets["read_r"]

    stim_all = sorted(set(zones["left"][0].tolist()) | set(zones["right"][0].tolist()))
    model = rp.TorchModel(1, n, DT_BRAIN_MS, rp.MODEL_PARAMS, assets["weights"],
                          exc_indices=stim_all, device=device)
    cond, delay_buf, spikes, v, refrac = model.state_init()
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    x0, y0 = pillar_track(condition, 0)
    sim, fly = build_standing_fly(x0, y0)
    controller = HybridTurningController(timestep=sim.timestep,
                                         preprogrammed_steps=PreprogrammedSteps())
    controller.reset(seed=seed)

    inner_brain = int(SYNC_MS / DT_BRAIN_MS)
    inner_phys = int(round(SYNC_MS / 1000.0 / sim.timestep))
    read_l_t = torch.tensor(read_l, dtype=torch.long, device=device)
    read_r_t = torch.tensor(read_r, dtype=torch.long, device=device)

    def step_brain(rates):
        nonlocal cond, delay_buf, spikes, v, refrac
        acc_l = acc_r = 0.0
        for _ in range(inner_brain):
            cond, delay_buf, spikes, v, refrac = model(
                rates, cond, delay_buf, spikes, v, refrac, generator=gen)
            acc_l += float(spikes[0, read_l_t].sum())
            acc_r += float(spikes[0, read_r_t].sum())
        sec = SYNC_MS / 1000.0
        return acc_l / sec / len(read_l), acc_r / sec / len(read_r)

    def step_body(cmd):
        for _ in range(inner_phys):
            obs = HybridControllerObservation.from_sim(sim, fly.name)
            apply_locomotion_action(sim, fly.name, controller.step(cmd, obs))
            sim.step()

    # Калибровка яркости: объект далеко, муха стоит. Походки нет, значит нет и
    # крена на 22 Гц — базовая линия чище, чем у идущей мухи.
    move_pillar(sim, FAR, FAR)
    base_acc = []
    for _ in range(10):
        step_body(np.array([0.0, 0.0]))
        base_acc.append(sim.get_ommatidia_readouts(fly.name).sum(axis=2))
    baseline = np.mean(base_acc, axis=0)                     # (2, 721)

    # Калибровка отклика: мозг на базовой частоте без объекта.
    flat = zone_rates(np.zeros((N_ZONES, 2)), zones, n, device)
    resp_l = resp_r = 0.0
    n_cal = int(2000.0 / SYNC_MS)
    for w in range(n_cal):
        a, b = step_brain(flat)
        if w >= n_cal // 4:
            resp_l += a
            resp_r += b
    m = max(n_cal - n_cal // 4, 1)
    base_l, base_r = max(resp_l / m, 1e-6), max(resp_r / m, 1e-6)

    ema_l = ema_r = None
    alpha = 1.0 - float(np.exp(-SYNC_MS / TAU_CMD_MS))
    rows, path = [], 0.0
    prev = None
    for cycle in range(cycles):
        px, py = pillar_track(condition, cycle)
        move_pillar(sim, px, py)
        ro = sim.get_ommatidia_readouts(fly.name).sum(axis=2, keepdims=True)
        dark = encode(np.repeat(ro, 2, axis=2), baseline, om_zone, condition)
        hz_l, hz_r = step_brain(zone_rates(dark, zones, n, device))
        if ema_l is None:
            ema_l, ema_r = hz_l, hz_r
        else:
            ema_l += alpha * (hz_l - ema_l)
            ema_r += alpha * (hz_r - ema_r)
        # Ноль при пустом поле: команда это подъём НАД базой, а не доля потолка.
        cmd_l = float(np.clip((ema_l - base_l) / (base_l * CMD_SPAN), 0.0, 1.0))
        cmd_r = float(np.clip((ema_r - base_r) / (base_r * CMD_SPAN), 0.0, 1.0))
        step_body(np.array([cmd_l, cmd_r]))

        pos = sim.get_body_positions(fly.name)[THORAX_IDX][:2]
        if cycle >= ONSET:
            path += 0.0 if prev is None else float(np.hypot(*(pos - prev)))
            prev = pos
        rows.append({"cycle": cycle, "cmd_left": cmd_l, "cmd_right": cmd_r,
                     "dark_max": float(dark.max()),
                     "x_mm": float(pos[0]), "y_mm": float(pos[1])})

    df = pd.DataFrame(rows)
    df.to_csv(out(f"looming_{condition}_s{seed}.csv"), index=False)
    return {"condition": condition, "seed": seed, "path_mm": path,
            "cmd_left_mean": df.cmd_left.mean(), "cmd_right_mean": df.cmd_right.mean(),
            "moved": path > 0.5}
```

`THORAX_IDX` берётся один раз при сборке тела тем же способом, что в
`closed_loop_vision.py`: `fly.get_bodysegs_order().index(type(fly).BODY_SEGMENT_CLASS("c_thorax"))`.

- [ ] **Step 4: Убедиться, что проверка покоя проходит**

Run: `.venv/bin/python closed_loop_looming.py --check-standing`
Expected: PASS, путь меньше 0.5 мм

- [ ] **Step 5: Прогнать все четыре условия по 8 seed**

Run: `.venv/bin/python closed_loop_looming.py --all --seeds 8 2>&1 | tee output/looming_run.log`
Expected: 32 прогона. Веса грузятся один раз на серию — как в `tools/replicate_vision.py`, иначе минута на каждый прогон.

- [ ] **Step 6: Закоммитить**

```bash
git add closed_loop_looming.py output/looming_summary.csv output/looming_run.log
git commit -m "Опыт с приближением: четыре условия по 8 seed"
```

---

### Task 6: Анализ, вердикт, запись в журнал

**Files:**
- Create: `tools/analyze_looming.py`
- Modify: `PROJECT_LOG.md`, `README.md`

**Критерий зафиксирован до прогона:** приближение против «без объекта» по Уэлчу |t| > 3, и эффект виден на каждом seed отдельно.

Отдельный анализатор, а не `tools/analyze_replication.py`: там сравниваются сцены по повороту курса, здесь — условия по пройденному пути, и добавлен критерий «на каждом seed», которого в существующем нет.

- [ ] **Step 1: Написать тест на арифметику вердикта**

```python
def self_check() -> None:
    """Вердикт на заведомо положительных и заведомо пустых данных."""
    strong = pd.DataFrame(
        [{"condition": "approach", "seed": s, "path_mm": 8.0 + 0.1 * s} for s in range(8)]
        + [{"condition": "none", "seed": s, "path_mm": 0.05 * s} for s in range(8)])
    t, per_seed = verdict(strong)
    assert t > 3.0 and per_seed, f"явный эффект не распознан: t={t:.2f}, per_seed={per_seed}"

    empty = pd.DataFrame(
        [{"condition": c, "seed": s, "path_mm": 0.1 * s}
         for c in ("approach", "none") for s in range(8)])
    t, per_seed = verdict(empty)
    assert t < 3.0 and not per_seed, f"пустые данные прошли порог: t={t:.2f}"

    # Средние расходятся, но один seed выбивается — «на каждом seed» обязано упасть.
    ragged = pd.DataFrame(
        [{"condition": "approach", "seed": s, "path_mm": 8.0 if s else 0.0} for s in range(8)]
        + [{"condition": "none", "seed": s, "path_mm": 0.1} for s in range(8)])
    _, per_seed = verdict(ragged)
    assert not per_seed, "выбивающийся seed не пойман"
    print("самопроверка вердикта: ОК")
```

- [ ] **Step 2: Убедиться, что тест падает, затем реализовать вердикт**

Run: `.venv/bin/python tools/analyze_looming.py --self-check`
Expected сначала: FAIL с `NameError: name 'verdict' is not defined`

```python
def verdict(df):
    a = df[df.condition == "approach"].path_mm.to_numpy()
    n = df[df.condition == "none"].path_mm.to_numpy()
    se = np.sqrt(a.var(ddof=1) / len(a) + n.var(ddof=1) / len(n))
    t = (a.mean() - n.mean()) / max(se, 1e-12)
    per_seed = bool((a > n.max()).all())   # эффект на каждом seed отдельно
    return t, per_seed
```

Run: `.venv/bin/python tools/analyze_looming.py --self-check`
Expected после: PASS, печатает `самопроверка вердикта: ОК`

- [ ] **Step 3: Посчитать вердикт на реальных данных**

Run: `.venv/bin/python tools/analyze_looming.py`
Expected: печатает t по всем трём сравнениям и флаг `per_seed`.

- [ ] **Step 4: Сверить с контролями**

Три сравнения, каждое отвечает на свой вопрос:

```
приближение против «без объекта»       двигает ли стимул вообще
приближение против удаления            приближение или любое изменение яркости
приближение против approach_scalar     добавила ли пространственная карта хоть что-то
```

Третье сравнение записывается честно независимо от знака. Если `approach_scalar` не хуже — карта ничего не дала, и так и пишем.

- [ ] **Step 5: Записать результат в журнал**

В `PROJECT_LOG.md` — числа по всем четырём условиям, три сравнения, критерий, зафиксированный заранее, и вывод. При отрицательном результате прежние записи не переписываются: в проекте принято сохранять и заявление, и его опровержение.

- [ ] **Step 6: Закоммитить**

```bash
git add PROJECT_LOG.md README.md tools/analyze_looming.py output/
git commit -m "Итог по пространственному входу: <вердикт и ключевые числа>"
```

---

## Отступление от спецификации

В спецификации файлов было три, здесь четыре: стимуляция зоны вынесена из
`tools/zone_discrimination.py` в отдельный `tools/zone_probe.py`. Причина в том,
что она нужна дважды — развёртке по частоте и самому шлагбауму, — и её
самопроверка (растёт ли отклик, нет ли полки) осмысленна отдельно от
самопроверки вердикта. На состав работ и на критерии это не влияет.

## Что план сознательно не делает

- Полную поомматидиевую карту. Она требует ориентации листа, которой нет, и неустойчива к неравномерности покрытия 7–12. Строить её имеет смысл только после того, как зонная покажет различимость.
- Утверждения о направлении («объект слева»). Из произвольного соответствия зон они не следуют.
- Новый механизм видео и новое считывание команды. `--video` и популяционное считывание уже есть и проверены.
