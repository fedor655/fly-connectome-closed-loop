"""Задача 2a: где по пути R1-6 -> нисходящие гаснет сигнал.

Диагностика после того, как стимуляция зоны 0 левых R1-6 дала 0.00 Гц на
нисходящих на всех частотах 25/50/100/200/400 Гц (Задача 2). Неизвестно,
дефект это вызова TorchModel(exc_indices=...) / вектора rates, или свойство
модели (путь просто гасит сигнал на каком-то слое). Отвечает не рассуждение,
а прогон с замером по промежуточным станциям пути:

    R1-6 (зона 0, стимулируемые) -> L1/L2/L3 -> Mi1/Tm1/Tm9 -> LC9/LPLC2
    -> descending

Два этапа в ОДНОМ процессе (веса грузятся один раз):

  ЭТАП 1 — решающая проверка. Окно 100 мс переходного процесса + 200 мс
  замера (~3000 шагов), только R1-6 и descending. Если сама стимулируемая
  популяция R1-6 не разряжается заметно выше нуля — это дефект стимуляции
  (TorchModel(exc_indices=...) / вектор rates), и на этом всё: дальше по
  пути смотреть незачем. Результат пишется в output/zone_pathway_stage1.csv
  сразу после этапа, чтобы не потерять его при обрыве связи.

  ЭТАП 2 — полная таблица по брифу, только если этап 1 показал, что R1-6
  разряжаются. Окно по брифу: 300 мс переходного процесса + 1000 мс замера,
  стимулированный и контрольный (rates=0) прогоны. Это ~26000 шагов модели
  и на измеренной скорости (~27 мс/шаг на этой машине) не укладывается в
  10-минутный лимit переднего плана — поэтому окно замера урезано до 500 мс
  (см. STAGE2_MEASURE_MS), как и предписано на случай нехватки времени.
  Результат пишется в output/zone_pathway.csv.

Ничего не запускается в фоне: предыдущая попытка с полным окном (1000 мс)
ушла в фон при превышении лимита и результат потерялся вместе с сессией.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flypaths import ANNOTATIONS, add_fly_brain_to_path, out  # noqa: E402
from tools.retina_zone_map import load_or_build  # noqa: E402
from closed_loop_vision import load_brain_assets, DT_BRAIN_MS  # noqa: E402

add_fly_brain_to_path()
import run_pytorch as rp  # noqa: E402
from benchmark import path_comp  # noqa: E402

STIM_RATE_HZ = 200.0
SEED = 0
DECISIVE_THRESHOLD_HZ = 1.0  # "заметно выше нуля" для решающей проверки этапа 1

# Порядок — как в брифе: от стимулируемого входа до выхода мозга.
CELL_TYPE_POPULATIONS = ["L1", "L2", "L3", "Mi1", "Tm1", "Tm9", "LC9", "LPLC2"]
POP_ORDER = ["R1-6"] + CELL_TYPE_POPULATIONS + ["descending"]

STAGE1_POPULATIONS = ["R1-6", "descending"]
STAGE1_TRANSIENT_MS = 100.0
STAGE1_MEASURE_MS = 200.0

STAGE2_TRANSIENT_MS = 300.0
STAGE2_MEASURE_MS = 500.0  # брифовые 1000 мс дают ~26000 шагов (~700 с) — не
                           # укладываются в лимит переднего плана; урезано.


def build_populations(flyid2i: dict) -> dict:
    """Индексы (в пространстве весовой матрицы) для каждой популяции пути.

    R1-6 — зона 0 левых R1-6 из retina_zone_map (стимулируемые нейроны).
    L1/L2/L3/Mi1/Tm1/Tm9/LC9/LPLC2 — по cell_type, только сторона left.
    descending — по super_class == "descending", обе стороны (как в
    closed_loop_vision.py / zone_probe.py).
    """
    ann = pd.read_csv(ANNOTATIONS, sep="\t", low_memory=False)
    ann["root_id"] = pd.to_numeric(ann["root_id"], errors="coerce")
    ann = ann.dropna(subset=["root_id"])
    ann["root_id"] = ann["root_id"].astype("int64")
    ann = ann[ann["root_id"].isin(flyid2i.keys())]

    pops = {}

    zones = load_or_build()
    left_idx, left_zone = zones["left"]
    pops["R1-6"] = np.asarray(left_idx[left_zone == 0], dtype=np.int64)

    for ct in CELL_TYPE_POPULATIONS:
        s = ann[(ann["cell_type"] == ct) & (ann["side"] == "left")]
        pops[ct] = np.array([flyid2i[i] for i in s["root_id"]], dtype=np.int64)

    desc = ann[(ann["super_class"] == "descending") & ann["side"].isin(["left", "right"])]
    pops["descending"] = np.array([flyid2i[int(x)] for x in desc["root_id"]], dtype=np.int64)

    for name in POP_ORDER:
        assert len(pops[name]) > 0, f"{name}: популяция пуста — проверить annotations/cell_type"

    return pops


def run_pathway(weights, n, pops, stim_idx, rate_hz, seed, device,
                transient_ms, measure_ms):
    """Один прогон модели, средние частоты по всем популяциям в pops сразу.

    exc_indices всегда = stim_idx (та же топология модели что и в
    стимулированном, и в контрольном прогоне — в контроле просто rates=0),
    это то же самое, что использует zone_probe.probe_zone. Порядок
    популяций — порядок ключей pops (сохраняется, dict в Python 3.7+).
    """
    stim = list(stim_idx)
    model = rp.TorchModel(1, n, DT_BRAIN_MS, rp.MODEL_PARAMS, weights,
                          exc_indices=stim, device=device)
    c, d, s, v, r = model.state_init()
    rates = torch.zeros(1, n, device=device)
    if rate_hz > 0:
        rates[:, stim] = rate_hz
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    names = list(pops.keys())
    sizes = [len(pops[name]) for name in names]
    idx_all = np.concatenate([pops[name] for name in names])
    idx_all_t = torch.tensor(idx_all, dtype=torch.long, device=device)
    acc = torch.zeros(len(idx_all), device=device)

    n_tr = int(round(transient_ms / DT_BRAIN_MS))
    n_me = int(round(measure_ms / DT_BRAIN_MS))
    with torch.no_grad():
        for step in range(n_tr + n_me):
            c, d, s, v, r = model(rates, c, d, s, v, r, generator=g)
            if step >= n_tr:
                acc.add_(s[0, idx_all_t])

    parts = torch.split(acc, sizes)
    measure_s = measure_ms / 1000.0
    return {name: float(p.sum().item()) / len(p) / measure_s
            for name, p in zip(names, parts)}


def print_table(rows, columns):
    widths = {"population": 12, "n_neurons": 10, "rate_stim_hz": 16, "rate_base_hz": 16}
    header = "".join(f"{c:<12}" if c == "population" else f"{c:>{widths[c]}}" for c in columns)
    print(header)
    for row in rows:
        line = ""
        for c in columns:
            if c == "population":
                line += f"{row[c]:<12}"
            elif c == "n_neurons":
                line += f"{row[c]:>{widths[c]}}"
            else:
                line += f"{row[c]:>{widths[c]}.3f}"
        print(line)


def main() -> None:
    t_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[zone_pathway_probe] устройство: {device}")

    comp = pd.read_csv(path_comp, index_col=0)
    flyid2i = {int(j): i for i, j in enumerate(comp.index)}
    pops = build_populations(flyid2i)
    print("[zone_pathway_probe] популяции построены: "
          + ", ".join(f"{name}={len(pops[name])}" for name in POP_ORDER))

    print("[zone_pathway_probe] загружаю веса коннектома...")
    assets = load_brain_assets(device, verbose=False)
    weights, n = assets["weights"], assets["n"]
    print(f"[zone_pathway_probe] веса загружены, нейронов n={n}, "
          f"t={time.time() - t_start:.1f}с")

    stim_idx = pops["R1-6"]

    # ---------------------------------------------------------------- ЭТАП 1
    stage1_pops = {name: pops[name] for name in STAGE1_POPULATIONS}
    print(f"\n[этап 1] решающая проверка: стимуляция {STIM_RATE_HZ:.0f} Гц "
          f"на зоне 0 левых R1-6 ({len(stim_idx)} нейронов), "
          f"transient={STAGE1_TRANSIENT_MS:.0f} мс, measure={STAGE1_MEASURE_MS:.0f} мс...")
    t0 = time.time()
    stage1_rates = run_pathway(weights, n, stage1_pops, stim_idx, STIM_RATE_HZ, SEED, device,
                               transient_ms=STAGE1_TRANSIENT_MS, measure_ms=STAGE1_MEASURE_MS)
    print(f"[этап 1] прогон завершён за {time.time() - t0:.1f}с")

    stage1_rows = [{"population": name, "n_neurons": len(stage1_pops[name]),
                     "rate_stim_hz": stage1_rates[name]} for name in STAGE1_POPULATIONS]
    stage1_csv = out("zone_pathway_stage1.csv")
    pd.DataFrame(stage1_rows, columns=["population", "n_neurons", "rate_stim_hz"]) \
        .to_csv(stage1_csv, index=False)
    print(f"[этап 1] записано: {stage1_csv}")
    print_table(stage1_rows, ["population", "n_neurons", "rate_stim_hz"])

    r16_stage1 = stage1_rates["R1-6"]
    if r16_stage1 <= DECISIVE_THRESHOLD_HZ:
        print(f"\n[этап 1] R1-6 при стимуляции: {r16_stage1:.3f} Гц — практически ноль.")
        print("[этап 1] РЕШЕНИЕ: стимуляция не доходит до самих стимулируемых "
              "нейронов — это дефект вызова exc_indices/rates, а не биология сети. "
              "Этап 2 не запускается: дальше по пути смотреть незачем.")
        return
    print(f"\n[этап 1] R1-6 при стимуляции: {r16_stage1:.3f} Гц — заметно выше нуля. "
          "Стимуляция доходит до стимулируемых нейронов. Перехожу к этапу 2.")

    # ---------------------------------------------------------------- ЭТАП 2
    print(f"\n[этап 2] полная таблица: transient={STAGE2_TRANSIENT_MS:.0f} мс, "
          f"measure={STAGE2_MEASURE_MS:.0f} мс (урезано с 1000 мс брифа, см. docstring)")
    t0 = time.time()
    rate_stim = run_pathway(weights, n, pops, stim_idx, STIM_RATE_HZ, SEED, device,
                            transient_ms=STAGE2_TRANSIENT_MS, measure_ms=STAGE2_MEASURE_MS)
    print(f"[этап 2] прогон 1/2 (стимуляция) завершён за {time.time() - t0:.1f}с")

    t0 = time.time()
    rate_base = run_pathway(weights, n, pops, stim_idx, 0.0, SEED, device,
                            transient_ms=STAGE2_TRANSIENT_MS, measure_ms=STAGE2_MEASURE_MS)
    print(f"[этап 2] прогон 2/2 (контроль) завершён за {time.time() - t0:.1f}с")

    rows = [{"population": name, "n_neurons": len(pops[name]),
             "rate_stim_hz": rate_stim[name], "rate_base_hz": rate_base[name]}
            for name in POP_ORDER]
    df = pd.DataFrame(rows, columns=["population", "n_neurons", "rate_stim_hz", "rate_base_hz"])

    csv_path = out("zone_pathway.csv")
    df.to_csv(csv_path, index=False)
    print(f"[этап 2] записано: {csv_path}")

    print()
    print_table(rows, ["population", "n_neurons", "rate_stim_hz", "rate_base_hz"])

    print()
    r16_stim = rate_stim["R1-6"]
    if r16_stim > DECISIVE_THRESHOLD_HZ:
        print(f"[zone_pathway_probe] R1-6 при стимуляции: {r16_stim:.3f} Гц — "
              "стимулируемые нейроны разряжаются, дефект в вызове TorchModel "
              "исключён на этом уровне.")
    else:
        print(f"[zone_pathway_probe] R1-6 при стимуляции: {r16_stim:.3f} Гц — "
              "почти ноль. Стимуляция не доходит до самих стимулируемых "
              "нейронов: это дефект вызова exc_indices/rates, а не биология сети.")

    print(f"\n[zone_pathway_probe] всего времени: {time.time() - t_start:.1f}с")


# ============================================================================
# Длинное окно (--long): закрывает оговорку "мало ждали" из первого прогона
# этапа 2 (там замер был урезан до 500 мс, чтобы уложиться в лимит переднего
# плана). Здесь тот же полный список из 10 популяций, но замер 2000 мс —
# ровно столько берёт _select_population в closed_loop_vision.py для отклика
# на зрительных проекционных, то есть окно заведомо достаточное. Рассчитан
# на запуск через nohup в истинный фон (см. docstring модуля): CSV пишется
# на диск сразу после КАЖДОГО из двух прогонов, а не только в конце.
# ============================================================================
LONG_TRANSIENT_MS = 300.0
LONG_MEASURE_MS = 2000.0


def run_long() -> None:
    t_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[zone_pathway_long] устройство: {device}")

    comp = pd.read_csv(path_comp, index_col=0)
    flyid2i = {int(j): i for i, j in enumerate(comp.index)}
    pops = build_populations(flyid2i)
    print("[zone_pathway_long] популяции построены: "
          + ", ".join(f"{name}={len(pops[name])}" for name in POP_ORDER))

    print("[zone_pathway_long] загружаю веса коннектома...")
    assets = load_brain_assets(device, verbose=False)
    weights, n = assets["weights"], assets["n"]
    print(f"[zone_pathway_long] веса загружены, нейронов n={n}, "
          f"t={time.time() - t_start:.1f}с")

    stim_idx = pops["R1-6"]
    csv_path = out("zone_pathway_long.csv")
    columns = ["population", "n_neurons", "rate_stim_hz", "rate_base_hz"]

    print(f"[zone_pathway_long] прогон 1/2: стимуляция {STIM_RATE_HZ:.0f} Гц "
          f"на зоне 0 левых R1-6 ({len(stim_idx)} нейронов), "
          f"transient={LONG_TRANSIENT_MS:.0f} мс, measure={LONG_MEASURE_MS:.0f} мс, "
          f"seed={SEED}...")
    t0 = time.time()
    rate_stim = run_pathway(weights, n, pops, stim_idx, STIM_RATE_HZ, SEED, device,
                            transient_ms=LONG_TRANSIENT_MS, measure_ms=LONG_MEASURE_MS)
    print(f"[zone_pathway_long] прогон 1/2 завершён за {time.time() - t0:.1f}с")

    rows_partial = [{"population": name, "n_neurons": len(pops[name]),
                      "rate_stim_hz": rate_stim[name], "rate_base_hz": float("nan")}
                     for name in POP_ORDER]
    pd.DataFrame(rows_partial, columns=columns).to_csv(csv_path, index=False)
    print(f"[zone_pathway_long] промежуточно записано (после прогона 1/2): {csv_path}")
    print_table(rows_partial, ["population", "n_neurons", "rate_stim_hz"])

    print(f"\n[zone_pathway_long] прогон 2/2: контроль без стимуляции (rates=0), "
          f"transient={LONG_TRANSIENT_MS:.0f} мс, measure={LONG_MEASURE_MS:.0f} мс, "
          f"seed={SEED} (свежий, не переиспользует контроль на 500 мс)...")
    t0 = time.time()
    rate_base = run_pathway(weights, n, pops, stim_idx, 0.0, SEED, device,
                            transient_ms=LONG_TRANSIENT_MS, measure_ms=LONG_MEASURE_MS)
    print(f"[zone_pathway_long] прогон 2/2 завершён за {time.time() - t0:.1f}с")

    rows = [{"population": name, "n_neurons": len(pops[name]),
             "rate_stim_hz": rate_stim[name], "rate_base_hz": rate_base[name]}
            for name in POP_ORDER]
    pd.DataFrame(rows, columns=columns).to_csv(csv_path, index=False)
    print(f"[zone_pathway_long] ИТОГ ЗАПИСАН: {csv_path}")

    print()
    print_table(rows, columns)

    print()
    r16_stim = rate_stim["R1-6"]
    if r16_stim > DECISIVE_THRESHOLD_HZ:
        print(f"[zone_pathway_long] R1-6 при стимуляции: {r16_stim:.3f} Гц — "
              "стимулируемые нейроны разряжаются, дефект в вызове TorchModel "
              "исключён на этом уровне.")
    else:
        print(f"[zone_pathway_long] R1-6 при стимуляции: {r16_stim:.3f} Гц — "
              "почти ноль. Стимуляция не доходит до самих стимулируемых "
              "нейронов: это дефект вызова exc_indices/rates, а не биология сети.")

    print(f"\n[zone_pathway_long] всего времени: {time.time() - t_start:.1f}с")


if __name__ == "__main__":
    if "--long" in sys.argv:
        run_long()
    else:
        main()
