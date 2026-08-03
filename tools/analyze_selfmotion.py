"""Какова природа помехи от собственного движения мухи?

В контрольном прогоне без объекта «темнота» доходит до 0.25, хотя смотреть
не на что. Причина в том, что при ходьбе тело качается и меняется доля неба и
земли в поле зрения. Прежде чем выбирать фильтр, надо измерить характер помехи.

Гипотеза: помеха СИНФАЗНА для обоих глаз (качка поднимает и опускает оба глаза
разом), а объект действует на один глаз. Тогда её снимет переход на
бинокулярную разность вместо абсолютной яркости — классическое подавление
синфазной составляющей.

Проверяем по логам трёх прогонов:
  1. корреляцию яркости левого и правого глаза;
  2. во сколько раз синфазная составляющая больше разностной;
  3. что стало бы с сигналом, считай мы разность вместо абсолютной яркости.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flypaths import OUTPUT_DIR  # noqa: E402

RUNS = {
    "без столба": "closed_loop_vision_none.csv",
    "столб слева": "closed_loop_vision_left.csv",
    "столб справа": "closed_loop_vision_right.csv",
}


def main():
    print("=" * 78)
    print(" ПРИРОДА ПОМЕХИ ОТ СОБСТВЕННОГО ДВИЖЕНИЯ")
    print("=" * 78)

    summary = []
    for label, fname in RUNS.items():
        p = OUTPUT_DIR / fname
        if not p.exists():
            print(f"  нет файла: {p}")
            continue
        df = pd.read_csv(p)
        il, ir = df["eye_left_int"].to_numpy(), df["eye_right_int"].to_numpy()

        common = (il + ir) / 2.0          # синфазная составляющая
        diff = il - ir                     # разностная

        corr = float(np.corrcoef(il, ir)[0, 1])
        print(f"\n----- {label} -----")
        print(f"  циклов: {len(df)}")
        print(f"  яркость левого:  среднее {il.mean():.4f}, ст.откл {il.std():.4f}")
        print(f"  яркость правого: среднее {ir.mean():.4f}, ст.откл {ir.std():.4f}")
        print(f"  корреляция глаз: {corr:+.3f}")
        print(f"  синфазная составляющая: ст.откл {common.std():.4f}")
        print(f"  разностная составляющая: ст.откл {diff.std():.4f}")
        ratio = common.std() / max(diff.std(), 1e-9)
        print(f"  отношение синфазная/разностная: {ratio:.2f}")

        # период колебаний синфазной составляющей: где первый максимум
        # автокорреляции после нуля. Шаг между отсчётами — окно 15 мс.
        c = common - common.mean()
        ac = np.correlate(c, c, mode="full")[len(c) - 1:]
        ac = ac / max(ac[0], 1e-12)
        peak = None
        for k in range(2, min(len(ac) - 1, 40)):
            if ac[k] > ac[k - 1] and ac[k] >= ac[k + 1] and ac[k] > 0.2:
                peak = k
                break
        if peak:
            print(f"  период синфазных колебаний: {peak} циклов = "
                  f"{peak * 15} мс = {1000 / (peak * 15):.1f} Гц "
                  f"(частота CPG задана 12 Гц)")
        else:
            print("  выраженной периодичности синфазной составляющей не найдено")

        summary.append({"run": label, "corr_eyes": corr,
                        "std_common": common.std(), "std_diff": diff.std(),
                        "ratio": ratio,
                        "dark_left_max": df["dark_left"].max(),
                        "dark_right_max": df["dark_right"].max()})

    print("\n" + "=" * 78)
    print(" ИТОГ")
    print("=" * 78)
    s = pd.DataFrame(summary)
    print(s.to_string(index=False))

    if len(s) >= 2:
        ctrl = s[s["run"] == "без столба"]
        obj = s[s["run"] != "без столба"]
        if len(ctrl):
            print(f"\n  в контроле без объекта: разностная составляющая "
                  f"{ctrl['std_diff'].iloc[0]:.4f}")
            print(f"  при объекте в среднем:  разностная составляющая "
                  f"{obj['std_diff'].mean():.4f}")
            gain = obj["std_diff"].mean() / max(ctrl["std_diff"].iloc[0], 1e-9)
            print(f"  объект увеличивает разностную составляющую в {gain:.2f} раза")
            print("\n  Читать так: если синфазная составляющая заметно больше")
            print("  разностной, а объект поднимает именно разностную, то переход")
            print("  на бинокулярную разность подавит помеху и сохранит сигнал.")

    s.to_csv(OUTPUT_DIR / "selfmotion_analysis.csv", index=False)
    print(f"\nсохранено: {OUTPUT_DIR / 'selfmotion_analysis.csv'}")

    filter_sweep()


def ema(x, tau_ms, dt_ms=15.0):
    if tau_ms <= 0:
        return np.asarray(x, dtype=float)
    a = 1.0 - np.exp(-dt_ms / tau_ms)
    out = np.empty(len(x), dtype=float)
    acc = float(x[0])
    for i, v in enumerate(x):
        acc += a * (v - acc)
        out[i] = acc
    return out


def filter_sweep():
    """Подобрать постоянную низкочастотного фильтра по данным.

    Помеха от походки сидит на 22 Гц (период 45 мс), приближение к объекту —
    медленный спад яркости за сотни миллисекунд. Значит их можно разделить по
    времени. Меряем на реальных логах: во сколько раз фильтр давит помеху в
    контроле и сколько при этом теряет полезного сигнала при объекте.
    """
    print("\n" + "=" * 78)
    print(" ПОДБОР ПОСТОЯННОЙ ФИЛЬТРА ПО ДАННЫМ")
    print("=" * 78)

    ctrl_p = OUTPUT_DIR / RUNS["без столба"]
    obj_p = OUTPUT_DIR / RUNS["столб слева"]
    if not (ctrl_p.exists() and obj_p.exists()):
        print("  нет нужных логов")
        return

    ctrl = pd.read_csv(ctrl_p)
    obj = pd.read_csv(obj_p)

    print(f"  {'tau, мс':>8s} {'помеха (контроль)':>19s} {'сигнал (объект)':>17s} "
          f"{'сигнал/помеха':>15s}")
    rows = []
    for tau in (0.0, 30.0, 60.0, 100.0, 150.0, 250.0, 400.0, 700.0):
        # помеха: разброс яркости в контроле, где смотреть не на что
        n_l = ema(ctrl["eye_left_int"].to_numpy(), tau).std()
        n_r = ema(ctrl["eye_right_int"].to_numpy(), tau).std()
        noise = (n_l + n_r) / 2

        # сигнал: насколько глубоко просаживается яркость глаза при объекте
        f = ema(obj["eye_left_int"].to_numpy(), tau)
        signal = float(f.max() - f.min())

        snr = signal / max(noise, 1e-9)
        print(f"  {tau:>8.0f} {noise:>19.5f} {signal:>17.5f} {snr:>15.1f}")
        rows.append({"tau_ms": tau, "noise": noise, "signal": signal, "snr": snr})

    df = pd.DataFrame(rows)
    best = df.loc[df["snr"].idxmax()]
    print(f"\n  лучшее отношение при tau = {best['tau_ms']:.0f} мс "
          f"(сигнал/помеха {best['snr']:.1f})")
    print("  Оговорка: фильтр добавляет задержку порядка tau, поэтому брать")
    print("  максимум вслепую нельзя — нужен компромисс с быстротой реакции.")
    df.to_csv(OUTPUT_DIR / "eye_filter_sweep.csv", index=False)
    print(f"  сохранено: {OUTPUT_DIR / 'eye_filter_sweep.csv'}")


if __name__ == "__main__":
    main()
