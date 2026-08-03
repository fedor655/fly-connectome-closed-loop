"""Повторность: то же самое на нескольких seed, чтобы отличить эффект от случая.

Зачем. Избегание препятствия было измерено по ОДНОМУ прогону каждой сцены.
Разница выглядела крупной (-93.7 против +21.3 градуса), но один прогон не
позволяет сказать, эффект это или разброс. Здесь три сцены гоняются на
нескольких независимых seed, и разница проверяется по распределениям.

Seed задаёт и пуассоновский вход мозга, и начальные фазы CPG, то есть каждый
прогон — независимая реализация.

Веса коннектома грузятся один раз на всю серию: чтение parquet и построение
разреженной матрицы занимает около минуты, повторять это 15 раз бессмысленно.
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

CONDITIONS = [
    ("без столба", dict(no_pillar=True)),
    ("столб слева", dict(no_pillar=False, pillar_y=+3.0)),
    ("столб справа", dict(no_pillar=False, pillar_y=-3.0)),
]


def welch(a, b):
    """t-критерий Уэлча без scipy.stats: возвращает t и число степеней свободы."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se2 = va / na + vb / nb
    if se2 <= 0:
        return float("nan"), float("nan")
    t = (a.mean() - b.mean()) / np.sqrt(se2)
    dof = se2 ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return float(t), float(dof)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--cycles", type=int, default=100)
    ap.add_argument("--tag", type=str, default="rep")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 78)
    print(" ПОВТОРНОСТЬ ЗРИТЕЛЬНОГО КОНТУРА")
    print("=" * 78)
    print(f"device={device}, сцен {len(CONDITIONS)}, seed на сцену {args.seeds}, "
          f"циклов {args.cycles}")
    print(f"всего прогонов: {len(CONDITIONS) * args.seeds}")

    assets = load_brain_assets(device)

    rows = []
    total = len(CONDITIONS) * args.seeds
    done = 0
    t_all = time.perf_counter()

    for label, kw in CONDITIONS:
        for seed in range(args.seeds):
            t0 = time.perf_counter()
            df, s = run_trial(assets, device, cycles=args.cycles, seed=seed,
                              verbose=False, **kw)
            done += 1
            s["condition"] = label
            rows.append(s)
            print(f"  [{done:2d}/{total}] {label:<14s} seed={seed}  "
                  f"поворот {s['turn_deg']:+7.1f}°  путь {s['path_mm']:5.2f} мм  "
                  f"асимметрия {s['cmd_asym_mean']:+.3f}  "
                  f"[{time.perf_counter() - t0:.0f} с]")
            pd.DataFrame(rows).to_csv(out(f"vision_replication_{args.tag}.csv"),
                                      index=False)
            df.round(4).to_csv(
                out(f"vision_rep_{args.tag}_{label.replace(' ', '_')}_s{seed}.csv"),
                index=False)

    print(f"\nвсего {time.perf_counter() - t_all:.0f} с")

    res = pd.DataFrame(rows)
    print("\n" + "=" * 78)
    print(" СВОДКА (среднее +- стандартное отклонение по seed)")
    print("=" * 78)
    print(f"  {'сцена':<14s} {'поворот, град':>20s} {'асимметрия команды':>22s} "
          f"{'путь, мм':>16s}")
    for label, _ in CONDITIONS:
        g = res[res["condition"] == label]
        print(f"  {label:<14s} "
              f"{g['turn_deg'].mean():>10.1f} ± {g['turn_deg'].std():<7.1f} "
              f"{g['cmd_asym_mean'].mean():>12.3f} ± {g['cmd_asym_mean'].std():<7.3f} "
              f"{g['path_mm'].mean():>8.2f} ± {g['path_mm'].std():<6.2f}")

    print("\n" + "=" * 78)
    print(" ПЕРЕДАЧА ЗРЕНИЯ В КОМАНДУ ВНУТРИ ПРОГОНА")
    print("=" * 78)
    print("  корреляция асимметрии темноты с асимметрией команды,")
    print("  по 100 точкам каждого прогона:")
    print(f"  {'сцена':<14s} {'без сдвига':>18s} {'сдвиг на цикл':>18s}")
    for label, _ in CONDITIONS:
        g = res[res["condition"] == label]
        print(f"  {label:<14s} "
              f"{g['corr_dark_cmd'].mean():>10.3f} ± {g['corr_dark_cmd'].std():<6.3f} "
              f"{g['corr_dark_cmd_lag1'].mean():>10.3f} ± {g['corr_dark_cmd_lag1'].std():<6.3f}")

    print("\n" + "=" * 78)
    print(" СРАВНЕНИЕ С КОНТРОЛЕМ (t-критерий Уэлча)")
    print("=" * 78)
    ctrl = res[res["condition"] == "без столба"]
    for label, _ in CONDITIONS[1:]:
        g = res[res["condition"] == label]
        for metric in ("turn_deg", "cmd_asym_mean"):
            t, dof = welch(g[metric], ctrl[metric])
            diff = g[metric].mean() - ctrl[metric].mean()
            verdict = "значимо" if abs(t) > 2.5 else ("на грани" if abs(t) > 2.0 else "нет")
            print(f"  {label:<14s} {metric:<16s} разница {diff:+9.3f}  "
                  f"t={t:+6.2f}  dof={dof:5.1f}  -> {verdict}")

    # прямое сравнение левой и правой сцены: знак поворота должен различаться
    left = res[res["condition"] == "столб слева"]
    right = res[res["condition"] == "столб справа"]
    t, dof = welch(left["turn_deg"], right["turn_deg"])
    print(f"\n  слева против справа: поворот "
          f"{left['turn_deg'].mean():+.1f} против {right['turn_deg'].mean():+.1f}, "
          f"t={t:+.2f}, dof={dof:.1f}")
    print("\n  |t| > 2.5 при таком числе степеней свободы примерно соответствует")
    print("  p < 0.05. Это грубая оценка, а не строгий вывод.")

    p = out(f"vision_replication_{args.tag}.csv")
    print(f"\nсохранено: {p}")


if __name__ == "__main__":
    main()
