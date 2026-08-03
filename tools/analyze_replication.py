"""Разбор серии повторностей: отличим ли зрительный эффект от разброса.

Зачем нужен отдельный разбор. Сравнение сцен по суммарному повороту даёт одно
число на прогон, и если курс мухи сам по себе гуляет от seed к seed, то эффект
в этом разбросе тонет. Первые же прогоны серии это показали: контроль БЕЗ
объекта дал +1.4, -92.3 и +21.6 градуса на трёх seed.

Поэтому считаем две вещи:

  1. Межпрогонное сравнение: поворот и асимметрия команды по сценам, с оценкой
     значимости. Это то, что было бы видно снаружи.

  2. Внутрипрогонную регрессию: предсказывает ли асимметрия темноты в цикле t
     скорость поворота в цикле t+1. Здесь сто точек на прогон, и постоянный
     дрейф конкретного прогона вычитается сам — метод намного чувствительнее.

Вторая метрика отвечает на вопрос «действует ли зрение на поведение», первая —
на вопрос «заметно ли это в итоговой траектории». Это разные вопросы.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flypaths import OUTPUT_DIR, out  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "main"
CONDITIONS = ["без_столба", "столб_слева", "столб_справа"]


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan"), float("nan")
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se2 = va / na + vb / nb
    if se2 <= 0:
        return float("nan"), float("nan")
    t = (a.mean() - b.mean()) / np.sqrt(se2)
    dof = se2 ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return float(t), float(dof)


def unwrap_deg(x):
    return np.degrees(np.unwrap(np.radians(np.asarray(x, dtype=float))))


def main():
    summ_path = OUTPUT_DIR / f"vision_replication_{TAG}.csv"
    if not summ_path.exists():
        print(f"нет файла: {summ_path}")
        return
    res = pd.read_csv(summ_path)
    print("=" * 78)
    print(f" РАЗБОР СЕРИИ: {len(res)} прогонов")
    print("=" * 78)

    # ---------- 1. межпрогонное ----------
    print("\n----- по сценам (среднее ± ст.откл по seed) -----")
    print(f"  {'сцена':<14s} {'n':>3s} {'поворот, град':>22s} {'асимметрия':>20s}")
    for c in res["condition"].unique():
        g = res[res["condition"] == c]
        print(f"  {c:<14s} {len(g):>3d} "
              f"{g['turn_deg'].mean():>12.1f} ± {g['turn_deg'].std():<7.1f} "
              f"{g['cmd_asym_mean'].mean():>10.3f} ± {g['cmd_asym_mean'].std():<7.3f}")

    ctrl_name = [c for c in res["condition"].unique() if "без" in c]
    if ctrl_name:
        ctrl = res[res["condition"] == ctrl_name[0]]
        print("\n----- сравнение с контролем (t-критерий Уэлча) -----")
        for c in res["condition"].unique():
            if c == ctrl_name[0]:
                continue
            g = res[res["condition"] == c]
            for metric in ("turn_deg", "cmd_asym_mean"):
                t, dof = welch(g[metric], ctrl[metric])
                d = g[metric].mean() - ctrl[metric].mean()
                verdict = ("значимо" if abs(t) > 2.5 else
                           "на грани" if abs(t) > 2.0 else "НЕ значимо")
                print(f"  {c:<14s} {metric:<15s} разница {d:+9.3f}  "
                      f"t={t:+6.2f} dof={dof:5.1f}  -> {verdict}")

    # ---------- 2. внутрипрогонное ----------
    print("\n" + "=" * 78)
    print(" ВНУТРИПРОГОННАЯ СВЯЗЬ: предсказывает ли зрение скорость поворота")
    print("=" * 78)
    print("  Для каждого прогона считаем корреляцию между асимметрией темноты")
    print("  в цикле t и приращением курса в цикле t+1. Постоянный дрейф")
    print("  прогона в приращениях сокращается.\n")

    rows = []
    for f in sorted(OUTPUT_DIR.glob(f"vision_rep_{TAG}_*.csv")):
        name = f.stem.replace(f"vision_rep_{TAG}_", "")
        cond = name.rsplit("_s", 1)[0]
        seed = int(name.rsplit("_s", 1)[1])
        df = pd.read_csv(f)
        if len(df) < 20:
            continue
        d_asym = (df["dark_left"] - df["dark_right"]).to_numpy()[:-1]
        heading = unwrap_deg(df["heading_deg"])
        dturn = np.diff(heading)
        if d_asym.std() < 1e-9 or dturn.std() < 1e-9:
            r = float("nan")
        else:
            r = float(np.corrcoef(d_asym, dturn)[0, 1])
        rows.append({"condition": cond, "seed": seed, "r_dark_vs_dturn": r,
                     "dark_asym_std": float(d_asym.std()),
                     "dturn_std": float(dturn.std()), "n": len(dturn)})

    if not rows:
        print("  нет пооконных логов прогонов")
        return
    ir = pd.DataFrame(rows)
    print(f"  {'сцена':<14s} {'n':>3s} {'r (темнота -> поворот)':>26s} "
          f"{'разброс темноты':>17s}")
    for c in sorted(ir["condition"].unique()):
        g = ir[ir["condition"] == c].dropna(subset=["r_dark_vs_dturn"])
        if not len(g):
            continue
        print(f"  {c:<14s} {len(g):>3d} "
              f"{g['r_dark_vs_dturn'].mean():>16.3f} ± {g['r_dark_vs_dturn'].std():<7.3f} "
              f"{g['dark_asym_std'].mean():>17.4f}")

    # проверка: отличается ли r от нуля в сценах со столбом
    print("\n  проверка отличия r от нуля (одновыборочный t):")
    for c in sorted(ir["condition"].unique()):
        g = ir[ir["condition"] == c].dropna(subset=["r_dark_vs_dturn"])
        if len(g) < 2:
            continue
        m, s, k = g["r_dark_vs_dturn"].mean(), g["r_dark_vs_dturn"].std(ddof=1), len(g)
        t = m / (s / np.sqrt(k)) if s > 0 else float("nan")
        verdict = ("значимо" if abs(t) > 2.5 else
                   "на грани" if abs(t) > 2.0 else "НЕ значимо")
        print(f"    {c:<14s} r={m:+.3f}  t={t:+6.2f}  n={k}  -> {verdict}")

    p = out(f"replication_analysis_{TAG}.csv")
    ir.to_csv(p, index=False)
    print(f"\nсохранено: {p}")
    print("\n  Как читать вместе. Если внутрипрогонная связь есть, а межпрогонная")
    print("  разница тонет — значит зрение на команду влияет, но итоговая")
    print("  траектория определяется в основном шумом, и заявлять «муха")
    print("  избегает препятствий» по одному прогону нельзя.")


if __name__ == "__main__":
    main()
