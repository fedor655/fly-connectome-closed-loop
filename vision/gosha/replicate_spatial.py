"""Этап 2: различает ли муха участки поля зрения на поведении.

Пять сцен на два режима входа по N seed = 5*2*N прогонов. Скалярный вход — это
не запасной путь, а обязательный контроль ценности карты: если он двигает муху
не хуже пространственного, карта ничего не добавила, и это записывается прямым
текстом.

Почему сцен пять, а не три. Задача 4 измерила: столб в (0, 6) виден ТОЛЬКО
левому глазу, правый даёт ровно 0. Значит пара «впереди (10,0) против сбоку
(0,10)» (пара 1) разделяет сцены в основном по тому, какой глаз видит объект —
а это латеральное разделение работает и на скалярном входе, оно не проверяет
ценность карты внутри глаза. Пара 2 («спереди-слева» против «сбоку-слева»,
A2/B2) держит примерно одинаковое затемнение ЛЕВОГО глаза в обеих сценах и
меняет только то, в какую полосу поля зрения попадает объект — скалярный вход
принципиально не может их различить, потому что видит только одно число на
глаз. Это и есть прямая проверка ценности карты; пара 1 — вспомогательная.

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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from flypaths import out  # noqa: E402
from closed_loop_vision import load_brain_assets, run_trial  # noqa: E402
from tools.replicate_vision import welch  # noqa: E402
from vision.gosha.visual_field_map import load_or_build  # noqa: E402

# Ключ (латиницей) идёт в имена файлов, подпись — в вывод. Сцены оси azimuth
# образуют ДВЕ пары плюс общий контроль:
#   пара 1 (A1/B1), вспомогательная: впереди против сбоку. Разделение здесь
#     достижимо и скалярным входом, потому что «сбоку» видит один глаз, а
#     «впереди» — оба; это в основном проверка «какой глаз видит», не карты.
#   пара 2 (A2/B2), главная: обе сцены темнят ЛЕВЫЙ глаз примерно одинаково
#     (столб в обоих случаях виден преимущественно левому глазу), но попадают
#     в разные полосы поля зрения. Скалярный вход их различить не может в
#     принципе — это и есть проверка ценности карты.
# Сцены оси elevation оставлены как запасной вариант (не используется, пока
# split-half контроль азимута проходит порог), тройка там не менялась.
SCENES = {
    "azimuth": [
        # Пара 1, бинокулярная: объект впереди против объекта сбоку. Разделение
        # здесь достижимо и скалярным входом, потому что «сбоку» видит один глаз.
        ("A1", "A1: впереди", dict(no_pillar=False, pillar_x=10.0, pillar_y=0.0)),
        ("B1", "B1: сбоку слева", dict(no_pillar=False, pillar_x=0.0, pillar_y=10.0)),
        # Пара 2, ипсилатеральная: обе сцены темнят ЛЕВЫЙ глаз примерно одинаково,
        # но попадают в разные полосы. Скалярный вход их различить не может в
        # принципе — это прямая проверка ценности карты.
        ("A2", "A2: спереди-слева", dict(no_pillar=False, pillar_x=10.0, pillar_y=2.0)),
        ("B2", "B2: сбоку-слева", dict(no_pillar=False, pillar_x=2.0, pillar_y=10.0)),
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

# Пары для разбора критериев 1 и 2. Подпись используется в печати; пара 2 —
# главный результат этапа, пара 1 — вспомогательный (см. докстринг модуля).
PAIRS = [
    ("A1", "B1", "пара 1 (впереди/сбоку, вспомогательная)"),
    ("A2", "B2", "пара 2 (спереди-слева/сбоку-слева, ГЛАВНАЯ)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--cycles", type=int, default=100)
    ap.add_argument("--tag", type=str, default="spatial")
    # Укороченная калибровка мозга — только для быстрой пробы серии на CPU.
    # Научной ценности прогон с уменьшенным --cal-brain-ms не имеет: норма
    # команды (ref_l/ref_r) оценивается по меньшему числу спайков и шумнее.
    ap.add_argument("--cal-brain-ms", type=float, default=3000.0)
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
                                  spatial=spatial, cal_brain_ms=args.cal_brain_ms,
                                  verbose=False, **kw)
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
    print(" КРИТЕРИЙ 1: A против Б по КАЖДОЙ паре, и есть ли разделение у "
          "скалярного контроля")
    print("=" * 78)
    ok1 = {}
    for key_a, key_b, pair_label in PAIRS:
        print(f"\n  {pair_label}")
        verdict = {}
        for inp in ("пространственный", "скалярный"):
            a = res[(res["scene"] == key_a) & (res["input"] == inp)]["turn_deg"]
            b = res[(res["scene"] == key_b) & (res["input"] == inp)]["turn_deg"]
            t, dof = welch(a, b)
            verdict[inp] = abs(t)
            print(f"    {inp:<18s} {a.mean():+7.1f} против {b.mean():+7.1f}  "
                  f"t={t:+6.2f}  dof={dof:5.1f}  -> "
                  f"{'разделяет' if abs(t) > 3 else 'не разделяет'}")
        ok = verdict["пространственный"] > 3 and verdict["скалярный"] <= 3
        ok1[(key_a, key_b)] = ok
        print(f"    критерий 1 для {pair_label} -> {'ПРОЙДЕН' if ok else 'НЕ ПРОЙДЕН'}")
        if verdict["скалярный"] > 3:
            print("    Скалярный вход разделяет сцены не хуже: карта ничего не добавила.")

    print(f"\nсохранено: {out(f'{args.tag}_replication.csv')}")

    print("\n" + "=" * 78)
    print(" КРИТЕРИЙ 2: разные ли МНОЖЕСТВА проекционных нейронов, по КАЖДОЙ паре")
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

    ok2 = {}
    for key_a, key_b, pair_label in PAIRS:
        jac = []
        for seed in range(args.seeds):
            sa, sb = lit_set(key_a, seed), lit_set(key_b, seed)
            inter, union = len(sa & sb), len(sa | sb)
            jac.append(inter / max(union, 1))
        jac = np.array(jac)
        print(f"\n  {pair_label}")
        print(f"    Жаккар {key_a} против {key_b} по {args.seeds} seed: "
              f"{jac.mean():.3f} ± {jac.std():.3f}, максимум {jac.max():.3f}")
        ok = bool(jac.mean() < 0.9)
        ok2[(key_a, key_b)] = ok
        print(f"    критерий 2 для {pair_label} -> {'ПРОЙДЕН' if ok else 'НЕ ПРОЙДЕН'}")

    # Обратная причинность. Связь «темнота -> поворот» значима ТОЛЬКО в
    # контроле без объекта: муха кренится при повороте, крен меняет яркость.
    # Признак артефакта — периодика по сдвигам вместо одного максимума.
    # Считается по ВСЕМ пяти сценам, как и было заявлено до прогона.
    print("\n" + "=" * 78)
    print(" КОНТРОЛЬ ОБРАТНОЙ ПРИЧИННОСТИ: профиль по сдвигам (все сцены)")
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

    print("\n" + "=" * 78)
    print(" ИТОГ ЭТАПА 2")
    print("=" * 78)
    main_ok = ok1[("A2", "B2")] and ok2[("A2", "B2")]
    aux_ok = ok1[("A1", "B1")] and ok2[("A1", "B1")]
    print(f"  ГЛАВНЫЙ результат (пара 2, спереди-слева/сбоку-слева): "
          f"{'ПРОЙДЕН' if main_ok else 'НЕ ПРОЙДЕН'}")
    print(f"  вспомогательный результат (пара 1, впереди/сбоку): "
          f"{'ПРОЙДЕН' if aux_ok else 'НЕ ПРОЙДЕН'}")


if __name__ == "__main__":
    main()
