"""Какая популяция нисходящих отзывается, когда вход приходит на стык с flyvis?

Зачем отдельный отбор. Рабочая популяция из 230 латерализованных нисходящих
отбиралась стимуляцией зрительных проекционных напрямую (150 Гц на все 8038 LC
разом). В гибриде вход приходит не туда: flyvis считает оптические доли, а его
выход ложится на T4/T5/Tm/TmY, и уже они через лобулу зажигают LC сами. Драйв
другой по адресу и на порядок слабее по силе. Значит и популяция, отобранная под
старый драйв, может оказаться не той: отбирать надо под тот вход, который будет
в контуре. Правило тут ровно то же, из-за которого популяцию вообще отбирают
стимуляцией, а не по анатомическим весам.

Что меряем. Стимулируем стык левого полушария, потом правого, на физиологической
для T4/T5 частоте, и смотрим отклик всех 1299 нисходящих. Дальше три вопроса:

  1. Сколько нисходящих отзывается и сколько из них латерализованы.
  2. Сохраняет ли латеральность СТАРАЯ популяция под новым драйвом — то есть
     можно ли её переиспользовать без перевыбора.
  3. Хватает ли спайков за окно 15 мс, чтобы шум считывания остался в единицах
     процентов, а не в десятках. Именно на этом развалилось поведение в прошлый
     раз: у одиночного DNp09 шум был 62 процента.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flypaths import ANNOTATIONS, add_fly_brain_to_path, out  # noqa: E402

add_fly_brain_to_path()
from benchmark import path_comp, path_con, path_wt  # noqa: E402
import run_pytorch as rp  # noqa: E402

DT = 0.1
SIM_MS = 2000.0
TRANSIENT_MS = 300.0
# Первый заход мерил только 50 Гц — физиологический потолок T4/T5 — и дал шум
# считывания 23.6 процента при допустимых 15. Значит точку работы надо не
# назначать, а выбирать: активность flyvis безразмерная, и её перевод в герцы
# всё равно свободный параметр. Разворачиваем его по частоте и смотрим, где
# шум опускается до рабочего, а где частота перестаёт быть правдоподобной.
STIM_RATES_HZ = [50.0, 100.0, 200.0]
WINDOW_S = 0.015          # окно синхронизации мозг-тело в контуре
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FLYVIS_TYPES = [
    "R7", "R8",
    "L1", "L2", "L3", "L4", "L5", "Lawf1", "Lawf2", "Am", "C2", "C3",
    "Mi1", "Mi2", "Mi4", "Mi9", "Mi10", "Mi13", "Mi14", "Mi15",
    "T1", "T2", "T2a", "T3",
    "T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d",
    "Tm1", "Tm2", "Tm3", "Tm4", "Tm5a", "Tm5b", "Tm5c", "Tm9", "Tm16", "Tm20",
    "TmY3", "TmY4", "TmY5a", "TmY9", "TmY10", "TmY14", "TmY15", "TmY18",
]


def main():
    print("=" * 78)
    print(" ОТБОР НИСХОДЯЩИХ ПОД ДРАЙВ ЧЕРЕЗ СТЫК С FLYVIS")
    print("=" * 78)

    comp = pd.read_csv(path_comp, index_col=0)
    flyid2i = {int(j): i for i, j in enumerate(comp.index)}
    n = len(flyid2i)

    ann = pd.read_csv(ANNOTATIONS, sep="\t", low_memory=False)
    ann["root_id"] = pd.to_numeric(ann["root_id"], errors="coerce")
    ann = ann.dropna(subset=["root_id"])
    ann["root_id"] = ann["root_id"].astype("int64")
    ann = ann[ann["root_id"].isin(flyid2i.keys())]
    ct = ann["cell_type"].fillna("")

    seam = ct.isin(FLYVIS_TYPES)
    seam_l = [flyid2i[int(x)] for x in ann.loc[seam & (ann["side"] == "left"), "root_id"]]
    seam_r = [flyid2i[int(x)] for x in ann.loc[seam & (ann["side"] == "right"), "root_id"]]
    print(f"стык: слева {len(seam_l)}, справа {len(seam_r)} нейронов")

    desc = ann[(ann["super_class"] == "descending") & ann["side"].isin(["left", "right"])].copy()
    dn_ids = desc["root_id"].tolist()
    dn_idx = [flyid2i[int(x)] for x in dn_ids]
    dn_side = np.array(desc["side"].tolist())
    dn_type = desc["cell_type"].fillna("(без типа)").tolist()
    print(f"нисходящих с известной стороной: {len(dn_idx)}")

    weights = rp.get_weights(str(path_con), str(path_comp), str(path_wt), csr=True).to(DEVICE)
    idx_t = torch.tensor(dn_idx, dtype=torch.long, device=DEVICE)

    def probe(stim, hz):
        m = rp.TorchModel(1, n, DT, rp.MODEL_PARAMS, weights,
                          exc_indices=list(stim), device=DEVICE)
        c, d, s, v, r = m.state_init()
        rates = torch.zeros(1, n, device=DEVICE)
        rates[:, stim] = hz
        g = torch.Generator(device=DEVICE)
        g.manual_seed(2024)
        acc = torch.zeros(len(dn_idx), device=DEVICE)
        n_tr, n_me = int(TRANSIENT_MS / DT), int(SIM_MS / DT)
        with torch.no_grad():
            for step in range(n_tr + n_me):
                c, d, s, v, r = m(rates, c, d, s, v, r, generator=g)
                if step >= n_tr:
                    acc.add_(s[0, idx_t])
        return np.array(acc.tolist()) / (SIM_MS / 1000.0)

    # ---------- развёртка по частоте стыка ----------
    print("\n" + "=" * 78)
    print(" РАЗВЁРТКА: где точка работы")
    print("=" * 78)
    print(f"  {'стык,Гц':>8s} {'отозв.':>7s} {'лат.>0.5':>9s} {'слева':>6s} {'справа':>7s} "
          f"{'спайков/окно L':>15s} {'шум L':>7s} {'спайков/окно R':>15s} {'шум R':>7s}")
    sweep, per_rate = [], {}
    for hz in STIM_RATES_HZ:
        t0 = time.perf_counter()
        a = probe(seam_l, hz)
        b = probe(seam_r, hz)
        ip = np.where(dn_side == "left", a, b)
        co = np.where(dn_side == "left", b, a)
        tt = ip + co
        lt = np.where(tt > 0, (ip - co) / np.maximum(tt, 1e-9), 0.0)
        keep = (ip > 1.0) & (lt > 0.5)
        row = {"stim_hz": hz, "n_responding": int((ip > 1.0).sum()),
               "n_selected": int(keep.sum())}
        cells = []
        for side in ("left", "right"):
            m_ = keep & (dn_side == side)
            k = float(ip[m_].sum()) * WINDOW_S
            rel = 1.0 / np.sqrt(max(k, 1e-9))
            row[f"n_{side}"] = int(m_.sum())
            row[f"spikes_window_{side}"] = k
            row[f"noise_{side}"] = rel
            cells += [int(m_.sum()), k, rel]
        print(f"  {hz:>8.0f} {row['n_responding']:>7d} {row['n_selected']:>9d} "
              f"{cells[0]:>6d} {cells[3]:>7d} {cells[1]:>15.1f} {cells[2]:>7.1%} "
              f"{cells[4]:>15.1f} {cells[5]:>7.1%}   [{time.perf_counter() - t0:.0f} с]")
        sweep.append(row)
        per_rate[hz] = (a, b, ip, co, lt, keep)
    pd.DataFrame(sweep).to_csv(out("dn_population_seam_sweep.csv"), index=False)

    # Дальше подробности для той частоты, где шум впервые опускается ниже 15%,
    # а если нигде — для самой высокой из измеренных.
    ok = [r for r in sweep if max(r["noise_left"], r["noise_right"]) < 0.15]
    STIM_HZ = ok[0]["stim_hz"] if ok else STIM_RATES_HZ[-1]
    print(f"\n  разбираю подробно частоту {STIM_HZ:.0f} Гц"
          + ("" if ok else " — ни одна не дала шум ниже 15%, беру верхнюю"))
    hz_l, hz_r, ipsi, contra, lat, _ = per_rate[STIM_HZ]

    df = pd.DataFrame({"root_id": dn_ids, "cell_type": dn_type, "side": dn_side,
                       "hz_stim_left": hz_l, "hz_stim_right": hz_r,
                       "hz_ipsi": ipsi, "hz_contra": contra, "lat_index": lat})

    # ---------- 1. кто отзывается ----------
    # Порог 5 Гц взят от старого отбора, где драйв был вдесятеро сильнее.
    # Показываем несколько порогов, чтобы решение не зависело от одного числа.
    print("\n----- сколько отзывается при разных порогах -----")
    print(f"  {'порог, Гц':>10s} {'отозвались':>12s} {'из них лат.>0.5':>17s} "
          f"{'слева':>7s} {'справа':>7s}")
    for thr in (0.5, 1.0, 2.0, 5.0):
        a = df[df["hz_ipsi"] > thr]
        g = a[a["lat_index"] > 0.5]
        print(f"  {thr:>10.1f} {len(a):>12d} {len(g):>17d} "
              f"{int((g['side'] == 'left').sum()):>7d} {int((g['side'] == 'right').sum()):>7d}")

    # ---------- 2. держится ли СТАРАЯ популяция ----------
    cache = Path(out("dn_population_cache.npz"))
    print("\n----- старая популяция (отобрана стимуляцией LC напрямую) -----")
    if cache.exists():
        z = np.load(cache)
        old = set(z["left"].tolist()) | set(z["right"].tolist())
        in_old = np.array([i in old for i in dn_idx])
        sub = df[in_old]
        print(f"  в старой популяции: {len(sub)} нисходящих")
        print(f"  своя сторона  {sub['hz_ipsi'].mean():>7.2f} Гц (медиана {sub['hz_ipsi'].median():.2f})")
        print(f"  чужая сторона {sub['hz_contra'].mean():>7.2f} Гц (медиана {sub['hz_contra'].median():.2f})")
        print(f"  индекс латеральности: медиана {sub['lat_index'].median():.2f}, "
              f"доля с индексом > 0.5: {(sub['lat_index'] > 0.5).mean():.1%}")
        for side in ("left", "right"):
            g = sub[sub["side"] == side]
            k = g["hz_ipsi"].sum() * WINDOW_S
            rel = 1.0 / np.sqrt(max(k, 1e-9))
            print(f"    {side:>6s}: нейронов {len(g):>3d}, спайков за окно {k:>6.1f}, "
                  f"шум {rel:>6.1%}")
    else:
        print("  кэша нет, сравнивать не с чем")

    # ---------- 3. новая популяция под новый драйв ----------
    print("\n----- новая популяция (отбор под драйв через стык) -----")
    good = df[(df["hz_ipsi"] > 1.0) & (df["lat_index"] > 0.5)].sort_values(
        "hz_ipsi", ascending=False)
    print(f"  отобрано {len(good)}: слева {int((good['side'] == 'left').sum())}, "
          f"справа {int((good['side'] == 'right').sum())}")
    for side in ("left", "right"):
        g = good[good["side"] == side]
        k = g["hz_ipsi"].sum() * WINDOW_S
        rel = 1.0 / np.sqrt(max(k, 1e-9))
        print(f"    {side:>6s}: нейронов {len(g):>3d}, спайков за окно {k:>6.1f}, "
              f"шум {rel:>6.1%}")

    print(f"\n  топ-20 по частоте:")
    print(f"  {'cell_type':<16s} {'сторона':>8s} {'своя,Гц':>9s} {'чужая,Гц':>9s} {'индекс':>8s}")
    for _, r in good.head(20).iterrows():
        print(f"  {str(r['cell_type'])[:16]:<16s} {r['side']:>8s} "
              f"{r['hz_ipsi']:>9.1f} {r['hz_contra']:>9.1f} {r['lat_index']:>8.2f}")

    df.to_csv(out("dn_population_seam.csv"), index=False)
    good.to_csv(out("dn_population_seam_selected.csv"), index=False)
    arr = np.array(dn_idx)
    keep = ((df["hz_ipsi"] > 1.0) & (df["lat_index"] > 0.5)).to_numpy()
    np.savez(out("dn_population_seam_cache.npz"),
             left=arr[keep & (dn_side == "left")],
             right=arr[keep & (dn_side == "right")])
    print(f"\nсохранено: {out('dn_population_seam.csv')}")
    print(f"сохранено: {out('dn_population_seam_cache.npz')}")

    print("\n" + "=" * 78)
    print(" КРИТЕРИЙ ПРИЁМКИ")
    print("=" * 78)
    print("  Команду можно считывать, если шум за окно 15 мс остаётся ниже 15")
    print("  процентов. При 62 процентах поведение разваливалось, при 8.6 —")
    print("  держалось. Если шум велик, окно придётся удлинить или популяцию")
    print("  расширить, и это надо решить ДО прогонов, а не после.")
    print(f"\n  выбранная точка работы: {STIM_HZ:.0f} Гц на стыке.")


if __name__ == "__main__":
    main()
