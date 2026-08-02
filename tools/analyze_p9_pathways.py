"""Анализ коннектома вокруг нисходящих нейронов P9.

Зачем: измерение показало, что стимуляция Sugar GRNs не вызывает у P9 ни одного
спайка — причинного пути «сахар -> P9» в коннектоме, судя по всему, нет.
Прежде чем строить замкнутый контур, надо понять по данным:
  1. кто вообще входит в P9 напрямую (пресинаптические партнёры и их веса);
  2. достижим ли P9 из Sugar GRNs за несколько шагов и с какой силой;
  3. какие популяции — самые сильные драйверы P9.

Всё считается напрямую по 2025_Connectivity_783.parquet, без симуляции.
"""
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

FLY_BRAIN_CODE = "/mnt/d/временное использование федей/мозг мухи/fly-brain/code"
sys.path.insert(0, FLY_BRAIN_CODE)
from benchmark import EXPERIMENTS, path_comp, path_con  # noqa: E402

P9_LEFT = 720575940627652358
P9_RIGHT = 720575940635872101


def main():
    df_comp = pd.read_csv(path_comp, index_col=0)
    flyid2i = {j: i for i, j in enumerate(df_comp.index)}
    i2flyid = {i: j for j, i in flyid2i.items()}
    n = len(flyid2i)
    print(f"нейронов: {n}")

    conn = pd.read_parquet(path_con)
    print(f"связей: {len(conn)}")
    print(f"колонки: {list(conn.columns)}")

    pre = conn["Presynaptic_Index"].to_numpy()
    post = conn["Postsynaptic_Index"].to_numpy()
    w = conn["Excitatory x Connectivity"].to_numpy().astype(np.float32)

    print(f"\nвеса: min={w.min():.3f} max={w.max():.3f} "
          f"положительных={int((w > 0).sum())} отрицательных={int((w < 0).sum())}")

    # W[post, pre] — та же ориентация, что в модели
    W = sp.coo_matrix((w, (post, pre)), shape=(n, n)).tocsr()
    W_abs = sp.coo_matrix((np.abs(w), (post, pre)), shape=(n, n)).tocsr()

    idx_l, idx_r = flyid2i[P9_LEFT], flyid2i[P9_RIGHT]
    print(f"\nP9 left  index={idx_l}")
    print(f"P9 right index={idx_r}")

    # ---------- 1. прямые входы в P9 ----------
    for name, idx in (("P9 LEFT", idx_l), ("P9 RIGHT", idx_r)):
        row = W.getrow(idx).tocoo()
        order = np.argsort(-np.abs(row.data))
        print(f"\n===== прямые пресинаптические входы в {name} =====")
        print(f"  всего входов: {row.nnz}, суммарный |вес|: {np.abs(row.data).sum():.1f}, "
              f"алгебраическая сумма: {row.data.sum():.1f}")
        print(f"  {'flywire_id':>20s} {'вес':>10s}")
        for k in order[:15]:
            print(f"  {i2flyid[row.col[k]]:>20d} {row.data[k]:>10.3f}")

    # ---------- 2. выходы P9 ----------
    Wc = W.tocsc()
    for name, idx in (("P9 LEFT", idx_l), ("P9 RIGHT", idx_r)):
        col = Wc.getcol(idx).tocoo()
        print(f"\n===== выходы {name} =====")
        print(f"  постсинаптических целей: {col.nnz}, суммарный |вес|: {np.abs(col.data).sum():.1f}")

    # ---------- 3. достижимость P9 из Sugar GRNs ----------
    sugar_idx = [flyid2i[x] for x in EXPERIMENTS["sugar"]["neu_exc"] if x in flyid2i]
    print(f"\n===== достижимость P9 из Sugar GRNs ({len(sugar_idx)} нейронов) =====")

    # фронт по абсолютным весам: reach[k] = множество достигнутых за <=k шагов
    frontier = np.zeros(n, dtype=bool)
    frontier[sugar_idx] = True
    reached = frontier.copy()
    A = (W_abs > 0)  # булева матрица смежности [post, pre]
    for hop in range(1, 7):
        # кто получает вход от текущего фронта
        nxt = (A @ frontier.astype(np.float32)) > 0
        new = nxt & ~reached
        reached |= nxt
        hit_l = "ДА" if reached[idx_l] else "нет"
        hit_r = "ДА" if reached[idx_r] else "нет"
        print(f"  шаг {hop}: достигнуто {int(reached.sum()):>6d} нейронов "
              f"(+{int(new.sum()):>6d}) | P9 L: {hit_l}, P9 R: {hit_r}")
        if reached[idx_l] and reached[idx_r]:
            break
        frontier = new
        if new.sum() == 0:
            print("  фронт исчерпан")
            break

    # ---------- 4. сильнейшие драйверы P9 за 2 шага ----------
    print("\n===== самые сильные источники влияния на P9 (1 и 2 шага, по |весу|) =====")
    for name, idx in (("P9 LEFT", idx_l), ("P9 RIGHT", idx_r)):
        e = np.zeros(n, dtype=np.float32)
        e[idx] = 1.0
        one = W_abs.T @ e          # влияние 1 шаг назад
        two = W_abs.T @ one        # 2 шага назад
        top = np.argsort(-two)[:10]
        print(f"\n  {name}: топ-10 источников за 2 шага")
        print(f"  {'flywire_id':>20s} {'путевой вес':>14s} {'прямой вес':>12s}")
        for k in top:
            print(f"  {i2flyid[int(k)]:>20d} {two[k]:>14.2f} {one[k]:>12.3f}")

    # пересечение с сахарными
    e = np.zeros(n, dtype=np.float32)
    e[[idx_l, idx_r]] = 1.0
    infl = e.copy()
    for hop in range(1, 5):
        infl = W_abs.T @ infl
        s = infl[sugar_idx].sum()
        print(f"\n  суммарное влияние Sugar GRNs на P9 через {hop} шаг(ов): {s:.6f}")


if __name__ == "__main__":
    main()
