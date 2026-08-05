"""Какая колонка flyvis какому омматидию flygym соответствует.

Первый заход искал ответ перебором симметрий шестиугольника и провалился ровно
так, как предсказывал его же второй критерий приёмки: все двенадцать вариантов
дали остаток 0.01 шага, отрыв победителя ровно 1.00. Правильный шестиугольник
переходит в себя при любой из двенадцати симметрий, поэтому геометрией решётки
ориентацию определить нельзя в принципе. Хорошо, что критерий это поймал, а не
мы поверили первому месту.

Ориентация задаётся не решёткой, а привязкой к осям изображения, и она есть у
обеих сторон.

  flyvis. В `hex_center_coordinates` колонки раскладываются по монитору как
  x = шаг * v и y = шаг * (u + v/2), где y — строка изображения, растущая вниз.
  То есть колонка с координатами (u, v) сидит в определённом месте картинки, и
  это место известно из кода. Ниже это ещё и проверяется пятном: в чёрную
  картинку ставится белый квадрат в известном углу, прогоняется через
  собственный рендерер flyvis, и смотрится, та ли колонка загорелась.

  flygym. Карта `ommatidia_id_map` прямо говорит, какие пиксели изображения
  глаза принадлежат какому омматидию. Это и есть привязка, проверять нечего.

Дальше сопоставление однозначно: центр омматидия в изображении глаза
сопоставляется центру колонки в изображении монитора, после приведения обоих к
общему масштабу.

Три оговорки, которые надо помнить дальше.

  1. Поле зрения. У flygym 157 градусов на глаз, у flyvis 31 омматидий по 5.8,
     то есть 179.8. Решётки совпадают по числу и форме, но угловой масштаб
     отличается в 1.15 раза: flyvis будет считать сцену на 15 процентов шире,
     чем она есть. На порядок колонок это не влияет, на абсолютные скорости
     движения — влияет.

  2. Дисторсия. У flygym в изображении рыбий глаз (коэффициент 3.8, зум 2.72).
     Она радиальная и монотонная, порядок омматидиев не меняет, но расстояния в
     изображении не пропорциональны углам.

  3. Крен камеры. Камеры глаз в flygym повёрнуты (`orientation: [1.57, 0, -0.47]`
     слева и зеркально справа), поэтому горизонталь изображения — не в точности
     азимут мухи. flyvis при этом получает правильно ориентированную картинку
     того, что видит глаз, и обе стороны повёрнуты зеркально одинаково.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from flypaths import out  # noqa: E402

HEX_EXTENT = 15
N_COLUMNS = 721


def flygym_centers():
    """Центры омматидиев flygym в координатах изображения глаза (x вправо, y вниз)."""
    from flygym.vision.retina import Retina
    r = Retina()
    m = r.ommatidia_id_map
    n = r.num_ommatidia_per_eye
    xy = np.zeros((n, 2))
    for i in range(1, n + 1):
        yy, xx = np.nonzero(m == i)
        xy[i - 1] = (xx.mean(), yy.mean())
    return xy, r


PPO = 13   # пикселей на колонку: 25 по умолчанию дают монитор 775x775 и
           # промежуточный тензор на 866 МБ, а нам хватит вчетверо меньшего


def flyvis_centers(monitor_px):
    """Центры колонок flyvis в координатах монитора (x вправо, y вниз) и их (u, v)."""
    from flyvis.utils.hex_utils import get_hex_coords
    from flyvis.datasets.rendering.utils import hex_center_coordinates
    u, v = get_hex_coords(HEX_EXTENT)
    xs, ys, _ = hex_center_coordinates(N_COLUMNS, monitor_px, monitor_px)
    return np.stack([xs, ys], 1), np.asarray(u), np.asarray(v)


def norm(A):
    A = A - A.mean(0)
    return A / np.sqrt((A ** 2).sum(1).mean())


def make_eye():
    import torch
    import flyvis
    from flyvis.datasets.rendering.eye import HexEye
    return HexEye(n_ommatidia=N_COLUMNS, ppo=PPO,
                  device=flyvis.device, dtype=torch.float16), torch, flyvis


def check_flyvis_anchor(eye, torch, flyvis, centers, u, v):
    """Пятно в известном месте картинки — та ли колонка загорается."""
    H, W = eye.monitor_height_px, eye.monitor_width_px
    print(f"\n  монитор рендерера flyvis: {W} x {H} пикселей, "
          f"{PPO} пикселей на колонку")
    spots = {"верх-лево": (0.25, 0.25), "верх-право": (0.75, 0.25),
             "низ-лево": (0.25, 0.75), "низ-право": (0.75, 0.75)}
    print(f"  {'пятно':<12s} {'ставлю в':>16s} {'загорелась колонка':>22s} "
          f"{'её место':>16s} {'промах, пикс':>13s}")
    errs = []
    half = max(PPO // 2, 3)
    for name, (fx, fy) in spots.items():
        img = torch.zeros((1, H, W), dtype=torch.float16, device=flyvis.device)
        cx, cy = int(fx * W), int(fy * H)
        img[0, cy - half:cy + half, cx - half:cx + half] = 1.0
        vals = eye(img, mode="mean")[0].float().cpu().numpy()
        j = int(np.argmax(vals))
        px, py = centers[j]
        err = float(np.hypot(px - cx, py - cy))
        errs.append(err)
        print(f"  {name:<12s} {f'({cx}, {cy})':>16s} "
              f"{f'u={u[j]}, v={v[j]}':>22s} {f'({px:.0f}, {py:.0f})':>16s} "
              f"{err:>13.1f}")
    return float(np.max(errs))


def main():
    print("=" * 78)
    print(" СОВМЕЩЕНИЕ РЕШЁТОК: 721 ОММАТИДИЙ FLYGYM ↔ 721 КОЛОНКА FLYVIS")
    print("=" * 78)

    P, retina = flygym_centers()
    eye, torch, flyvis = make_eye()
    Q, u, v = flyvis_centers(eye.monitor_width_px)
    print(f"\nомматидиев flygym: {len(P)}; колонок flyvis: {len(Q)}")

    print("\n----- проверка привязки flyvis пятном -----")
    max_err = check_flyvis_anchor(eye, torch, flyvis, Q, u, v)
    ppo = PPO
    print(f"\n  наибольший промах {max_err:.1f} пикселя при размере колонки "
          f"{ppo} пикселей = {max_err / ppo:.2f} колонки")

    print("\n----- сопоставление по индексам решётки -----")
    # Сопоставлять расстояниями бесполезно: правильный шестиугольник симметричен,
    # и любой из шести поворотов даёт тот же остаток. Первый заход это показал
    # (все двенадцать вариантов по 0.01 шага), второй тоже (0.77 против 0.77).
    # Поэтому идём от привязки к осям: у обеих решёток x изображения растёт
    # вправо, y вниз, и обе разложены на 31 вертикальный столбец.
    p_col = lattice_index(P[:, 0])
    q_col = lattice_index(Q[:, 0])
    print(f"  столбцов: у flygym {p_col.max() + 1}, у flyvis {q_col.max() + 1}")
    sizes_p = np.bincount(p_col)
    sizes_q = np.bincount(q_col)
    print(f"  размеры столбцов совпадают: "
          f"{'да' if np.array_equal(sizes_p, sizes_q) else 'НЕТ'} "
          f"({sizes_p.min()}..{sizes_p.max()} против {sizes_q.min()}..{sizes_q.max()})")

    mapping = np.empty(len(P), dtype=np.int64)
    for c in range(p_col.max() + 1):
        pi = np.where(p_col == c)[0]
        qi = np.where(q_col == c)[0]
        if len(pi) != len(qi):
            raise SystemExit(f"столбец {c}: {len(pi)} против {len(qi)} — решётки разные")
        mapping[pi[np.argsort(P[pi, 1])]] = qi[np.argsort(Q[qi, 1])]
    print(f"  сопоставление взаимно однозначное: "
          f"{'да' if len(set(mapping.tolist())) == len(mapping) else 'НЕТ'}")

    print("\n----- сквозная проверка: пятно через оба конвейера -----")
    err_cols = check_end_to_end(retina, eye, torch, flyvis, mapping, Q, P)

    np.savez(out("hex_lattice_align.npz"), ommatidium_to_column=mapping,
             u=u[mapping], v=v[mapping],
             anchor_max_err_columns=max_err / ppo,
             end_to_end_err_columns=err_cols)
    print(f"\nсохранено: {out('hex_lattice_align.npz')}")

    print("\n" + "=" * 78)
    print(" КРИТЕРИЙ ПРИЁМКИ")
    print("=" * 78)
    ok1 = max_err / ppo < 1.0
    ok2 = len(set(mapping.tolist())) == len(mapping)
    ok3 = err_cols < 2.0
    print(f"  1. пятно попадает в свою колонку flyvis: "
          f"{'да' if ok1 else 'НЕТ'} ({max_err / ppo:.2f} колонки)")
    print(f"  2. сопоставление взаимно однозначное: {'да' if ok2 else 'НЕТ'}")
    print(f"  3. сквозной промах меньше двух колонок: "
          f"{'да' if ok3 else 'НЕТ'} ({err_cols:.2f})")


def lattice_index(x, n_cols=2 * HEX_EXTENT + 1):
    """Номер вертикального столбца решётки по координате x.

    Допуск считается от ожидаемого шага между столбцами, а не от разностей
    соседних значений: у flygym центры омматидиев получены усреднением пикселей
    и потому слегка дрожат, и наивный допуск дробил 31 столбец на 149.
    """
    order = np.argsort(x)
    lab = np.zeros(len(x), int)
    tol = 0.4 * (x.max() - x.min()) / (n_cols - 1)
    k, prev = 0, x[order[0]]
    for i in order[1:]:
        if x[i] - prev > tol:
            k += 1
        lab[i] = k
        prev = x[i]
    return lab


def check_end_to_end(retina, eye, torch, flyvis, mapping, Q, P):
    """Пятно ставится в изображение глаза flygym, проходит его сетчатку,
    перекладывается на колонки flyvis и сверяется с ожидаемым местом."""
    H, W = retina.nrows, retina.ncols
    spots = {"верх-лево": (0.3, 0.3), "верх-право": (0.7, 0.3),
             "низ-лево": (0.3, 0.7), "низ-право": (0.7, 0.7), "центр": (0.5, 0.5)}
    step_q = np.median(np.diff(np.unique(np.round(Q[:, 0], 3))))
    print(f"  {'пятно':<12s} {'место в глазу':>16s} {'колонка flyvis':>18s} "
          f"{'ожидалось':>18s} {'промах, колонок':>16s}")
    errs = []
    for name, (fx, fy) in spots.items():
        img = np.zeros((H, W, 3), dtype=np.uint8)
        cx, cy = int(fx * W), int(fy * H)
        img[max(cy - 30, 0):cy + 30, max(cx - 30, 0):cx + 30, :] = 255
        vals = retina.raw_image_to_hex_pxls(img.astype(np.float32) / 255.0)
        omm = int(np.argmax(vals.sum(axis=1)))
        col = int(mapping[omm])
        # ожидаемое место: та же относительная позиция на мониторе flyvis
        ex = fx * (Q[:, 0].max() - Q[:, 0].min()) + Q[:, 0].min()
        ey = fy * (Q[:, 1].max() - Q[:, 1].min()) + Q[:, 1].min()
        err = float(np.hypot(Q[col, 0] - ex, Q[col, 1] - ey) / step_q)
        errs.append(err)
        print(f"  {name:<12s} {f'({cx}, {cy})':>16s} "
              f"{f'({Q[col, 0]:.0f}, {Q[col, 1]:.0f})':>18s} "
              f"{f'({ex:.0f}, {ey:.0f})':>18s} {err:>16.2f}")
    return float(np.max(errs))


if __name__ == "__main__":
    main()
