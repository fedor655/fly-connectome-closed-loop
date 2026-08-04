"""Обученный flyvis: работает ли он и на какие направления настроены T4.

Зачем. Скачаны обученные веса. До того как строить на них контур, надо
убедиться в двух вещах, и обе проверяются одним замером.

  1. Сеть загружается и считает. Ноль на входе даёт установившееся состояние
     покоя, ненулевой вход его сдвигает.
  2. Подтипы T4 различают направления движения. Это главная проверка: T4a, T4b,
     T4c и T4d в живой мухе настроены на четыре разных направления, и если
     обученная сеть их воспроизводит, значит и веса те, и решётка ориентирована
     так, как мы думаем. Если все четыре откликаются одинаково — либо веса не
     те, либо вход подан неверно.

Стимул — движущаяся полоса, рисуется прямо на решётке flyvis, по четырём
направлениям. Отклик меряется по всем нейронам каждого подтипа.

Отрицательный контроль — неподвижная полоса: она обязана давать заметно более
слабый и одинаковый по подтипам отклик, иначе меряется не движение, а яркость.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flypaths import out  # noqa: E402

import flyvis  # noqa: E402
from flyvis.utils.hex_utils import get_hex_coords, hex_to_pixel  # noqa: E402

MODEL = "flow/0000/000"
DT = 1 / 100.0
N_FRAMES = 60
EXTENT = 15


def moving_bar(angle_deg, speed_cols_per_frame=0.6, static=False):
    """Тёмная полоса, едущая по решётке в заданном направлении.

    Возвращает (1, N_FRAMES, 1, 721) — форма, которую ждёт Stimulus.add_input.
    """
    u, v = get_hex_coords(EXTENT)
    x, y = hex_to_pixel(np.asarray(u), np.asarray(v))
    x = (x - x.mean()) / 1.5          # в единицах колонок
    y = (y - y.mean()) / np.sqrt(3)
    a = np.radians(angle_deg)
    proj = x * np.cos(a) + y * np.sin(a)
    seq = np.ones((N_FRAMES, len(u)), dtype=np.float32) * 0.5
    start = proj.min() - 3
    for t in range(N_FRAMES):
        pos = start if static else start + speed_cols_per_frame * t
        seq[t, np.abs(proj - pos) < 1.5] = 0.0
    return torch.tensor(seq[None, :, None, :])


def main():
    print("=" * 78)
    print(" ПРОВЕРКА ОБУЧЕННОГО FLYVIS")
    print("=" * 78)

    nv = flyvis.NetworkView(MODEL)
    net = nv.init_network()
    print(f"\nмодель: {MODEL}")
    print(f"устройство: {flyvis.device}")
    n_nodes = net.connectome.nodes.type[:].shape[0]
    print(f"узлов в сети: {n_nodes}")

    types = [t.decode() if isinstance(t, bytes) else str(t)
             for t in net.connectome.nodes.type[:]]
    types = np.array(types)
    t4 = {k: np.where(types == f"T4{k}")[0] for k in "abcd"}
    t5 = {k: np.where(types == f"T5{k}")[0] for k in "abcd"}
    print(f"нейронов T4a: {len(t4['a'])}, T5a: {len(t5['a'])}")

    stim = net.stimulus
    results = {}
    for label, static in (("движется", False), ("стоит", True)):
        print(f"\n----- полоса {label} -----")
        print(f"  {'направление':>12s}" + "".join(f"{'T4' + k:>9s}" for k in "abcd")
              + "".join(f"{'T5' + k:>9s}" for k in "abcd"))
        for ang in (0, 90, 180, 270):
            # simulate ждёт сам вход в форме (образец, кадр, 1, колонка),
            # а не собранный Stimulus — тот нужен для более тонких протоколов
            x = moving_bar(ang, static=static).to(flyvis.device)
            with torch.no_grad():
                states = net.simulate(x, dt=DT)
            act = states[0].cpu().numpy()          # (frames, nodes)
            base = act[:5].mean(0)
            resp = act[10:].mean(0) - base
            row = [float(resp[t4[k]].mean()) for k in "abcd"] + \
                  [float(resp[t5[k]].mean()) for k in "abcd"]
            results[(label, ang)] = row
            print(f"  {ang:>12d}" + "".join(f"{v:>9.3f}" for v in row))

    print("\n" + "=" * 78)
    print(" КРИТЕРИЙ ПРИЁМКИ")
    print("=" * 78)
    mv = np.array([results[("движется", a)] for a in (0, 90, 180, 270)])
    st = np.array([results[("стоит", a)] for a in (0, 90, 180, 270)])
    sel_mv = float(np.mean(mv.max(0) - mv.min(0)))
    sel_st = float(np.mean(st.max(0) - st.min(0)))
    print(f"  разброс отклика по направлениям: движется {sel_mv:.3f}, "
          f"стоит {sel_st:.3f}")
    print(f"  отношение: {sel_mv / max(sel_st, 1e-9):.1f}")
    for k, i in zip("abcd", range(4)):
        best = (0, 90, 180, 270)[int(np.argmax(mv[:, i]))]
        print(f"  T4{k} сильнее всего отзывается на направление {best}°")
    ok = sel_mv > 3 * max(sel_st, 1e-9)
    print(f"\n  направления различаются заметно сильнее, чем у неподвижной "
          f"полосы: {'да' if ok else 'НЕТ'}")
    if not ok:
        print("  Если нет — меряется яркость, а не движение. Дальше идти нельзя.")

    np.savez(out("flyvis_smoke_check.npz"), moving=mv, static=st)
    print(f"\nсохранено: {out('flyvis_smoke_check.npz')}")


if __name__ == "__main__":
    main()
