"""Проверка карты поля зрения и пространственной подачи.

Ломается, если карта построена неверно или если подача перестала различать
участки поля зрения. Мозг не запускается: сцена, глаза, арифметика подачи —
секунды, а не минуты.

    .venv/bin/python test_spatial.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flyreplay import build_scene  # noqa: E402
from tools.visual_field_map import N_STRIPS, load_or_build  # noqa: E402
from closed_loop_vision import strip_intensity  # noqa: E402


def eye_profile(pillar_xy, om):
    sim = build_scene(pillar_xy)
    sim.warmup(0.05)
    name = next(iter(sim.world.fly_lookup))
    raw = sim.get_ommatidia_readouts(name).sum(axis=2)      # (2, 721)
    prof = strip_intensity(raw, om, N_STRIPS)
    sim.close()
    return raw, prof


def eye_profile_z(xy, z, h, om):
    """То же, что eye_profile, но объект поднят: сцены запасного варианта."""
    from flygym.simulation import Simulation
    from flygym.utils.math import Rotation3D
    from flygym_demo.complex_terrain import make_locomotion_fly
    from flyreplay import PillarWorld

    fly = make_locomotion_fly()
    fly.add_vision()
    world = PillarWorld(xy[0], xy[1], z=z, h=h)
    world.add_fly(fly, spawn_position=[0.0, 0.0, 0.5],
                  spawn_rotation=Rotation3D("quat", [1, 0, 0, 0]),
                  add_ground_contact_sensors=True)
    sim = Simulation(world)
    sim.reset()
    sim.warmup(0.05)
    raw = sim.get_ommatidia_readouts(fly.name).sum(axis=2)
    prof = strip_intensity(raw, om, N_STRIPS)
    sim.close()
    return raw, prof


def main():
    m = load_or_build()
    om = m["ommatidia"]                                      # (2, 721)

    raw_f, prof_f = eye_profile((6.0, 0.0), om)             # столб по курсу
    raw_s, prof_s = eye_profile((0.0, 6.0), om)             # столб слева
    assert prof_f.shape == (2, N_STRIPS), f"профиль {prof_f.shape}"

    # 1. скалярный режим — частный случай: одна полоса даёт ровно старую величину.
    #    Поправка А: карта омматидиев (2, 721) — своя строка на свой глаз, поэтому
    #    и заглушка для скалярного режима должна быть (2, 721), а не (721,).
    one = strip_intensity(raw_f, np.zeros((2, 721), int), 1)
    assert np.allclose(one[:, 0], raw_f.mean(axis=1)), \
        "одна полоса перестала совпадать со средним по глазу"

    # 2. разные сцены дают разные ПРОФИЛИ, а не только разную сумму
    df = prof_f - prof_f.mean(axis=1, keepdims=True)
    ds = prof_s - prof_s.mean(axis=1, keepdims=True)
    spread = float(np.abs(df - ds).max())
    assert spread > 1e-3, (
        f"профили сцен различаются только уровнем (максимум разности форм "
        f"{spread:.2e}) — пространственная подача ничего не добавляет")
    print(f"  формы профилей разошлись на {spread:.4f}")

    # 3. контроль на самопроверку: перемешанная карта обязана дать другое.
    #    Перемешиваем каждую строку (свой глаз) независимо, чтобы не выйти за
    #    форму (2, 721), которую ожидает strip_intensity.
    rng = np.random.default_rng(0)
    shuf = np.stack([rng.permutation(om[0]), rng.permutation(om[1])])
    alt = strip_intensity(raw_f, shuf, N_STRIPS)
    assert not np.allclose(alt, prof_f), \
        "перемешанная карта дала тот же профиль: проверка ничего не проверяет"

    # 4. карта покрывает обе стороны и полосы населены
    for side in ("left", "right"):
        c = np.bincount(m[f"{side}_strip"], minlength=N_STRIPS)
        assert c.min() * 2 >= c.max(), f"{side}: полосы неравны: {c.tolist()}"
        assert c.sum() > 3800, f"{side}: покрыто {c.sum()} проекционных"
        print(f"  {side}: полосы {c.tolist()}")

    # 5. сцены этапа 2: две позиции объекта вдоль оси карты
    from flyreplay import PillarWorld
    import inspect
    p = inspect.signature(PillarWorld.__init__).parameters
    assert "z" in p and "h" in p, "PillarWorld не принимает высоту"
    assert p["z"].default is None and p["h"].default is not None, \
        "старый вызов PillarWorld(x, y) должен работать без изменений"
    raw_up, prof_up = eye_profile_z((4.0, 0.0), 4.0, 1.5, om)
    raw_dn, prof_dn = eye_profile_z((4.0, 0.0), 0.75, 1.5, om)
    du = prof_up - prof_up.mean(axis=1, keepdims=True)
    dd = prof_dn - prof_dn.mean(axis=1, keepdims=True)
    assert float(np.abs(du - dd).max()) > 1e-3, \
        "объект вверху и внизу дают одинаковый профиль полос"
    print(f"  верх против низа: формы разошлись на {float(np.abs(du - dd).max()):.4f}")

    print("OK: карта и подача целы")


if __name__ == "__main__":
    main()
