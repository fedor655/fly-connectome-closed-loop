"""Проверка ТОЛЬКО тела, без мозга (CPU, MuJoCo).

Смысл: прежде чем замыкать контур, надо убедиться, что вторая половина работает
сама по себе. Подаём на HybridTurningController постоянный нисходящий сигнал и
смотрим, идёт ли муха и меняются ли флаги касания лапок.

Критерии:
  [1.0, 1.0] — муха должна заметно продвинуться, касания должны меняться;
  [0.0, 0.0] — муха должна стоять;
  [1.0, 0.3] — муха должна повернуть (боковое смещение и разворот курса).
"""
import os
import sys
import time

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

from flygym.compose import FlatGroundWorld
from flygym.simulation import Simulation
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain import (
    HybridControllerObservation,
    HybridTurningController,
    PreprogrammedSteps,
    apply_locomotion_action,
    make_locomotion_fly,
)

DURATION_S = 1.5


def run(descending, label):
    fly = make_locomotion_fly()
    world = FlatGroundWorld()
    world.add_fly(
        fly,
        spawn_position=[0.0, 0.0, 0.5],
        spawn_rotation=Rotation3D("quat", [1, 0, 0, 0]),
        add_ground_contact_sensors=True,
    )
    sim = Simulation(world)
    sim.reset()

    controller = HybridTurningController(
        timestep=sim.timestep, preprogrammed_steps=PreprogrammedSteps()
    )
    controller.reset(seed=0)
    sim.warmup(0.05)

    body_order = fly.get_bodysegs_order()
    bodyseg_cls = type(fly).BODY_SEGMENT_CLASS
    thorax_idx = body_order.index(bodyseg_cls("c_thorax"))
    thorax_body_id = sim._internal_bodyids_by_fly[fly.name][thorax_idx]

    n_steps = int(DURATION_S / sim.timestep)
    sig = np.asarray(descending, dtype=float)

    start = sim.get_body_positions(fly.name)[thorax_idx].copy()
    head0 = sim.mj_data.xmat[thorax_body_id].reshape(3, 3)[:, 0].copy()

    contact_history = []
    t0 = time.perf_counter()
    for step in range(n_steps):
        obs = HybridControllerObservation.from_sim(sim, fly.name)
        action = controller.step(sig, obs)
        apply_locomotion_action(sim, fly.name, action)
        sim.step()
        if step % 150 == 0:  # раз в 15 мс, как окно синхронизации
            cf, *_ = sim.get_ground_contact_info(fly.name)
            contact_history.append((np.asarray(cf) > 0).astype(int))
    elapsed = time.perf_counter() - t0

    end = sim.get_body_positions(fly.name)[thorax_idx].copy()
    head1 = sim.mj_data.xmat[thorax_body_id].reshape(3, 3)[:, 0].copy()

    d = end - start
    dist = float(np.hypot(d[0], d[1]))
    cos_turn = float(np.clip(np.dot(head0[:2], head1[:2]) /
                             (np.linalg.norm(head0[:2]) * np.linalg.norm(head1[:2])), -1, 1))
    turn_deg = np.degrees(np.arccos(cos_turn))
    cross = head0[0] * head1[1] - head0[1] * head1[0]
    turn_deg = turn_deg if cross >= 0 else -turn_deg

    ch = np.array(contact_history)
    n_states = len({tuple(r) for r in ch})

    print(f"\n----- сигнал {label} = {sig} -----")
    print(f"  шагов: {n_steps}, время расчёта: {elapsed:.1f} с "
          f"({n_steps / elapsed:.0f} шагов/с)")
    print(f"  старт груди:  x={start[0]:+.3f} y={start[1]:+.3f} z={start[2]:.3f}")
    print(f"  финиш груди:  x={end[0]:+.3f} y={end[1]:+.3f} z={end[2]:.3f}")
    print(f"  смещение:     dx={d[0]:+.3f} dy={d[1]:+.3f} путь={dist:.3f} мм")
    print(f"  поворот курса: {turn_deg:+.1f} град")
    print(f"  среднее число касаний: {ch.sum(axis=1).mean():.2f} из 6")
    print(f"  различных паттернов касания: {n_states} из {len(ch)} замеров")
    sim.close()
    return dist, turn_deg, n_states


def main():
    print("=" * 78)
    print(" ПРОВЕРКА ТЕЛА БЕЗ МОЗГА")
    print("=" * 78)
    print(f"длительность каждого прогона: {DURATION_S} с")

    walk_dist, walk_turn, walk_states = run([1.0, 1.0], "ходьба вперёд")
    stand_dist, _, stand_states = run([0.0, 0.0], "стоять")
    turn_dist, turn_turn, _ = run([1.0, 0.3], "поворот")

    print("\n" + "=" * 78)
    print(" ИТОГ")
    print("=" * 78)
    ok_walk = walk_dist > 0.5 and walk_states > 3
    ok_stand = stand_dist < walk_dist / 3
    ok_turn = abs(turn_turn) > abs(walk_turn) + 5
    print(f"  ходьба вперёд даёт путь {walk_dist:.3f} мм, паттернов касания {walk_states}"
          f"   -> {'ОК' if ok_walk else 'ПРОВАЛ'}")
    print(f"  стоя путь {stand_dist:.3f} мм (должно быть заметно меньше)"
          f"   -> {'ОК' if ok_stand else 'ПРОВАЛ'}")
    print(f"  поворот: курс {turn_turn:+.1f} град против {walk_turn:+.1f} при прямой ходьбе"
          f"   -> {'ОК' if ok_turn else 'ПРОВАЛ'}")
    return 0 if (ok_walk and ok_stand) else 1


if __name__ == "__main__":
    sys.exit(main())
