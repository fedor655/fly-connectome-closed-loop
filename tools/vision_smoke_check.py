"""Проверка зрения тела: видят ли глаза мухи объект и различают ли его сторону.

Прежде чем строить отображение омматидиев на LC-нейроны, надо убедиться, что
сама зрительная часть работает: что get_ommatidia_readouts возвращает то, что
ожидается, и что при объекте слева левый глаз реагирует иначе правого.

Ставим тёмный столб сбоку от траектории, гоним муху вперёд с постоянной
командой и смотрим разницу яркости между глазами. Затем переносим столб на
другую сторону — знак разницы обязан смениться. Если не сменится, значит
глаза перепутаны или показания не привязаны к сторонам, и строить на этом
ретинотопию нельзя.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flypaths import out  # noqa: E402

# MUJOCO_GL под платформу выставляет flypaths, импортированный выше

from flygym.compose import FlatGroundWorld  # noqa: E402
from flygym.simulation import Simulation  # noqa: E402
from flygym.utils.math import Rotation3D  # noqa: E402
from flygym.utils.mjcf import GEOM_TYPES  # noqa: E402
from flygym_demo.complex_terrain import (  # noqa: E402
    HybridControllerObservation,
    HybridTurningController,
    PreprogrammedSteps,
    apply_locomotion_action,
    make_locomotion_fly,
)

SYNC_MS = 15.0
CYCLES = 60
PILLAR_X = 8.0        # мм впереди точки старта
PILLAR_Y = 3.0        # мм вбок; знак задаёт сторону
PILLAR_R = 1.2
PILLAR_H = 6.0


class PillarWorld(FlatGroundWorld):
    """Плоский грунт плюс один тёмный столб сбоку."""

    def __init__(self, side_sign: float) -> None:
        super().__init__(name="pillar_world", half_size=200)
        self.mjcf_root.worldbody.add_geom(
            type=GEOM_TYPES["cylinder"],
            name="pillar",
            size=[PILLAR_R, PILLAR_H / 2, 0.0],
            pos=[PILLAR_X, side_sign * PILLAR_Y, PILLAR_H / 2],
            rgba=[0.05, 0.05, 0.05, 1.0],
            contype=0,
            conaffinity=0,
        )


def run(side_sign, label):
    fly = make_locomotion_fly()
    fly.add_vision()
    world = PillarWorld(side_sign)
    world.add_fly(
        fly,
        spawn_position=[0.0, 0.0, 0.5],
        spawn_rotation=Rotation3D("quat", [1, 0, 0, 0]),
        add_ground_contact_sensors=True,
    )
    sim = Simulation(world)
    sim.reset()

    controller = HybridTurningController(
        timestep=sim.timestep, preprogrammed_steps=PreprogrammedSteps())
    controller.reset(seed=0)
    sim.warmup(0.05)

    inner = int(round(SYNC_MS / 1000.0 / sim.timestep))
    cmd = np.array([0.9, 0.9])

    left_mean, right_mean = [], []
    shape_reported = False
    t0 = time.perf_counter()

    for cycle in range(CYCLES):
        for _ in range(inner):
            obs = HybridControllerObservation.from_sim(sim, fly.name)
            apply_locomotion_action(sim, fly.name, controller.step(cmd, obs))
            sim.step()

        readouts = sim.get_ommatidia_readouts(fly.name)
        if not shape_reported:
            print(f"  форма показаний: {readouts.shape}  "
                  f"(глаза, омматидии, каналы жёлтый/бледный)")
            print(f"  омматидиев на глаз: {readouts.shape[1]}")
            shape_reported = True

        # каждый омматидий отдаёт ненулевое значение только в своём канале,
        # поэтому суммируем по каналам, а не усредняем
        per_om = readouts.sum(axis=2)
        left_mean.append(float(per_om[0].mean()))
        right_mean.append(float(per_om[1].mean()))

    elapsed = time.perf_counter() - t0
    l = np.array(left_mean)
    r = np.array(right_mean)
    diff = l - r

    print(f"\n----- столб {label} (y={side_sign * PILLAR_Y:+.1f} мм) -----")
    print(f"  циклов {CYCLES}, время {elapsed:.0f} с")
    print(f"  левый глаз:  среднее {l.mean():.4f}, разброс {l.std():.4f}")
    print(f"  правый глаз: среднее {r.mean():.4f}, разброс {r.std():.4f}")
    print(f"  разница лево-право: среднее {diff.mean():+.4f}, "
          f"диапазон {diff.min():+.4f}..{diff.max():+.4f}")
    sim.close()
    return diff


def main():
    print("=" * 78)
    print(" ПРОВЕРКА ЗРЕНИЯ: ВИДЯТ ЛИ ГЛАЗА СТОРОНУ ОБЪЕКТА")
    print("=" * 78)

    d_left = run(+1.0, "СЛЕВА")
    d_right = run(-1.0, "СПРАВА")

    print("\n" + "=" * 78)
    print(" ИТОГ")
    print("=" * 78)
    print(f"  столб слева:  разница лево-право {d_left.mean():+.4f}")
    print(f"  столб справа: разница лево-право {d_right.mean():+.4f}")
    flipped = np.sign(d_left.mean()) != np.sign(d_right.mean())
    magnitude = abs(d_left.mean() - d_right.mean())
    print(f"  знак сменился: {'ДА' if flipped else 'НЕТ'}")
    print(f"  размах эффекта: {magnitude:.4f}")
    ok = flipped and magnitude > 1e-3
    print(f"\n  {'ОК: глаза различают сторону объекта' if ok else 'ПРОВАЛ: сторона не различается'}")

    np.savetxt(out("vision_smoke_check.csv"),
               np.column_stack([d_left, d_right]),
               delimiter=",", header="diff_pillar_left,diff_pillar_right", comments="")
    print(f"  сохранено: {out('vision_smoke_check.csv')}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
