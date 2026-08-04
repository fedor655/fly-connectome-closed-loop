"""Проверка: кадр, восстановленный из записи, совпадает с кадром живого прогона.

Ломается, если запись или воспроизведение сломаны — если в npz перестанет
попадать нужное состояние, если сцена соберётся не той же, если геометрия мира
разъедется.

Мозг не запускается: 0.3 с ходьбы под постоянной командой, как в
tools/body_walk_check.py. Секунды, а не минуты.

    .venv/bin/python test_replay.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flyreplay import Recorder, apply_flight, build_scene, flight_entry  # noqa: E402

import mujoco  # noqa: E402
from flygym_demo.complex_terrain import (  # noqa: E402
    HybridControllerObservation,
    HybridTurningController,
    PreprogrammedSteps,
    apply_locomotion_action,
)

DURATION_S = 0.3
EVERY = 10
PILLAR = (12.0, 3.0)
RES = (240, 320)          # высота, ширина — мелко и быстро, для сравнения хватает


def render(sim, cam="nmf/trackcam"):
    r = mujoco.Renderer(sim.mj_model, height=RES[0], width=RES[1])
    r.update_scene(sim.mj_data, camera=cam)
    img = r.render().copy()
    r.close()
    return img


def main():
    # --- живой прогон: те же движения, что в настоящем эксперименте, но без мозга
    sim = build_scene(PILLAR)
    controller = HybridTurningController(
        timestep=sim.timestep, preprogrammed_steps=PreprogrammedSteps())
    controller.reset(seed=0)
    sim.warmup(0.05)

    rec = Recorder(sim, every=EVERY)
    fly_name = next(iter(sim.world.fly_lookup))
    cmd = np.array([1.0, 0.3])          # поворот: и позы ног, и курс меняются
    live = {}
    n_steps = round(DURATION_S / sim.timestep)
    # Сравнивать можно только на шагах, которые попали в запись: между ними
    # состояние живёт, а в npz его нет — на то и every.
    shots = {(n_steps // 3) // EVERY * EVERY, (n_steps - 1) // EVERY * EVERY}
    for k in range(n_steps):
        obs = HybridControllerObservation.from_sim(sim, fly_name)
        apply_locomotion_action(sim, fly_name, controller.step(cmd, obs))
        sim.step()
        rec.grab()
        if k in shots:
            # mj_step интегрирует состояние, но производные величины (xpos, с
            # которых рисуется кадр) остаются от состояния ДО шага. Без этого
            # вызова живой кадр отстаёт от записанного qpos на 0.1 мс — глазу
            # не видно, побитовому сравнению видно.
            mujoco.mj_forward(sim.mj_model, sim.mj_data)
            live[len(rec.qpos) - 1] = render(sim)

    with tempfile.TemporaryDirectory() as tmp:
        path = rec.save(str(Path(tmp) / "traj.npz"), PILLAR)
        sim.close()

        # --- воспроизведение: сцена с нуля, состояние из файла
        z = np.load(path)
        sim2 = build_scene(z["pillar"], z["geom_pos"])
        qpos = z["qpos"]
        assert qpos.shape[1] == sim2.mj_model.nq, "nq не совпал"
        assert len(qpos) == -(-n_steps // EVERY), "потеряны кадры"
        assert abs(float(z["timestep"]) - sim2.timestep) < 1e-12

        for idx, shot in live.items():
            sim2.mj_data.qpos[:] = qpos[idx]
            mujoco.mj_forward(sim2.mj_model, sim2.mj_data)
            back = render(sim2)
            diff = np.abs(back.astype(int) - shot.astype(int))
            assert diff.max() == 0, (
                f"кадр {idx} разошёлся: максимум {diff.max()}, "
                f"среднее {diff.mean():.3f}")
            print(f"  кадр {idx}: совпал побитово")

        # столб на месте: иначе сцена воспроизводится неверно, а поза — верно
        pid = mujoco.mj_name2id(sim2.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "pillar")
        assert np.allclose(sim2.mj_model.geom_pos[pid][:2], PILLAR), "столб уехал"

        # контроль на самопроверку: неверное состояние обязано дать другой кадр
        sim2.mj_data.qpos[:] = qpos[0]
        mujoco.mj_forward(sim2.mj_model, sim2.mj_data)
        assert np.abs(render(sim2).astype(int)
                      - live[max(live)].astype(int)).max() > 0, \
            "проверка ничего не проверяет: разные кадры вышли одинаковыми"
        sim2.close()

    # Полёт камеры: просмотрщик пишет json, рендер его читает. Окно здесь не
    # поднять, поэтому проверяется сам шов — через json и обратно в камеру.
    src = mujoco.MjvCamera()
    src.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    src.trackbodyid, src.distance, src.azimuth, src.elevation = 7, 12.5, 137.0, -21.0
    src.lookat[:] = (1.0, -2.0, 3.0)
    dst = mujoco.MjvCamera()
    apply_flight(dst, json.loads(json.dumps(flight_entry(src, 42))))
    assert (dst.type, dst.trackbodyid) == (src.type, src.trackbodyid)
    assert (dst.distance, dst.azimuth, dst.elevation) == (12.5, 137.0, -21.0)
    assert np.allclose(dst.lookat, src.lookat), "lookat не пережил json"

    print(f"OK: {len(qpos)} кадров, {qpos.nbytes / 1e6:.2f} МБ")


if __name__ == "__main__":
    main()
