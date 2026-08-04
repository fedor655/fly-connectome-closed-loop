"""Запись траектории прогона и пересборка сцены для воспроизведения.

Смысл: прогон стоит минут (мозг на 138 639 нейронов), а посмотреть на него
хочется с разных ракурсов. Пиксели для этого хранить незачем — MuJoCo
восстанавливает кадр из qpos: положил состояние, вызвал mj_forward, отрендерил
любой камерой. nq = 73, то есть 584 байта на кадр; 1500 кадров — 0.88 МБ,
втрое меньше нынешнего mp4 с одной камеры.

Скомпилированную модель в запись не кладём: сборка сцены с нуля занимает 0.07 с,
а MJB весит 22.9 МБ.

Сцена одна на все прогоны: столб в контрольных сценах не убирается, а уносится
на FAR_AWAY — так же, как это уже делает closed_loop_vision.py. Поэтому для
воспроизведения достаточно двух чисел.
"""
from __future__ import annotations

import numpy as np

import flypaths  # noqa: F401  — ставит MUJOCO_GL до импорта mujoco

from flygym.compose import FlatGroundWorld
from flygym.simulation import Simulation
from flygym.utils.math import Rotation3D
from flygym.utils.mjcf import GEOM_TYPES
from flygym_demo.complex_terrain import make_locomotion_fly

PILLAR_R = 1.2
PILLAR_H = 8.0
FAR_AWAY = 500.0          # куда убирать столб в контрольной сцене


class PillarWorld(FlatGroundWorld):
    """Плоский грунт и один тёмный столб."""

    def __init__(self, x: float, y: float) -> None:
        super().__init__(name="pillar_world", half_size=300)
        self.mjcf_root.worldbody.add_geom(
            type=GEOM_TYPES["cylinder"], name="pillar",
            size=[PILLAR_R, PILLAR_H / 2, 0.0],
            pos=[x, y, PILLAR_H / 2],
            rgba=[0.05, 0.05, 0.05, 1.0],
            contype=0, conaffinity=0,
        )


class Recorder:
    """Складывает qpos каждый every-й шаг физики.

    Шаг физики — 0.1 мс, поэтому писать каждый смысла нет: every=10 даёт
    килогерцовую запись, которой хватает на любую замедленную перемотку.
    """

    def __init__(self, sim: Simulation, every: int = 10) -> None:
        self.sim = sim
        self.every = every
        self.qpos: list[np.ndarray] = []
        self.t: list[float] = []
        self._i = 0

    def grab(self) -> None:
        if self._i % self.every == 0:
            self.qpos.append(self.sim.mj_data.qpos.copy())
            self.t.append(float(self.sim.mj_data.time))
        self._i += 1

    def save(self, path: str, pillar_xy: tuple[float, float]) -> str:
        np.savez(
            path,
            qpos=np.asarray(self.qpos),
            t=np.asarray(self.t),
            # то, что живёт в модели, а не в состоянии: без этого сцена
            # воспроизведётся неверно, если геометрию двигали в рантайме
            geom_pos=self.sim.mj_model.geom_pos.copy(),
            pillar=np.asarray(pillar_xy, dtype=float),
            timestep=float(self.sim.timestep),
            every=self.every,
        )
        return path


def eye_view(sim: Simulation, fly_name: str, step: int = 1) -> np.ndarray:
    """Что видят глаза: 721 омматидий на глаз, серым, левый | правый.

    Два канала омматидия (yellow/pale) складываются: ненулевой у него ровно
    один, так что сумма и есть яркость. Именно эта величина идёт на вход мозгу
    в closed_loop_vision.py, только там она ещё усредняется в одно число на глаз.

    step прореживает картинку для оверлея в окне: 512x904 в углу не нужны.
    """
    r = sim.get_ommatidia_readouts(fly_name)                    # (2, 721, 2)
    eyes = [sim.retina.hex_pxls_to_human_readable(r[k].sum(axis=1), color_8bit=True)
            for k in (0, 1)]
    gap = np.full((eyes[0].shape[0], 4), 255, np.uint8)
    gray = np.hstack([eyes[0], gap, eyes[1]])[::step, ::step]
    return np.repeat(gray[:, :, None], 3, axis=2)


def flight_entry(cam, frame: int) -> dict:
    """Состояние камеры одним кадром полёта. Пишет replay_view, читает replay_render.

    В режиме привязки к телу lookat считает сам MuJoCo, поэтому запоминаем и
    режим с номером тела — иначе орбита вокруг мухи при рендере рассыплется.
    """
    return {"frame": int(frame), "type": int(cam.type),
            "trackbodyid": int(cam.trackbodyid),
            "lookat": list(map(float, cam.lookat)),
            "distance": float(cam.distance),
            "azimuth": float(cam.azimuth),
            "elevation": float(cam.elevation)}


def apply_flight(cam, e: dict) -> None:
    """Обратное к flight_entry."""
    cam.type = e["type"]
    cam.trackbodyid = e["trackbodyid"]
    cam.lookat[:] = e["lookat"]
    cam.distance = e["distance"]
    cam.azimuth = e["azimuth"]
    cam.elevation = e["elevation"]


def build_scene(pillar_xy, geom_pos=None) -> Simulation:
    """Пересобрать ту же сцену без мозга. Около 0.07 с."""
    fly = make_locomotion_fly()
    fly.add_vision()
    fly.add_tracking_camera(name="trackcam")
    world = PillarWorld(float(pillar_xy[0]), float(pillar_xy[1]))
    world.add_fly(fly, spawn_position=[0.0, 0.0, 0.5],
                  spawn_rotation=Rotation3D("quat", [1, 0, 0, 0]),
                  add_ground_contact_sensors=True)
    sim = Simulation(world)
    sim.reset()
    if geom_pos is not None:
        geom_pos = np.asarray(geom_pos)
        if geom_pos.shape == sim.mj_model.geom_pos.shape:
            sim.mj_model.geom_pos[:] = geom_pos
        else:
            # запись сделана в сцене с другим числом геометрий (например,
            # closed_loop_v2 строит мир без столба). Столб на своём месте по
            # pillar_xy, остальное совпадает — это не повод падать.
            print(f"[replay] geom_pos пропущен: {geom_pos.shape} вместо "
                  f"{sim.mj_model.geom_pos.shape}")
    return sim
