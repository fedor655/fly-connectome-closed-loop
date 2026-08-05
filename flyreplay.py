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
    """Плоский грунт и один тёмный столб.

    z и h нужны сценам этапа 2 запасного варианта: объект вверху против объекта
    внизу проверяет ось элевации. По умолчанию столб стоит на грунте во всю
    высоту — ровно как раньше, поэтому вызов PillarWorld(x, y) не меняется.

    light гасит мир. В сцене нет ни одного источника света (nlight = 0), всё
    освещение даёт headlight MuJoCo плюс собственные цвета неба и грунта,
    поэтому выключение headlight роняет яркость глаз всего с 0.55 до 0.46.
    По-настоящему темно становится, если погасить сами текстуры, а они
    процедурные и задаются ДО компиляции: skybox и checker строятся из rgb1/rgb2.

    Измерено, средняя яркость омматидиев при разном light:
        1.00 -> 0.553/0.578    0.50 -> 0.280/0.292    0.20 -> 0.106/0.108
        0.05 -> 0.032/0.030    0.00 -> 0.010/0.007

    При light = 1.0 ассеты не трогаются вовсе, поэтому прежние сцены
    воспроизводятся побитово (это проверяет test_replay.py).
    """

    def __init__(self, x: float, y: float,
                 z: float | None = None, h: float = PILLAR_H,
                 light: float = 1.0) -> None:
        super().__init__(name="pillar_world", half_size=300)
        if light != 1.0:
            for t in self.mjcf_root.textures:
                if t.name == "skybox":
                    t.rgb1 = [light] * 3
                    t.rgb2 = [light] * 3
                elif t.name == "checker":
                    t.rgb1 = [light * 0.3] * 3
                    t.rgb2 = [light * 0.4] * 3
            for g in self.mjcf_root.worldbody.geoms:
                if g.name == "ground_plane":
                    g.rgba = [light * 0.5, light * 0.5, light * 0.5, 1.0]
        self.mjcf_root.worldbody.add_geom(
            type=GEOM_TYPES["cylinder"], name="pillar",
            size=[PILLAR_R, h / 2, 0.0],
            pos=[x, y, h / 2 if z is None else z],
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

    def save(self, path: str, pillar_xy: tuple[float, float],
             light: float = 1.0) -> str:
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
            # Освещённость сцены НЕ выводится из qpos и не лежит в geom_pos:
            # это параметры процедурных текстур, заданные до компиляции. Без
            # неё воспроизведение тёмного прогона рисовало бы светлую картинку,
            # то есть то, что муха увидела бы, если бы свет был.
            light=float(light),
        )
        return path


def strip_pixels(id_map, om_strip_eye, n_strips):
    """Номер полосы для каждого пикселя картинки глаза и для каждой её колонки.

    Полосы режутся по квантилям координаты омматидия, а сетка гексагональная,
    поэтому у границы омматидии соседних полос перемежаются и чистой
    вертикальной линии не выходит: диапазоны колонок перекрываются (у левого
    глаза полоса 0 занимает 0-147, полоса 1 — 129-233). Поэтому граница
    рисуется попиксельно, а подпись под мозаикой — по преобладающей в колонке
    полосе. Так подпись стоит ровно под своим куском сетчатки.
    """
    # int64 обязателен: карта хранится в uint16, и 0 - 1 там уходит в 65535.
    ids = id_map.astype(np.int64)
    sp = np.where(ids > 0, om_strip_eye[np.clip(ids - 1, 0, None)], -1)
    col = np.full(sp.shape[1], -1, int)
    for c in range(sp.shape[1]):
        v = sp[:, c]
        v = v[v >= 0]
        if v.size:
            col[c] = np.bincount(v, minlength=n_strips).argmax()
    return sp, col


def eye_view(sim: Simulation, fly_name: str, step: int = 1,
             strips=None, id_map=None) -> np.ndarray:
    """Что видят глаза: 721 омматидий на глаз, серым, левый | правый.

    Два канала омматидия (yellow/pale) складываются: ненулевой у него ровно
    один, так что сумма и есть яркость. Именно эта величина идёт на вход мозгу
    в closed_loop_vision.py, но там она ещё усредняется: в скалярном режиме до
    одного числа на глаз, в пространственном (--spatial) до четырёх — по одному
    на полосу поля зрения, и каждая полоса стимулирует свою группу примерно из
    тысячи зрительных проекционных нейронов. Разбивку на полосы даёт
    tools/visual_field_map.py; здесь она НЕ показана, рисуется сырая мозаика.

    step прореживает картинку для оверлея в окне: 512x904 в углу не нужны.

    strips (карта (2, 721) из tools/visual_field_map.py) и id_map включают
    разметку на полосы: жёлтые границы и голубая метка «перёд». Границы
    считаются на УЖЕ прореженной картинке — линия в один пиксель при step=3
    иначе просто не попала бы в выборку.
    """
    r = sim.get_ommatidia_readouts(fly_name)                    # (2, 721, 2)
    eyes = [sim.retina.hex_pxls_to_human_readable(r[k].sum(axis=1), color_8bit=True)
            for k in (0, 1)]
    gap = np.full((eyes[0].shape[0], 4), 255, np.uint8)
    gray = np.hstack([eyes[0], gap, eyes[1]])[::step, ::step]
    img = np.repeat(gray[:, :, None], 3, axis=2)
    if strips is None or id_map is None:
        return img

    n_strips = int(np.asarray(strips).max()) + 1
    sp = [strip_pixels(id_map, np.asarray(strips)[k], n_strips)[0] for k in (0, 1)]
    void = np.full((sp[0].shape[0], 4), -1, int)
    spa = np.hstack([sp[0], void, sp[1]])[::step, ::step]
    edge = np.zeros(spa.shape, bool)
    edge[:, 1:] = (spa[:, 1:] != spa[:, :-1]) & (spa[:, 1:] >= 0) & (spa[:, :-1] >= 0)
    img[edge] = (255, 220, 0)
    # Метка «перёд»: у левого глаза она справа, у правого слева — глаза
    # зеркальны по азимуту, и в окне это должно быть видно так же, как в mp4.
    front = np.zeros((6, spa.shape[1], 3), np.uint8)
    front[:, spa.max(axis=0) == n_strips - 1] = (0, 200, 255)
    return np.vstack([img, front])


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


def build_scene(pillar_xy, geom_pos=None, light=1.0) -> Simulation:
    """Пересобрать ту же сцену без мозга. Около 0.07 с.

    light берётся из записи (ключ light в npz); у записей, сделанных до его
    появления, ключа нет, и вызывающий подставляет 1.0 — прежнее поведение.
    """
    fly = make_locomotion_fly()
    fly.add_vision()
    fly.add_tracking_camera(name="trackcam")
    world = PillarWorld(float(pillar_xy[0]), float(pillar_xy[1]),
                        light=float(light))
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
