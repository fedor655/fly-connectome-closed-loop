"""Интерактивный просмотр записанной траектории: свободная камера + перемотка.

Мозг здесь не запускается: сцена собирается заново за 0.07 с, состояние берётся
из npz, MuJoCo восстанавливает кадр вызовом mj_forward. Камеру можно крутить
мышью прямо во время воспроизведения, и перемотка её не сбрасывает — мы трогаем
только qpos, а cam не трогаем никогда, кроме клавиши T.

Клавишей R включается запись полёта камеры: положение камеры пишется покадрово
в json, и replay_render.py потом рендерит офскрин ровно тот же полёт в полном
разрешении.

    .venv/bin/mjpython replay_view.py output/closed_loop_vision_left_traj.npz

На macOS ТОЛЬКО через mjpython: mujoco.viewer требует главного потока.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Окну нужен glfw, а flypaths ставит безоконный бэкенд (cgl/egl) для рендера в
# файл. setdefault там не перебьёт уже выставленное значение, поэтому ставим до
# импорта — и flypaths трогать не приходится, на нём завязаны все остальные
# скрипты.
os.environ["MUJOCO_GL"] = "glfw"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flyreplay import build_scene, flight_entry  # noqa: E402

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402

# Коды GLFW. Простые буквы не заняты встроенными горячими клавишами simulate,
# в отличие от Tab (панели), Backspace (сброс) и Esc (свободная камера).
K_SPACE, K_RIGHT, K_LEFT, K_DOWN, K_UP = 32, 262, 263, 264, 265
K_PGUP, K_PGDN = 266, 267
K_B, K_H, K_R, K_T = 66, 72, 82, 84

KEYS_HELP = """
  Пробел      пуск / пауза          B    реверс
  → / ←       кадр вперёд / назад   T    камера: за мухой / свободная
  PgUp/PgDn   ±25 кадров            H    в начало
  ↑ / ↓       скорость ×2 / ÷2      R    запись полёта камеры в json
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("npz", help="запись прогона (output/*_traj.npz)")
    ap.add_argument("--speed", type=float, default=0.2,
                    help="доля реального времени на старте (0.2 = замедление в 5 раз)")
    ap.add_argument("--flight", default=None,
                    help="куда писать полёт камеры (по умолчанию рядом с npz)")
    ap.add_argument("--flight-fps", type=float, default=30.0,
                    help="частота записи полёта — она же частота кадров будущего mp4")
    args = ap.parse_args()

    if sys.platform == "darwin" and getattr(mujoco.viewer, "_MJPYTHON", None) is None:
        sys.exit("На macOS просмотр работает только через mjpython — окну нужен "
                 "главный поток. Запустите:\n"
                 f"    .venv/bin/mjpython {' '.join(sys.argv)}")

    z = np.load(args.npz)
    qpos, t = z["qpos"], z["t"]
    n = len(qpos)
    frame_dt = float(z["timestep"]) * int(z["every"])
    flight_path = args.flight or str(Path(args.npz).with_name(
        Path(args.npz).stem.replace("_traj", "") + "_flight.json"))

    sim = build_scene(z["pillar"], z["geom_pos"])
    m, d = sim.mj_model, sim.mj_data
    if qpos.shape[1] != m.nq:
        sys.exit(f"запись не подходит к сцене: nq {qpos.shape[1]} против {m.nq}")

    # Тело, за которым ходит следящая камера — оно же цель привязки по T.
    trackcam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "nmf/trackcam")
    track_body = int(m.cam_bodyid[trackcam])

    print(f"{n} кадров, {t[-1] - t[0]:.3f} с симуляции, шаг {frame_dt * 1000:.1f} мс")
    print(KEYS_HELP)

    state = {"i": 0.0, "play": True, "speed": args.speed, "dir": 1,
             "track": False, "rec": None}

    def key(code):
        if code == K_SPACE:
            state["play"] = not state["play"]
        elif code == K_RIGHT:
            state["i"], state["play"] = state["i"] + 1, False
        elif code == K_LEFT:
            state["i"], state["play"] = state["i"] - 1, False
        elif code == K_PGUP:
            state["i"], state["play"] = state["i"] + 25, False
        elif code == K_PGDN:
            state["i"], state["play"] = state["i"] - 25, False
        elif code == K_UP:
            state["speed"] *= 2
        elif code == K_DOWN:
            state["speed"] /= 2
        elif code == K_B:
            state["dir"] = -state["dir"]
        elif code == K_H:
            state["i"] = 0.0
        elif code == K_T:
            state["track"] = not state["track"]
        elif code == K_R:
            if state["rec"] is None:
                state["rec"] = []
                print("[полёт] запись пошла")
            else:
                Path(flight_path).write_text(
                    json.dumps({"npz": os.path.basename(args.npz),
                                "fps": args.flight_fps,
                                "cam": state["rec"]}, indent=1))
                print(f"[полёт] {len(state['rec'])} кадров -> {flight_path}")
                state["rec"] = None

    with mujoco.viewer.launch_passive(m, d, key_callback=key) as viewer:
        prev = time.perf_counter()
        next_shot = 0.0
        while viewer.is_running():
            now = time.perf_counter()
            dt, prev = now - prev, now

            if state["play"]:
                state["i"] += dt * state["speed"] / frame_dt * state["dir"]
            i = int(state["i"]) % n
            state["i"] = float(i) if not state["play"] else state["i"] % n

            d.qpos[:] = qpos[i]
            mujoco.mj_forward(m, d)

            with viewer.lock():
                cam = viewer.cam
                want = (mujoco.mjtCamera.mjCAMERA_TRACKING if state["track"]
                        else mujoco.mjtCamera.mjCAMERA_FREE)
                if cam.type != want:
                    cam.type = want
                    cam.trackbodyid = track_body
                # Цикл крутится сотни раз в секунду, а видео потом рисуется по
                # кадру на запись — пишем ровно с частотой будущего mp4.
                if state["rec"] is not None and now >= next_shot:
                    next_shot = now + 1.0 / args.flight_fps
                    state["rec"].append(flight_entry(cam, i))
            speed = f"{state['speed'] * state['dir']:+g}×"
            if not state["play"]:
                speed += "  пауза"
            flight = "—" if state["rec"] is None else f"ЗАПИСЬ {len(state['rec'])}"
            viewer.set_texts((None, None, "кадр\nвремя\nскорость\nполёт",
                              f"{i}/{n - 1}\n{t[i] - t[0]:.3f} с\n{speed}\n{flight}"))
            viewer.sync()
            time.sleep(0.002)

    if state["rec"]:
        print("[полёт] окно закрыто на записи, json не сохранён")


if __name__ == "__main__":
    main()
