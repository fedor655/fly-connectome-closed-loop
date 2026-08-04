"""Рендер mp4 из записанной траектории: любой ракурс, мозг не запускается.

Источники картинки:

    --flight f.json   полёт, снятый мышью в replay_view.py, кадр в кадр
    --cam trackcam    следящая камера прогона (она же по умолчанию)
    --cam l_eye_cam   камера на месте левого глаза, обычная картинка
    --cam eyes        то, что реально видит мозг: 721 омматидий на глаз
                      в оттенках серого, левый | правый

    .venv/bin/python replay_render.py output/..._traj.npz --cam eyes -o eyes.mp4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flyreplay import apply_flight, build_scene, eye_view  # noqa: E402

import mujoco  # noqa: E402


def resolve_camera(m, name):
    """Имя камеры с точностью до префикса пространства имён ('nmf/')."""
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(m.ncam)]
    for full in names:
        if full == name or full.split("/")[-1].startswith(name):
            return full
    sys.exit(f"камеры «{name}» нет. Есть: {', '.join(names)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("npz", help="запись прогона (output/*_traj.npz)")
    ap.add_argument("-o", "--out", default=None, help="mp4 (по умолчанию рядом с npz)")
    ap.add_argument("--flight", default=None, help="json полёта камеры из replay_view.py")
    ap.add_argument("--cam", default="trackcam",
                    help="встроенная камера модели либо «eyes» — 721 омматидий "
                         "на глаз в оттенках серого, как их видит мозг")
    ap.add_argument("--res", type=int, nargs=2, default=(640, 480), metavar=("W", "H"))
    ap.add_argument("--fps", type=float, default=None,
                    help="кадров в секунду в mp4 (с --flight берётся из json)")
    ap.add_argument("--speed", type=float, default=0.2,
                    help="доля реального времени (только без --flight)")
    ap.add_argument("--from", dest="lo", type=int, default=0, help="первый кадр записи")
    ap.add_argument("--to", dest="hi", type=int, default=None, help="последний кадр")
    args = ap.parse_args()

    z = np.load(args.npz)
    qpos = z["qpos"]
    n = len(qpos)
    frame_dt = float(z["timestep"]) * int(z["every"])
    hi = n if args.hi is None else min(args.hi, n)

    sim = build_scene(z["pillar"], z["geom_pos"])
    m, d = sim.mj_model, sim.mj_data
    if qpos.shape[1] != m.nq:
        sys.exit(f"запись не подходит к сцене: nq {qpos.shape[1]} против {m.nq}")

    w, h = args.res
    # Офскрин-буфер модели может быть меньше запрошенного — Renderer тогда падает.
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, w)
    m.vis.global_.offheight = max(m.vis.global_.offheight, h)

    eyes = args.cam == "eyes"
    fly_name = next(iter(sim.world.fly_lookup))

    if args.flight and not eyes:
        flight = json.loads(Path(args.flight).read_text())
        fps = args.fps or flight.get("fps", 30.0)
        cam = mujoco.MjvCamera()
        steps = [(e["frame"], e) for e in flight["cam"]]
        what = f"полёт из {args.flight}, {len(steps)} кадров"
    else:
        fps = args.fps or 25.0
        cam = None if eyes else resolve_camera(m, args.cam)
        step = max((1.0 / fps) * args.speed / frame_dt, 1e-9)
        steps = [(int(args.lo + k * step), None)
                 for k in range(int((hi - args.lo) / step))]
        what = ("глаза мухи" if eyes else f"камера {cam}") + \
               f", замедление {args.speed:g}×"

    suffix = Path(args.flight).stem if args.flight and not eyes else args.cam
    out_path = args.out or f"{Path(args.npz).with_suffix('')}_{suffix}.mp4"
    # Размер картинки с глаз задаёт сетка омматидиев, --res к ней не относится
    renderer = None if eyes else mujoco.Renderer(m, height=h, width=w)
    size = "512×904" if eyes else f"{w}×{h}"
    print(f"{what} -> {out_path} ({len(steps)} кадров {size} @ {fps:g} fps)")

    with imageio.get_writer(out_path, fps=fps, macro_block_size=1) as writer:
        for i, entry in steps:
            d.qpos[:] = qpos[min(max(i, 0), n - 1)]
            mujoco.mj_forward(m, d)
            if eyes:
                writer.append_data(eye_view(sim, fly_name))
                continue
            if entry is not None:
                apply_flight(cam, entry)
            renderer.update_scene(d, camera=cam)
            writer.append_data(renderer.render())

    print(f"готово: {out_path}")


if __name__ == "__main__":
    main()


