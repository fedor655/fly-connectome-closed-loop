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


def eyes_panel(retina, gray, mu, lo, hi):
    """Мозаика омматидиев, а под ней — то, что от неё остаётся мозгу.

    Полоса залита ровно тем числом, которое идёт на вход зрительным нейронам:
    одно на глаз вместо 721. Под ней — его же положение внутри диапазона всей
    записи, иначе движения может быть не видно вовсе: вдали от объекта средняя
    яркость глаза гуляет на единицы процентов, ради этого в контуре и стоит
    DARK_GAIN = 6.
    """
    cols = []
    for k in (0, 1):
        mosaic = retina.hex_pxls_to_human_readable(gray[k], color_8bit=True)
        w = mosaic.shape[1]
        flat = np.full((50, w), np.uint8(np.clip(mu[k], 0.0, 1.0) * 255))
        frac = 0.5 if hi <= lo else float((mu[k] - lo) / (hi - lo))
        meter = np.zeros((16, w), np.uint8)
        meter[:, :max(int(frac * w), 1)] = 255
        rule = np.full((4, w), 255, np.uint8)
        cols.append(np.vstack([mosaic, rule, flat, rule, meter]))
    gap = np.full((cols[0].shape[0], 4), 255, np.uint8)
    return np.repeat(np.hstack([cols[0], gap, cols[1]])[:, :, None], 3, axis=2)


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


def strips_panel(retina, gray, om_strip, id_map, rates, hz_max, front_strip, draw=None):
    """Мозаика с разметкой на полосы, а под ней — частота каждой полосы.

    Это и есть то, что на самом деле уходит в мозг в режиме --spatial: не одно
    число на глаз, а по числу на полосу, и каждая полоса стимулирует свою группу
    примерно из тысячи зрительных проекционных нейронов.

    Глаза ЗЕРКАЛЬНЫ по азимуту, и это здесь видно глазами: у левого глаза полоса
    «перёд» лежит у правого края картинки (колонки 315-447), у правого — у левого
    (0-132). Пока карта была одна на оба глаза, «перёд» попадал в полосу 3 слева и
    в полосу 0 справа, и подача правого глаза шла перевёрнутой. Ошибка стоила бы
    всего этапа 2, поэтому метка «перёд» вынесена в картинку.
    """
    n_strips = int(om_strip.max()) + 1
    cols, centers = [], []
    for k in (0, 1):
        mosaic = retina.hex_pxls_to_human_readable(gray[k], color_8bit=True)
        sp, colstrip = strip_pixels(id_map, om_strip[k], n_strips)
        img = np.repeat(mosaic[:, :, None], 3, axis=2)
        edge = np.zeros_like(sp, bool)
        edge[:, 1:] = (sp[:, 1:] != sp[:, :-1]) & (sp[:, 1:] >= 0) & (sp[:, :-1] >= 0)
        img[edge] = (255, 220, 0)                       # жёлтая граница полос
        w = img.shape[1]
        band = np.zeros((64, w, 3), np.uint8)
        mark = np.zeros((30, w, 3), np.uint8)
        for c, s in enumerate(colstrip):
            if s < 0:
                continue
            v = int(np.clip(rates[k][s] / hz_max, 0.0, 1.0) * 255)
            band[:, c] = (v, v, v)
            if s == front_strip:
                mark[:, c] = (0, 200, 255)              # где «перёд» на этой сетчатке
        rule = np.full((3, w, 3), 255, np.uint8)
        cols.append(np.vstack([img, rule, band, rule, mark]))
        # Центр каждой полосы В ПИКСЕЛЯХ, а не по номеру: у правого глаза
        # полосы идут зеркально, и подпись, поставленная по номеру, встаёт над
        # чужим сегментом. Ровно та же зеркальность один раз уже перевернула
        # подачу правого глаза, поэтому здесь она считается, а не угадывается.
        off = k * (w + 6)
        centers.append([off + float(np.where(colstrip == s)[0].mean())
                        if (colstrip == s).any() else off + w / 2
                        for s in range(n_strips)])
    gap = np.full((cols[0].shape[0], 6, 3), 255, np.uint8)
    panel = np.hstack([cols[0], gap, cols[1]])
    if draw is not None:
        panel = draw(panel, rates, hz_max, np.array(centers), front_strip)
    return panel


FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",      # macOS
    "/usr/share/fonts/noto/NotoSans-Regular.ttf",        # Arch, CachyOS
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",   # Debian, Ubuntu
)


def _font(size):
    """TTF с кириллицей. Встроенный шрифт PIL её рисует квадратами."""
    from PIL import ImageFont
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return None


def label_panel(panel, rates, hz_max, centers, front_strip):
    """Подписи частот на полосах. Без PIL или без TTF возвращает картинку как есть."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return panel
    font, small = _font(26), _font(20)
    if font is None:
        return panel
    im = Image.fromarray(panel)
    d = ImageDraw.Draw(im)
    n = rates.shape[1]
    eye_w = (panel.shape[1] - 6) // 2
    band_top = panel.shape[0] - 97                       # 64 полоса + 3 линия + 30 метка
    for k in (0, 1):
        x0 = k * (eye_w + 6)
        for s in range(n):
            txt = f"{rates[k][s]:.0f}"
            x = centers[k][s]
            w = d.textlength(txt, font=font)
            # частота пишется поверх своей полосы; цвет по яркости фона, иначе
            # на светлой полосе красное по белому не читается
            fill = (200, 0, 0) if rates[k][s] / hz_max > 0.55 else (255, 120, 120)
            d.text((x - w / 2, band_top + 16), txt, fill=fill, font=font)
        d.text((x0 + 8, band_top - 34),
               ("левый глаз" if k == 0 else "правый глаз") + "   Гц на полосу",
               fill=(120, 255, 120), font=small)
        # подпись к голубой метке «перёд» — она у разных глаз с разных сторон
        xf = centers[k][front_strip]
        w = d.textlength("перёд", font=small)
        # чёрным: текст ложится ПОВЕРХ голубой метки, и голубым по голубому
        # его не видно вовсе
        d.text((xf - w / 2, panel.shape[0] - 27), "перёд", fill=(0, 0, 0), font=small)
    return np.asarray(im)


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
    ap.add_argument("--mean", action="store_true",
                    help="с --cam eyes: под каждым глазом то, что от него "
                         "остаётся мозгу — одно усреднённое число вместо 721")
    ap.add_argument("--strips", action="store_true",
                    help="с --cam eyes: разметить сетчатку на полосы поля зрения "
                         "и показать частоту, которую каждая полоса гонит в свою "
                         "группу зрительных проекционных нейронов (режим --spatial)")
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

    sim = build_scene(z["pillar"], z["geom_pos"],
                      light=float(z["light"]) if "light" in z else 1.0)
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

    if eyes and args.strips:
        # Импорты ленивые: без --strips просмотрщик не должен тянуть torch.
        from tools.visual_field_map import OMMATIDIA_MAP, load_or_build
        from closed_loop_vision import LIGHT_HZ, SHADE_HZ, strip_intensity

        vfm = load_or_build()
        om = vfm["ommatidia"]                       # (2, 721), строка на глаз
        id_map = np.load(OMMATIDIA_MAP)
        n_strips = int(om.max()) + 1
        FRONT = n_strips - 1                        # проверено ассертом в карте
        for k, side in ((0, "левый"), (1, "правый")):
            sp, col = strip_pixels(id_map, om[k], n_strips)
            rng = [f"{s}:{np.where(col == s)[0].min()}-{np.where(col == s)[0].max()}"
                   for s in range(n_strips) if (col == s).any()]
            print(f"{side} глаз, колонки полос: {' '.join(rng)}  «перёд» = полоса {FRONT}")

        raw = []
        for i, _ in steps:
            d.qpos[:] = qpos[min(max(i, 0), n - 1)]
            mujoco.mj_forward(m, d)
            raw.append(sim.get_ommatidia_readouts(fly_name).sum(axis=2))
        # Частоты считаются той же формулой, что в контуре (--drive light), но
        # БЕЗ фильтра tau=100 мс: здесь мгновенные значения кадра, а мозг видит
        # сглаженные. Разметка полос от этого не меняется, частоты дрожат сильнее.
        rates = []
        for g in raw:
            inten = strip_intensity(g, om, n_strips)
            B = float(inten.mean())
            rates.append(LIGHT_HZ * B + SHADE_HZ * np.clip(B - inten, 0.0, None))
        rates = np.array(rates)
        hz_max = float(rates.max())
        print(f"частоты на входе: {rates.min():.1f}..{hz_max:.1f} Гц; "
              f"освещённость B по записи "
              f"{min(float(strip_intensity(g, om, n_strips).mean()) for g in raw):.3f}.."
              f"{max(float(strip_intensity(g, om, n_strips).mean()) for g in raw):.3f}")
        with imageio.get_writer(out_path, fps=fps, macro_block_size=1) as writer:
            for g, r in zip(raw, rates):
                writer.append_data(strips_panel(sim.retina, g, om, id_map, r,
                                                hz_max, FRONT, draw=label_panel))
        print(f"готово: {out_path}")
        return

    if eyes and args.mean:
        # Первым проходом собираем сами показания: диапазон усреднённой яркости
        # по всей записи нужен до того, как рисовать первый кадр. Дорогая часть —
        # рендер глаз, поэтому мозаику потом строим из сохранённого, не заново.
        raw = []
        for i, _ in steps:
            d.qpos[:] = qpos[min(max(i, 0), n - 1)]
            mujoco.mj_forward(m, d)
            raw.append(sim.get_ommatidia_readouts(fly_name).sum(axis=2))
        mus = np.array([r.mean(axis=1) for r in raw])
        lo, hi = float(mus.min()), float(mus.max())
        print(f"усреднение по глазу: {lo:.4f}..{hi:.4f} "
              f"(размах {100 * (hi - lo) / hi:.1f}% от 721 омматидия)")
        with imageio.get_writer(out_path, fps=fps, macro_block_size=1) as writer:
            for gray, mu in zip(raw, mus):
                writer.append_data(eyes_panel(sim.retina, gray, mu, lo, hi))
        print(f"готово: {out_path}")
        return

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


