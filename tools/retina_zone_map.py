"""Зонная карта: какой участок поля зрения на какие фоторецепторы идёт.

Сейчас весь глаз сводится к одному числу и все зрительные нейроны своей
стороны получают одинаковую стимуляцию. Здесь строится грубая пространственная
карта: лист R1-6 режется на 4 зоны, поле зрения flygym — на те же 4 зоны.

Ориентация листа относительно поля зрения из координат НЕ выводится, поэтому
соответствие зон произвольно и зафиксировано один раз. Утверждать по этой карте
«объект слева» нельзя. Для различимости зон и для приближения объекта
ориентация не нужна: оба свойства не зависят от поворота карты.

Ретинотопия по LC закрыта отрицательно (tools/lc_retinotopy.py): их координаты
почти вырождены в линию. На входе картина другая — у R1-6 третья главная
компонента 0.000, лист плоский.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flypaths import ANNOTATIONS, OUTPUT_DIR, add_fly_brain_to_path  # noqa: E402

add_fly_brain_to_path()
from benchmark import path_comp  # noqa: E402

N_ZONES = 4
CACHE = OUTPUT_DIR / "retina_zone_map.npz"
OMMATIDIA_MAP = (Path(__import__("flygym").__file__).parent /
                 "assets/model/neuromechfly/vision/ommatidia_id_map.npy")


def _split_2d(P: np.ndarray) -> np.ndarray:
    """Разрезать облако точек на 4 по медиане вдоль каждой из двух осей.

    Медиана, а не среднее: она делит пополам по числу точек, поэтому зоны
    выходят равными по населению даже при неравномерном покрытии листа
    (у R1-6 отношение максимума к медиане расстояния до соседа 7-12).
    """
    a = (P[:, 0] > np.median(P[:, 0])).astype(int)
    b = (P[:, 1] > np.median(P[:, 1])).astype(int)
    return a * 2 + b


def photoreceptor_zones(n_split: int = 2) -> dict:
    comp = pd.read_csv(path_comp, index_col=0)
    flyid2i = {int(j): i for i, j in enumerate(comp.index)}

    ann = pd.read_csv(ANNOTATIONS, sep="\t", low_memory=False)
    ann["root_id"] = pd.to_numeric(ann["root_id"], errors="coerce")
    ann = ann.dropna(subset=["root_id"])
    ann["root_id"] = ann["root_id"].astype("int64")
    ann = ann[ann["root_id"].isin(flyid2i.keys()) & ann["pos_x"].notna()]

    out = {}
    for side in ("left", "right"):
        s = ann[(ann["cell_type"] == "R1-6") & (ann["side"] == side)]
        idx = np.array([flyid2i[i] for i in s["root_id"]], dtype=np.int64)
        X = s[["pos_x", "pos_y", "pos_z"]].to_numpy(float)
        X = X - X.mean(axis=0)
        # Две первые главные оси задают плоскость листа. Третья у R1-6 забирает
        # 0.000-0.001 дисперсии, то есть лист действительно плоский.
        Vt = np.linalg.svd(X, full_matrices=False)[2]
        out[side] = (idx, _split_2d(X @ Vt[:2].T))
    return out


def ommatidia_zones(n_split: int = 2) -> np.ndarray:
    """Номер зоны для каждого из 721 омматидия.

    В карте flygym id=0 это фон (60112 пикселей), реальные омматидии
    пронумерованы 1..721. Показание с индексом i соответствует id i+1 —
    это утверждение проверяется ассертом ниже.
    """
    m = np.load(OMMATIDIA_MAP)
    ids = np.unique(m)
    ids = ids[ids > 0]
    assert ids.min() == 1 and ids.max() == 721 and len(ids) == 721, \
        f"неожиданная нумерация омматидиев: {ids.min()}..{ids.max()}, {len(ids)} шт"

    flat = m.ravel().astype(np.int64)
    yy, xx = np.divmod(np.arange(flat.size), m.shape[1])
    cnt = np.bincount(flat, minlength=722).astype(float)
    cy = np.bincount(flat, weights=yy, minlength=722)[ids] / cnt[ids]
    cx = np.bincount(flat, weights=xx, minlength=722)[ids] / cnt[ids]
    return _split_2d(np.stack([cx, cy], axis=1))


def load_or_build(n_split: int = 2) -> dict:
    if CACHE.exists():
        z = np.load(CACHE)
        return {"ommatidia": z["ommatidia"],
                "left": (z["left_idx"], z["left_zone"]),
                "right": (z["right_idx"], z["right_zone"])}
    pr = photoreceptor_zones(n_split)
    om = ommatidia_zones(n_split)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, ommatidia=om,
             left_idx=pr["left"][0], left_zone=pr["left"][1],
             right_idx=pr["right"][0], right_zone=pr["right"][1])
    return {"ommatidia": om, "left": pr["left"], "right": pr["right"]}


def self_check() -> None:
    """Проверки, которые ломаются, если карта построена неверно."""
    zones = load_or_build()

    om = zones["ommatidia"]
    assert om.shape == (721,), f"омматидиев должно быть 721, получено {om.shape}"
    assert set(np.unique(om)) == set(range(N_ZONES)), "не все зоны заполнены"
    counts = np.bincount(om, minlength=N_ZONES)
    assert counts.min() * 2 >= counts.max(), (
        f"зоны поля зрения слишком неравны: {counts.tolist()}")

    for side in ("left", "right"):
        idx, zn = zones[side]
        assert len(idx) == len(zn), f"{side}: длины не совпали"
        assert len(idx) == len(set(idx.tolist())), f"{side}: индексы повторяются"
        assert set(np.unique(zn)) == set(range(N_ZONES)), f"{side}: не все зоны заполнены"
        c = np.bincount(zn, minlength=N_ZONES)
        assert c.min() * 2 >= c.max(), f"{side}: зоны листа слишком неравны: {c.tolist()}"

    assert not (set(zones["left"][0].tolist()) & set(zones["right"][0].tolist())), \
        "левые и правые фоторецепторы пересеклись"
    print("самопроверка карты зон: ОК")


if __name__ == "__main__":
    self_check()
