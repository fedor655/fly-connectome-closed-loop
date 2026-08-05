"""Карта поля зрения: какому проекционному нейрону какой участок подавать.

Из координат самих проекционных нейронов ретинотопию взять нельзя — закрыто в
tools/lc_retinotopy.py (доли дисперсии 0.913/0.078/0.008, почти линия; координаты
это точки у сомы). Из координат сом медуллы — тоже нельзя: сохраняется одна ось
из двух (R2 0.94-0.98 против 0.02-0.24 на пяти типах клеток и двух полушариях).

Поэтому координаты сом промежуточных ступеней не используются вовсе. Положение
распространяется по синапсам от сетчатки, где лист проверен (третья главная ось
0.000-0.005) и ориентирован каймой DRA.

Оптические доли здесь не задействованы: карта решает только, кому что подавать,
а стимуляция идёт прямо в проекционные нейроны.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from flypaths import ANNOTATIONS, OUTPUT_DIR, add_fly_brain_to_path, out  # noqa: E402

add_fly_brain_to_path()
from benchmark import path_comp, path_con  # noqa: E402

# Воксели FAFB анизотропны: x и y по 4 нм, z по 40 нм. Без приведения размах по
# z выходит 3949 против 110905 по x, и облако кажется плоским, хотя это единицы.
VOXEL_NM = np.array([4.0, 4.0, 40.0])

# Анатомические реперы (смещение центра популяции от центра мозга, нм):
#   оцеллярные фоторецепторы, нерв OCN:  y = -167303  -> дорсально это -y
#   антеннальный нерв AN:                z = -116658  -> антериально это -z
#   шейный коннектив CV:                 z =  +40200, y = +54128 (зад и низ)
DORSAL_AXIS, DORSAL_SIGN = 1, -1.0
ANTERIOR_AXIS, ANTERIOR_SIGN = 2, -1.0

N_STRIPS = 4
MEDULLA_CLASSES = ("ME", "ME>LO", "ME>LOP", "ME>LO.LOP")
CACHE = OUTPUT_DIR / "visual_field_map.npz"
OMMATIDIA_MAP = (Path(__import__("flygym").__file__).parent /
                 "assets/model/neuromechfly/vision/ommatidia_id_map.npy")

# Ось карты. Split-half контроль (seed=0, output/visual_field_split_half.csv):
#   left:  n=3895, r_elev=0.974, r_azim=0.947
#   right: n=3975, r_elev=0.986, r_azim=0.915
# Положительный контроль (элевация) пройден с большим запасом: min 0.974 > 0.9.
# Азимут тоже пройден: min 0.915 > 0.5. Поэтому карта строится по азимуту.
MAP_AXIS = "azimuth"


def load_tables():
    """Аннотации и связи, отфильтрованные до нейронов нашей модели."""
    comp = pd.read_csv(path_comp, index_col=0)
    flyid2i = {int(j): i for i, j in enumerate(comp.index)}
    ann = pd.read_csv(ANNOTATIONS, sep="\t", low_memory=False)
    ann["root_id"] = pd.to_numeric(ann["root_id"], errors="coerce")
    ann = ann.dropna(subset=["root_id"])
    ann["root_id"] = ann["root_id"].astype("int64")
    ann = ann[ann["root_id"].isin(flyid2i)].drop_duplicates("root_id")
    con = pd.read_parquet(path_con,
                          columns=["Presynaptic_ID", "Postsynaptic_ID", "Connectivity"])
    return flyid2i, ann, con[con["Connectivity"] > 0]


def retina_of(ann, side):
    # Поправка А: оцеллярные фоторецепторы (cell_sub_class == "ocellar", 100
    # слева / 97 справа) сидят на темени, а не в глазу. Не исключить их —
    # значит испортить и плоскость листа, и привязку к кайме DRA: оцелли
    # тянут именно дорсальное направление, по которому идёт вся привязка.
    return ann[(ann["side"] == side) & (ann["super_class"] == "sensory")
               & (ann["cell_class"] == "visual")
               & (ann["cell_sub_class"] != "ocellar") & ann["pos_x"].notna()]


def sheet_axes(retina):
    """Оси листа сетчатки, привязанные к анатомии, а не к номеру компоненты.

    Брать первую и вторую главные компоненты нельзя: у двух глаз они нумеруются
    по-разному, и «ось 1» слева и справа означает разное (наблюдалось: контроль
    дал 0.912 против 0.585). Элевация определяется каймой DRA — dorsal rim area
    лежит у дорсального края глаза по построению. Азимут — перпендикуляр к ней
    в плоскости листа, знак по антеннальному нерву.

    `retina` — полный набор сетчатки своей стороны (R1-6, R7, R8; без
    оцеллярных, см. retina_of). Внутри функция разделяет его по ролям
    (поправка Б): из 155 нейронов с cell_sub_class == "DRA" 77 это R7 и
    77 это R8, и лишь один R1-6 — кайма DRA почти отсутствует среди R1-6.

    - mu и плоскость P (первые две главные оси) считаются ТОЛЬКО по
      cell_type == "R1-6": у них лист чистый (третья главная ось 0.000-0.005),
      а отход остальной сетчатки (R7, R8) от этой плоскости — всего 5.0 мкм
      (left) и 11.7 мкм (right) при размахе листа 100-280 мкм.
    - смещение каймы d считается по cell_sub_class == "DRA" из ПОЛНОГО набора
      retina (R1-6 + R7 + R8), а не только из R1-6 — там DRA всего один
      нейрон и норма d выродилась бы.
    """
    r16 = retina[retina["cell_type"].to_numpy() == "R1-6"]
    X16 = r16[["pos_x", "pos_y", "pos_z"]].to_numpy(float) * VOXEL_NM
    mu = X16.mean(axis=0)
    P = np.linalg.svd(X16 - mu, full_matrices=False)[2][:2]    # плоскость листа по R1-6

    X = retina[["pos_x", "pos_y", "pos_z"]].to_numpy(float) * VOXEL_NM
    dra = retina["cell_sub_class"].to_numpy() == "DRA"          # кайма по всей сетчатке
    d = (X[dra].mean(axis=0) - mu) @ P.T
    e2 = d / np.linalg.norm(d)                                 # к кайме = дорсально
    a2 = np.array([-e2[1], e2[0]])                             # поворот на 90°
    E, A = e2 @ P, a2 @ P
    E /= np.linalg.norm(E)
    A /= np.linalg.norm(A)
    if A[ANTERIOR_AXIS] * ANTERIOR_SIGN < 0:
        A = -A                                                 # знак азимута по антеннам
    return mu, E, A


def _wmean(edges, srcpos):
    """Взвешенное среднее положений пресинаптических партнёров."""
    p = np.array([srcpos[i] for i in edges["Presynaptic_ID"]])
    w = edges["Connectivity"].to_numpy(float)
    d = pd.DataFrame({"post": edges["Postsynaptic_ID"].to_numpy(),
                      "e": p[:, 0] * w, "a": p[:, 1] * w, "w": w}).groupby("post").sum()
    return {int(k): np.array([r.e / r.w, r.a / r.w]) for k, r in d.iterrows()}


def _edges_to(con, targets, srcpos):
    tg = set(int(x) for x in targets)
    return con[con["Postsynaptic_ID"].isin(tg) & con["Presynaptic_ID"].isin(srcpos)]


def field_positions(ann, con, side):
    """Положение каждого проекционного нейрона в поле зрения.

    Цепочка: сетчатка (координаты) -> ламина -> медулла -> проекционные.
    Координаты сом ламины и медуллы не используются: из них восстанавливается
    только одна ось из двух.
    """
    A_ = ann[ann["side"] == side]
    ret = retina_of(ann, side)
    mu, E, Az = sheet_axes(ret)
    X = ret[["pos_x", "pos_y", "pos_z"]].to_numpy(float) * VOXEL_NM - mu
    pos = {int(r): np.array([x @ E, x @ Az]) for r, x in zip(ret["root_id"], X)}

    lam = _wmean(_edges_to(con, A_.loc[A_["cell_class"] == "LA>ME", "root_id"], pos), pos)
    src1 = {**pos, **lam}
    med = _wmean(_edges_to(con, A_.loc[A_["cell_class"].isin(MEDULLA_CLASSES),
                                       "root_id"], src1), src1)
    src2 = {**med, **lam}
    vp_ids = A_.loc[A_["super_class"] == "visual_projection", "root_id"]
    vp = _wmean(_edges_to(con, vp_ids, src2), src2)
    return vp, (mu, E, Az), src2, set(int(x) for x in vp_ids)


def split_half(ann, con, side, seed=0):
    """Положение по случайной половине входов против второй половины.

    Если ось несёт сигнал, две независимые оценки совпадут. Если ось — шум,
    половины разойдутся. Элевация здесь положительный контроль метода: она
    обязана дать r > 0.9, иначе сломан расчёт, а не ось.

    Важная оговорка о том, что именно здесь проверяется. `field_positions`
    вызывается один раз: словарь `src` (положения ламины и медуллы) считается
    по ВСЕЙ связности и общий для обеих половин. Делятся пополам только рёбра
    последнего перехода `src -> vp`. То есть контроль в том виде, как он
    здесь написан, меряет устойчивость последнего усреднения (ламина/медулла
    -> проекционные), а не независимость всей цепочки
    сетчатка -> ламина -> медулла -> проекционные — общий `src` у половин
    контроль этого не проверяет.

    Три дополнительных прогона (сделаны при ревью, независимо от кода этого
    файла, числа приведены как получены) закрывают это сомнение:
    - «глубокое» разбиение, независимое на КАЖДОЙ ступени цепочки (отдельные
      случайные половины уже на переходах сетчатка->ламина и ламина->медулла,
      а не только на последнем): r_azim 0.938 (left) и 0.886 (right) против
      0.947 и 0.915 у обычного split_half; r_elev 0.974 и 0.987. Значит,
      общий `src` не завышает заявленную надёжность — «глубокая» версия даёт
      сопоставимые числа.
    - перестановочный контроль: значения `src` перемешаны между ключами
      (структура рёбер сохранена) — r_elev -0.013 и 0.000, r_azim 0.018 и
      0.012, то есть на перемешанных данных корреляция пропадает. Значит,
      измеренная корреляция отражает реальную пространственную структуру,
      а не артефакт процедуры усреднения.
    - разброс выведенных положений проекционных нейронов против разброса
      исходной сетчатки: по азимуту 89.9% (left) и 66.7% (right), по
      элевации 60.5% и 94.2%. Значит, положения проекционных нейронов не
      стянуты в точку общим `src`, а сохраняют содержательный разброс.
    """
    _, _, src, vp_ids = field_positions(ann, con, side)
    e = _edges_to(con, vp_ids, src)
    rng = np.random.default_rng(seed)
    m = rng.random(len(e)) < 0.5
    a, b = _wmean(e[m], src), _wmean(e[~m], src)
    both = sorted(set(a) & set(b))
    A_ = np.array([a[i] for i in both])
    B_ = np.array([b[i] for i in both])
    return {"side": side, "n": len(both),
            "r_elev": float(np.corrcoef(A_[:, 0], B_[:, 0])[0, 1]),
            "r_azim": float(np.corrcoef(A_[:, 1], B_[:, 1])[0, 1])}


def strips(vals, n=N_STRIPS):
    """Номер полосы по квантилям: полосы равны по населению, а не по координате.

    Покрытие листа неравномерное (отношение максимума к медиане расстояния до
    соседа 7-12), поэтому резать по координате нельзя — крайние полосы выйдут
    полупустыми.
    """
    vals = np.asarray(vals, float)
    q = np.quantile(vals, np.linspace(0.0, 1.0, n + 1)[1:-1])
    return np.searchsorted(q, vals).astype(int)


def _ommatidia_centroids():
    """Центры тяжести (строка, столбец) каждого из 721 омматидиев на карте flygym.

    В карте id=0 это фон, реальные омматидии пронумерованы 1..721; показание
    с индексом i соответствует id i+1 — проверяется ассертом здесь же, один
    раз на оба места, которые раньше дублировали этот блок построчно.
    """
    m = np.load(OMMATIDIA_MAP)
    ids = np.unique(m); ids = ids[ids > 0]
    assert ids.min() == 1 and ids.max() == 721 and len(ids) == 721, \
        f"неожиданная нумерация омматидиев: {ids.min()}..{ids.max()}, {len(ids)} шт"
    flat = m.ravel().astype(np.int64)
    yy, xx = np.divmod(np.arange(flat.size), m.shape[1])
    cnt = np.bincount(flat, minlength=722).astype(float)
    cy = np.bincount(flat, weights=yy, minlength=722)[ids] / cnt[ids]
    cx = np.bincount(flat, weights=xx, minlength=722)[ids] / cnt[ids]
    return ids, np.stack([cy, cx], axis=1)


def _eye_readout(xy):
    """Затемнение сетчатки при столбе в точке xy: форма (2, 721).

    Индекс 0 — левый глаз, индекс 1 — правый: это порядок самого
    `sim.get_ommatidia_readouts` (подтверждено измерением при ревью задачи 4),
    и он же используется как порядок строк в карте `ommatidia`.
    """
    from flyreplay import build_scene

    sim = build_scene(xy)
    sim.warmup(0.05)
    name = next(iter(sim.world.fly_lookup))
    v = sim.get_ommatidia_readouts(name).sum(axis=2)   # (2, 721)
    sim.close()
    return v


def measure_ommatidia_axis():
    """Какая ось картинки омматидиев отвечает за азимут и где у неё перёд — на каждый глаз отдельно.

    Постулировать нельзя: нумерация омматидиев в flygym — это пиксели
    отрендеренного глаза, и связь их осей с полем зрения нигде не объявлена.
    Меряем раздельно для левого и правого глаза: глаза зеркальны по азимуту
    (ревью задачи 4 измерило: left flip=False, right flip=True при одинаковой
    оси), общая на два глаза ориентация меняла бы «перёд» на «зад» на одной
    из сторон. У каждого глаза своя боковая сцена — столб под глазом снаружи,
    иначе он окажется за головой и второй глаз даст нулевое затемнение.
    Ставим столб прямо по курсу и сбоку, смотрим, какие омматидии темнеют, и
    берём ту ось, вдоль которой центры тяжести затемнения разошлись сильнее.
    Знак — по тому, куда сместился «перёд».
    """
    empty = _eye_readout((500.0, 500.0))            # FAR_AWAY: столб унесён
    front = empty - _eye_readout((6.0, 0.0))         # общая сцена для обоих глаз
    side_xy = ((0.0, 6.0), (0.0, -6.0))              # (левому глазу, правому глазу)
    sidew = np.stack([empty[eye] - _eye_readout(side_xy[eye])[eye] for eye in (0, 1)])

    _, cent = _ommatidia_centroids()

    def com(d):
        w = np.clip(d, 0, None)
        return (cent * w[:, None]).sum(axis=0) / max(w.sum(), 1e-9), w.sum()

    names = ("left", "right")
    result = {}
    for eye in (0, 1):
        cf, wf = com(front[eye])
        cs, ws = com(sidew[eye])
        assert wf > 0, f"{names[eye]}: столб впереди не затемнил этот глаз"
        assert ws > 0, f"{names[eye]}: столб сбоку не затемнил этот глаз"
        shift = cf - cs
        axis = int(np.argmax(np.abs(shift) / cent.std(axis=0)))
        flip = bool(shift[axis] < 0)      # перёд должен получить БОЛЬШИЙ номер полосы
        result[names[eye]] = {
            "axis": axis, "flip": flip,
            "contrast": float(abs(shift[axis]) / cent[:, axis].std()),
            "weight_front": float(wf), "weight_side": float(ws),
        }
    return result


def ommatidia_strips(axis, flip, n=N_STRIPS):
    """Номер полосы для каждого из 721 омматидия одного глаза по его оси/перевороту."""
    _, cent = _ommatidia_centroids()
    c = cent[:, axis]
    return strips(-c if flip else c, n)


def load_or_build():
    """Карта целиком. Кэш — производное от аннотаций, в git не идёт."""
    if CACHE.exists():
        z = np.load(CACHE)
        return {k: (str(z[k]) if k == "axis" else z[k]) for k in z.files}
    flyid2i, ann, con = load_tables()      # один раз: parquet на 15 млн строк
    o = measure_ommatidia_axis()           # {"left": {...}, "right": {...}} — раздельно по глазам
    om = np.stack([ommatidia_strips(o[eye]["axis"], o[eye]["flip"])
                   for eye in ("left", "right")])      # (2, 721): 0 левый глаз, 1 правый
    res = {"axis": MAP_AXIS, "ommatidia": om,
           "om_axis": np.array([o["left"]["axis"], o["right"]["axis"]]),
           "om_flip": np.array([o["left"]["flip"], o["right"]["flip"]]),
           "om_contrast": np.array([o["left"]["contrast"], o["right"]["contrast"]])}
    col = 0 if MAP_AXIS == "elevation" else 1
    for side in ("left", "right"):
        vp, _, _, _ = field_positions(ann, con, side)
        keys = np.array(sorted(vp))
        vals = np.array([vp[k][col] for k in keys])
        res[f"{side}_idx"] = np.array([flyid2i[int(k)] for k in keys], dtype=np.int64)
        res[f"{side}_strip"] = strips(vals)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, **res)
    return res


def self_check() -> None:
    """Ломается, если оси листа определены неверно."""
    _, ann, con = load_tables()
    for side in ("left", "right"):
        ret = retina_of(ann, side)
        mu, E, A = sheet_axes(ret)
        assert abs(np.linalg.norm(E) - 1) < 1e-9 and abs(np.linalg.norm(A) - 1) < 1e-9
        assert abs(E @ A) < 1e-9, f"{side}: оси листа не ортогональны"
        assert E[DORSAL_AXIS] * DORSAL_SIGN > 0, \
            f"{side}: элевация по кайме DRA смотрит не туда, куда оцелли"
        assert A[ANTERIOR_AXIS] * ANTERIOR_SIGN > 0, \
            f"{side}: азимут смотрит не в сторону антенн"
        X = ret[["pos_x", "pos_y", "pos_z"]].to_numpy(float) * VOXEL_NM - mu
        dra = ret["cell_sub_class"].to_numpy() == "DRA"
        assert dra.sum() > 50, f"{side}: каймы DRA всего {dra.sum()} нейронов"
        pe = (X @ E)
        pct = float((pe < pe[dra].mean()).mean() * 100)
        assert pct > 75, f"{side}: кайма DRA не у дорсального края, процентиль {pct:.0f}"
        print(f"  {side}: DRA на {pct:.0f}-м процентиле по элевации")

    rows = [split_half(ann, con, s, seed=0) for s in ("left", "right")]
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(out("visual_field_split_half.csv"), index=False)
    assert df["r_elev"].min() > 0.9, (
        f"положительный контроль провален: элевация {df['r_elev'].min():.3f} < 0.9. "
        "Сломан расчёт, а не ось — чинить расчёт, в запасной вариант не уходить")
    axis = "azimuth" if df["r_azim"].min() > 0.5 else "elevation"
    print(f"ось карты: {axis} (азимут {df['r_azim'].min():.3f}, порог 0.5)")

    print("самопроверка осей листа: ОК")

    m = load_or_build()
    om = m["ommatidia"]
    assert om.shape == (2, 721), \
        f"карта омматидиев должна быть (2, 721) — по глазу на строку, получено {om.shape}"
    eyes = ("left", "right")
    for eye, name in enumerate(eyes):
        c = np.bincount(om[eye], minlength=N_STRIPS)
        assert c.min() * 2 >= c.max(), \
            f"{name}: полосы поля зрения слишком неравны: {c.tolist()}"
    for side in ("left", "right"):
        idx, st = m[f"{side}_idx"], m[f"{side}_strip"]
        assert len(idx) == len(st) and len(idx) == len(set(idx.tolist()))
        cs = np.bincount(st, minlength=N_STRIPS)
        assert cs.min() * 2 >= cs.max(), f"{side}: полосы листа неравны: {cs.tolist()}"
        assert cs.sum() > 3800, f"{side}: покрыто всего {cs.sum()} проекционных"
    assert not (set(m["left_idx"].tolist()) & set(m["right_idx"].tolist())), \
        "левые и правые проекционные пересеклись"
    print(f"карта: ось {m['axis']}, полос {N_STRIPS}, "
          f"проекционных {len(m['left_idx'])}/{len(m['right_idx'])}")

    for eye, name in enumerate(eyes):
        contrast = float(m["om_contrast"][eye])
        assert contrast > 0.3, (
            f"{name}: столб впереди и сбоку затемняют почти одно и то же место "
            f"сетчатки (контраст {contrast:.2f}) — ориентация не определена")
        print(f"  ориентация омматидиев {name}: ось {int(m['om_axis'][eye])}, "
              f"переворот {bool(m['om_flip'][eye])}, контраст {contrast:.2f}")

    # Главный инвариант (нашло ревью задачи 4): «перёд» обязан попадать в одну
    # и ту же полосу у обоих глаз. Раньше единая ось/переворот на оба глаза
    # приводили к тому, что номер полосы означал противоположные физические
    # направления слева и справа — карта была перевёрнута перёд-зад на одной
    # из сторон, хотя со стороны коннектома «перёд» у обоих полушарий один.
    front = _eye_readout((500.0, 500.0)) - _eye_readout((6.0, 0.0))
    peak = [int(np.argmax(np.bincount(om[eye], weights=np.clip(front[eye], 0, None),
                                       minlength=N_STRIPS)))
            for eye in (0, 1)]
    assert peak[0] == peak[1], (
        f"«перёд» попадает в разные полосы у двух глаз: left -> полоса {peak[0]}, "
        f"right -> полоса {peak[1]} — карта омматидиев рассинхронизирована между глазами")
    print(f"  инвариант «перёд»: left -> полоса {peak[0]}, right -> полоса {peak[1]}")


if __name__ == "__main__":
    self_check()
