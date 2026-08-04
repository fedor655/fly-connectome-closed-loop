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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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


if __name__ == "__main__":
    self_check()
