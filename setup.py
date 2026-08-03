#!/usr/bin/env python3
"""Установка проекта на чистую машину: macOS (Apple Silicon / Intel) и Linux.

    python3 setup.py                      полная установка
    python3 setup.py --skip-weights       без прогрева весов мозга (~580 МБ, минуты)
    python3 setup.py --skip-check         без финальной проверки тела

Что делается по шагам:

  1. Интерпретатор. flygym требует CPython 3.12–3.14; если текущий не подходит,
     скрипт сам перезапускается на найденном подходящем.
  2. `.venv` в корне проекта + flygym (тело NeuroMechFly v2) и torch (мозг).
  3. `fly-brain` — код LIF-модели и данные коннектома FlyWire v783
     (138 639 нейронов, 15 091 983 связи), ~180 МБ архивом.
  4. Аннотации типов клеток Schlegel et al. — ~30 МБ, без них не работают
     tools/visual_to_dn.py, dn_input_composition.py, analyze_annotations.py.
  5. Проверка окружения: flypaths.py сам находит данные в fly-brain/data и сам
     выбирает бэкенд рендера (egl на Linux/WSL, cgl на macOS — egl там нет вовсе,
     mujoco падает прямо на импорте). Экспортировать переменные не нужно.
  6. Прогрев весов: weight_coo.pkl + weight_csr.pkl (~580 МБ) строятся один раз
     из parquet, дальше все прогоны берут готовое.
  7. Проверка: tools/body_walk_check.py — реальная ходьба в MuJoCo с критериями.

Всё тяжёлое кладётся внутрь проекта (`fly-brain/data`), пути пробрасываются
переменными окружения, домашний каталог не трогается.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
VENV = PROJECT / ".venv"
BRAIN = PROJECT / "fly-brain"
DATA = BRAIN / "data"

PY_MIN, PY_MAX = (3, 12), (3, 14)  # диапазон flygym: >=3.12,<3.15

# Тело: flygym 2.x даёт и flygym.compose, и flygym_demo.complex_terrain
# (CPG, HybridTurningController) — оба нужны, оба в одном колесе с PyPI.
# torch/pandas/pyarrow — для мозга: разреженная матрица весов и чтение parquet.
REQUIREMENTS = ["flygym==2.1.0", "torch", "pandas<3", "pyarrow"]

FLY_BRAIN_ZIP = "https://codeload.github.com/eonsystemspbc/fly-brain/zip/refs/heads/main"
ANNOTATIONS_URL = (
    "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/"
    "main/supplemental_files/Supplemental_file1_neuron_annotations.tsv"
)

WARMUP = """
import sys
sys.path.insert(0, {code!r})
from benchmark import path_comp, path_con, path_wt
import run_pytorch as rp
w = rp.get_weights(str(path_con), str(path_comp), str(path_wt), csr=True)
print("веса мозга:", tuple(w.shape), "ненулевых связей:", w.values().numel())
"""


def step(msg: str) -> None:
    print(f"\n=== {msg}", flush=True)


def run(*cmd: object) -> None:
    print("$", " ".join(map(str, cmd)), flush=True)
    try:
        subprocess.run([str(c) for c in cmd], check=True)
    except subprocess.CalledProcessError as exc:
        sys.exit(f"\nшаг не прошёл (код {exc.returncode}), причина выше")


def find_python(minor: int) -> str | None:
    """python3.<minor> в PATH или там, где его кладут типовые установщики.

    На чистой macOS `python3` — это 3.9 из Command Line Tools, а свежий питон
    от Homebrew или python.org в PATH может и не попасть.
    """
    name = f"python3.{minor}"
    if exe := shutil.which(name):
        return exe
    known = [
        Path(f"/opt/homebrew/bin/{name}"),                                  # brew, Apple Silicon
        Path(f"/usr/local/bin/{name}"),                                     # brew, Intel
        Path(f"/Library/Frameworks/Python.framework/Versions/3.{minor}/bin/{name}"),  # python.org
        *sorted(Path.home().glob(f".pyenv/versions/3.{minor}.*/bin/{name}")),         # pyenv
    ]
    return next((str(p) for p in known if p.exists()), None)


def ensure_python() -> None:
    """Перезапуститься на подходящем интерпретаторе, если текущий не подходит."""
    if PY_MIN <= sys.version_info[:2] <= PY_MAX:
        return
    for minor in range(PY_MIN[1], PY_MAX[1] + 1):
        exe = find_python(minor)
        if exe:
            print(f"текущий Python {sys.version.split()[0]} не подходит, "
                  f"перезапуск на {exe}")
            os.execv(exe, [exe, str(Path(__file__).resolve()), *sys.argv[1:]])
    sys.exit(
        f"нужен CPython 3.12–3.14 (ограничение flygym), сейчас {sys.version.split()[0]}\n"
        "  macOS:  brew install python@3.13\n"
        "  Linux:  apt install python3.13-venv  (или pyenv install 3.13)"
    )


def venv_python() -> Path:
    return VENV / "bin" / "python"


def make_venv() -> None:
    step("окружение .venv")
    if not venv_python().exists():
        run(sys.executable, "-m", "venv", VENV)
    run(venv_python(), "-m", "pip", "install", "--upgrade", "pip")
    run(venv_python(), "-m", "pip", "install", *REQUIREMENTS)


def download(url: str, dest: Path, label: str) -> Path:
    if dest.exists():
        print(f"  {label}: уже на месте")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url) as resp, open(part, "wb") as fh:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
            done += len(chunk)
            of = f" из {total >> 20}" if total else ""
            print(f"\r  {label}: {done >> 20}{of} МБ", end="", flush=True)
    print()
    part.rename(dest)
    return dest


def fetch_fly_brain() -> None:
    step("мозг: код LIF-модели + коннектом FlyWire v783")
    if (BRAIN / "code" / "benchmark.py").exists():
        print("  fly-brain: уже на месте")
        return
    # tmp рядом с проектом, а не в /tmp: 380 МБ архива и распаковки могут не влезть
    # в tmpfs, а на одной ФС с целью move — это переименование, а не копирование
    with tempfile.TemporaryDirectory(dir=PROJECT) as tmp:
        archive = download(FLY_BRAIN_ZIP, Path(tmp) / "fly-brain.zip",
                           "fly-brain (~180 МБ)")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(tmp)
        # по одному элементу, а не всю папку: если fly-brain уже есть (оборванная
        # прошлая попытка), shutil.move вложил бы её внутрь — fly-brain/fly-brain-main
        BRAIN.mkdir(exist_ok=True)
        for item in next(Path(tmp).glob("fly-brain-*")).iterdir():
            if not (BRAIN / item.name).exists():
                shutil.move(str(item), str(BRAIN / item.name))
    print(f"  распаковано в {BRAIN}")


def fetch_annotations() -> None:
    step("аннотации типов клеток (Schlegel et al., Nature 2024)")
    download(ANNOTATIONS_URL, DATA / "annotations" / "neuron_annotations.tsv",
             "аннотации (~30 МБ)")


def verify_env() -> None:
    """Проверить, что пути и бэкенд рендера сходятся, до того как что-то считать.

    Раньше это делалось файлом `.pth` внутри venv. На macOS оказалось нерабочим:
    свежесозданным файлам под ~/Documents прилетает флаг UF_HIDDEN, а site.py
    такие .pth молча пропускает («Skipping hidden .pth file»). Окружение выглядело
    настроенным, а на деле mujoco падал на 'invalid value ... MUJOCO_GL: egl'.
    Теперь всё решает flypaths.py в самом репозитории — подставлять нечего.
    """
    step("проверка окружения")
    for junk in ("_flyenv.py", "_flyenv.pth", "sitecustomize.py"):  # от прежних версий setup.py
        (next(VENV.glob("lib/python3.*/site-packages")) / junk).unlink(missing_ok=True)
    probe = (
        f"import sys; sys.path.insert(0, {str(PROJECT)!r})\n"
        "import os, flypaths\n"
        "print(flypaths.DATA_DIR, os.environ['MUJOCO_GL'], sep='\\n')\n"
    )
    res = subprocess.run([str(venv_python()), "-c", probe],
                         capture_output=True, text=True)
    if res.returncode:
        sys.exit(f"окружение не собралось:\n{res.stderr}")
    data_dir, backend = res.stdout.split()
    print(f"  данные коннектома: {data_dir}")
    print(f"  бэкенд рендера:    {backend}")
    if not (Path(data_dir) / "2025_Connectivity_783.parquet").exists():
        sys.exit(f"в {data_dir} нет 2025_Connectivity_783.parquet")


def warm_weights() -> None:
    step("прогрев весов мозга (первый раз — минуты и ~580 МБ на диск)")
    run(venv_python(), "-c", WARMUP.format(code=str(BRAIN / "code")))


def check() -> None:
    step("проверка: тело идёт, стоит и поворачивает")
    run(venv_python(), PROJECT / "tools" / "body_walk_check.py")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--skip-weights", action="store_true",
                    help="не строить кэш весов (построится при первом прогоне)")
    ap.add_argument("--skip-check", action="store_true",
                    help="не гонять финальную проверку тела")
    args = ap.parse_args()

    ensure_python()
    make_venv()
    fetch_fly_brain()
    fetch_annotations()
    verify_env()
    if not args.skip_weights:
        warm_weights()
    if not args.skip_check:
        check()

    py = venv_python().relative_to(Path.cwd()) if venv_python().is_relative_to(Path.cwd()) else venv_python()
    print(f"""
=== готово

  {py} tools/p9_tuning_curve.py
  {py} closed_loop_vision.py --cycles 200 --pillar-y 3 --tag left --video
  {py} closed_loop_v2.py --cycles 210 --tau 100 --fb-base 20 --fb-span 180 --perturb --video

Пути и бэкенд рендера скрипты выбирают сами (flypaths.py), экспортировать ничего не нужно.
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
