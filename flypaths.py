"""Единая точка настройки путей проекта.

Ничего не зашито под конкретную машину: всё берётся из переменных окружения,
а если их нет — из расположения самого репозитория и домашнего каталога.

    FLY_PROJECT_DIR   корень репозитория (по умолчанию — папка этого файла)
    FLY_BRAIN_CODE    каталог с кодом LIF-модели (по умолчанию <проект>/fly-brain/code)
    FLY_BRAIN_DATA    каталог с данными коннектома (по умолчанию ~/fly-brain-data)

Данные коннектома тяжёлые (~763 МБ) и в git не входят. Их держат в нативной
файловой системе, а не на смонтированном диске: 9p-mount под WSL примерно втрое
медленнее на файловом I/O, и это заметно на времени загрузки весов.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_DIR = Path(os.environ.get("FLY_PROJECT_DIR") or Path(__file__).resolve().parent)
FLY_BRAIN_CODE = Path(os.environ.get("FLY_BRAIN_CODE") or PROJECT_DIR / "fly-brain" / "code")


def _data_dir() -> Path:
    """Где лежат данные коннектома: переменная, потом ~/fly-brain-data, потом репозиторий.

    Домашний каталог проверяется первым, чтобы не сломать машины, где данные уже
    разложены по описанному выше правилу. На чистой установке его нет — setup.py
    кладёт всё в fly-brain/data, туда же смотрит и benchmark.py.
    """
    if env := os.environ.get("FLY_BRAIN_DATA"):
        return Path(env)
    home = Path.home() / "fly-brain-data"
    return home if home.is_dir() else PROJECT_DIR / "fly-brain" / "data"


DATA_DIR = _data_dir()
OUTPUT_DIR = PROJECT_DIR / "output"

# Бэкенд offscreen-рендера mujoco. egl бывает только на Linux/WSL, на macOS
# его нет вовсе и mujoco падает прямо на импорте; cgl работает без окна.
# Ставится здесь, до того как любой скрипт проекта импортирует flygym.
os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")

# Аннотации типов клеток (Schlegel et al., Nature 2024). В git не входят,
# скачиваются отдельно — см. tools/fetch_annotations.md
ANNOTATIONS = Path(
    os.environ.get("FLY_ANNOTATIONS") or DATA_DIR / "annotations" / "neuron_annotations.tsv"
)


def add_fly_brain_to_path() -> None:
    """Сделать импортируемыми benchmark.py и run_pytorch.py из fly-brain."""
    p = str(FLY_BRAIN_CODE)
    if p not in sys.path:
        sys.path.insert(0, p)


def out(name: str) -> str:
    """Путь к файлу результата, каталог создаётся при необходимости."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return str(OUTPUT_DIR / name)


def require(path: Path, hint: str) -> Path:
    """Понятная ошибка вместо невнятного traceback, если чего-то нет на месте."""
    if not path.exists():
        raise FileNotFoundError(f"не найдено: {path}\n{hint}")
    return path
