from importlib.resources import files
from pathlib import Path


def get_project_root():
    return str(Path.cwd())


def get_abs_path(relative_path):
    path = Path(relative_path)
    if not path.is_absolute() and path.parts and path.parts[0] in {"prompts", "config"}:
        return str(files("evidenceloop.resources").joinpath(path.as_posix()))
    return str(path.expanduser().resolve())


def relation_map_path(name):
    return Path(str(files("evidenceloop.resources").joinpath("config", name)))
