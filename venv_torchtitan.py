from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import site
import sys
from types import ModuleType


def _site_package_candidates() -> list[str]:
    candidates: list[str] = []
    for path in site.getsitepackages():
        if path and path not in candidates:
            candidates.append(path)
    user_site = site.getusersitepackages()
    if user_site and user_site not in candidates:
        candidates.append(user_site)
    return candidates


def ensure_venv_torchtitan() -> ModuleType:
    if "torchtitan" in sys.modules:
        module = sys.modules["torchtitan"]
        module_path = getattr(module, "__file__", "")
        if module_path and (
            "site-packages" in module_path or "dist-packages" in module_path
        ):
            return module

        for name in list(sys.modules.keys()):
            if name == "torchtitan" or name.startswith("torchtitan."):
                sys.modules.pop(name, None)

    candidates = _site_package_candidates()
    for path in candidates:
        if os.path.isdir(os.path.join(path, "torchtitan")):
            spec = importlib.machinery.PathFinder.find_spec("torchtitan", [path])
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules["torchtitan"] = module
                spec.loader.exec_module(module)
                return module

    raise ModuleNotFoundError(
        "Unable to import torchtitan from site-packages. "
        "Install torchtitan in the active venv or update your PYTHONPATH."
    )
