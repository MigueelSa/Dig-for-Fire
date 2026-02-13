import os, sys
from typing import Any

def resource_path(*parts: Any) -> str:
    if getattr(sys, "frozen", False):
        base_dir = sys._MEIPASS
    else:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

    return os.path.join(base_dir, *parts)

def output_path(*parts: Any) -> str:
    if getattr(sys, "frozen", False):
        base_dir = os.path.dirname(os.path.abspath(os.path.join(sys.executable, "..")))
    else:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

    return os.path.join(base_dir, *parts)
