import os, sys
from typing import Any

def resource_path(*parts: Any) -> str:
    if getattr(sys, "frozen", False):
        base = sys._MEIPASS
    else:
        base = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))

    return os.path.join(base, *parts)

def output_path(*parts: Any) -> str:
    if getattr(sys, "frozen", False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(base_dir, *parts)
