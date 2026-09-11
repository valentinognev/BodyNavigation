import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[3] / "Python" / "tools"
if _TOOLS.is_dir() and str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))
