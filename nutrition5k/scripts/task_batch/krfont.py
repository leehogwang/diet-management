"""Register a Korean-capable font for matplotlib (Noto Sans CJK KR)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

_CAND = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
]


def setup():
    name = None
    for p in _CAND:
        if Path(p).exists():
            try:
                fm.fontManager.addfont(p)
                name = fm.FontProperties(fname=p).get_name()
            except Exception:
                pass
    if name:
        plt.rcParams["font.family"] = name
    plt.rcParams["axes.unicode_minus"] = False
    return name


if __name__ == "__main__":
    print("registered:", setup())
