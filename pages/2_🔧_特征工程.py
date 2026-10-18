import runpy
import sys
from pathlib import Path

# 根目录 pages/ 转发器：把同名页面转发到 app/pages/ 下运行
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
runpy.run_path(str(ROOT / "app" / "pages" / Path(__file__).name), run_name="__main__")
