import sys
from pathlib import Path

# ModelScope 创空间入口：把根目录 app.py 转发到 app/streamlit_app.py（首页）
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.streamlit_app import *  # noqa: F401,F403
