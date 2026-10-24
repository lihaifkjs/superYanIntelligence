"""兼容 shim：原 pipeline_common 已拆为 prompts / judge / llm_io 三个模块。

新代码请按需直接 import 子模块（build_sft 只 import judge 就不会拖 torch）。
本文件仅保证旧 import 方式不炸。
"""

from judge import *  # noqa: F401,F403
from llm_io import *  # noqa: F401,F403
from prompts import *  # noqa: F401,F403
