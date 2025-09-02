"""
日志设置模块。
"""

import logging
from typing import Optional, List

def setup_logging_original_fix(verbosity: int = 1, log_file: Optional[str] = None):
    """
    配置全局日志记录器。
    
    Args:
        verbosity (int): 日志详细级别 (0=WARN, 1=INFO, 2=DEBUG).
        log_file (str, optional): 日志输出文件路径.
    """
    level_map = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG, 3: logging.DEBUG}
    level = level_map.get(verbosity, logging.INFO)
    fmt = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    datefmt = "%Y-%m-%dT%H:%M:%S"
    
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers, force=True)
