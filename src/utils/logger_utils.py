'''
Author: 13594053100@163.com
Date: 2025-09-30 09:42:45
LastEditTime: 2025-09-30 09:42:51
'''

import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

# ===================== 彩色日志格式 =====================

COLOR_MAP = {
    "DEBUG": "\033[90m",     # 灰色
    "INFO": "\033[92m",      # 绿色
    "WARNING": "\033[93m",   # 黄色
    "ERROR": "\033[91m",     # 红色
    "CRITICAL": "\033[95m",  # 洋红色
}
RESET = "\033[0m"


class ColorFormatter(logging.Formatter):
    """带颜色的控制台日志格式化器"""

    def format(self, record: logging.LogRecord) -> str:
        color = COLOR_MAP.get(record.levelname, "")
        message = super().format(record)
        return f"{color}{message}{RESET}"


# ===================== 日志工具函数 =====================

def get_logger(
    name: str,
    level=logging.INFO,
    file_name: str | None = None,
    color_console: bool = True,
    *,
    log_root: str | Path = "logs",   # 日志根目录
    rotate: str = "size",            # "size" | "time" | "off"
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 10,
    when: str = "midnight",
    interval: int = 1,
    utc: bool = False
) -> logging.Logger:
    """
    获取独立 logger：控制台彩色；文件纯文本（可选）；支持日志轮转，日志文件自动按日期分目录。

    Args:
        name (str): logger 名称
        level (int): 日志级别
        file_name (str|None): 日志文件名（如 "app.log"）；None 则不写文件
        color_console (bool): 控制台是否彩色
        log_root (str|Path): 日志根目录，默认 logs/
        rotate (str): "size" 按大小轮转；"time" 按时间轮转；"off" 不轮转
        max_bytes (int): 按大小轮转阈值
        backup_count (int): 保留的日志文件数
        when (str): 时间轮转单位
        interval (int): 时间轮转间隔
        utc (bool): 是否按 UTC 时间轮转
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        # 控制台 Handler
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(ColorFormatter(fmt) if color_console else logging.Formatter(fmt))
        logger.addHandler(sh)

        # 文件 Handler
        if file_name:
            date_dir = datetime.now().strftime("%Y-%m-%d")  # 当天日期目录
            log_dir = Path(log_root) / date_dir
            log_dir.mkdir(parents=True, exist_ok=True)
            file_path = log_dir / file_name

            if rotate == "size":
                fh = RotatingFileHandler(
                    file_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
                )
            elif rotate == "time":
                fh = TimedRotatingFileHandler(
                    file_path, when=when, interval=interval,
                    backupCount=backup_count, encoding="utf-8", utc=utc
                )
            else:  # "off"
                fh = logging.FileHandler(file_path, encoding="utf-8")

            fh.setFormatter(logging.Formatter(fmt))
            logger.addHandler(fh)

    logger.propagate = False
    return logger