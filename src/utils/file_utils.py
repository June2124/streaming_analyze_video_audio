'''
Author: 13594053100@163.com
Date: 2025-09-30 09:45:32
LastEditTime: 2025-09-30 09:45:34
'''

from __future__ import annotations
import logging
import mimetypes
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, List
from urllib.parse import unquote, urlparse

import requests
from src.utils.logger_utils import get_logger

logger = get_logger(name="FileUtils", level=logging.DEBUG)


class FileUtils:
    """文件操作与下载工具类"""

    @staticmethod
    def collect_file_paths_from_folder(
        folder_path: str,
        extensions: List[str] = [".jpg", ".jpeg", ".png", ".bmp", ".svg"],
    ) -> list[str]:
        """
        收集指定文件夹下指定格式的所有文件路径。

        Args:
            folder_path: 关键帧所在文件夹路径
            extensions: 支持的格式后缀列表，默认常见图片格式
        Returns:
            图片文件路径列表（按文件名排序）
        """
        if not os.path.isdir(folder_path):
            raise ValueError(f"指定路径不是有效文件夹: {folder_path}")

        file_paths_list = []
        for filename in os.listdir(folder_path):
            if any(filename.lower().endswith(ext) for ext in extensions):
                full_path = os.path.join(folder_path, filename)
                file_paths_list.append(full_path)

        file_paths_list.sort()
        return file_paths_list

    @staticmethod
    def sanitize_tag(tag: str, illegal: str) -> str:
        """
        清理标签字符串，替换非法字符为下划线。

        Args:
            tag: 原始标签字符串
            illegal: 非法字符集
        Returns:
            清理后的标签字符串
        """
        tag = (tag or "").strip()
        return "".join("_" if ch in illegal else ch for ch in tag)

    @staticmethod
    def download_resource(url: str) -> Tuple[str, Dict[str, Any]]:
        """
        同步下载并验证资源文件。
        支持 http/https 和 file:// 协议。
        Windows|Linux|MacOS的本地文件都必须以 file:/// 或 file:// 开头。

        Args:
            url: 资源地址 (http/https/file://)

        Returns:
            (file_path, file_info)
            file_path: 下载或复制后的本地文件完整路径
            file_info: {
                "file_name": 文件名,
                "file_size": 文件大小（字节数）,
                "content_type": MIME 类型,
                "download_time": 下载/复制完成时间
            }

        Raises:
            RuntimeError: HTTP 下载失败或响应异常
            FileNotFoundError: 指定的本地文件不存在
            ValueError: URL 协议不支持
        """
        logger.info("[资源下载] 接收到 URL: %s", url)
        temp_dir = Path("storage/temp/resource_downloads")
        temp_dir.mkdir(parents=True, exist_ok=True)

        parsed = urlparse(url)
        logger.info("[资源下载] URL 解析结果 scheme=%s, path=%s", parsed.scheme, parsed.path)

        # ---------- http/https ----------
        if parsed.scheme in ("http", "https"):
            raw_name = url.split("/")[-1] or f"resource_{uuid.uuid4().hex[:8]}"
            if "?" in raw_name:
                raw_name = raw_name.split("?")[0]
            file_name = unquote(raw_name) or f"resource_{uuid.uuid4().hex[:8]}"
            file_path = temp_dir / file_name

            logger.info("[资源下载] 开始 HTTP 下载: %s", url)
            with requests.get(url, stream=True, timeout=300) as resp:
                if resp.status_code != 200:
                    raise RuntimeError(f"文件下载失败: HTTP {resp.status_code}")

                content_type = resp.headers.get("content-type", "application/octet-stream")
                total_size = 0
                with open(file_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if not chunk:
                            continue
                        f.write(chunk)
                        total_size += len(chunk)

            file_info = {
                "file_name": file_name,
                "file_size": total_size,
                "content_type": content_type,
                "download_time": datetime.now().isoformat(),
            }
            logger.info("[资源下载] HTTP 下载完成: %s (%d bytes)", file_name, total_size)
            return str(file_path), file_info

        # ---------- file:// ----------
        elif parsed.scheme == "file":
            # 解析本地路径（跨平台处理 Windows 盘符与 UNC）
            netloc = parsed.netloc  # Windows UNC 的 server 名或空
            local_path = unquote(parsed.path or "")

            if os.name == "nt":
                if netloc:
                    # UNC: file://server/share/folder/file.mp4
                    # parsed.path 通常是 "/share/folder/file.mp4"
                    # 组装为 \\server\share\folder\file.mp4
                    local_path = local_path.lstrip("/\\")
                    local_path = rf"\\{netloc}\{local_path}"
                else:
                    # 盘符: file:///D:/path -> "/D:/path" 需要去掉前导 "/"
                    if local_path.startswith("/") and len(local_path) > 3 and local_path[2] == ":":
                        local_path = local_path[1:]
                local_path = os.path.normpath(local_path)
            else:
                # POSIX: 直接当绝对路径
                if not local_path.startswith("/"):
                    local_path = "/" + local_path
                local_path = os.path.normpath(local_path)

            if not os.path.exists(local_path) or not os.path.isfile(local_path):
                raise FileNotFoundError(f"本地文件不存在: {local_path}")

            file_name = os.path.basename(local_path) or f"resource_{uuid.uuid4().hex[:8]}"
            file_path = temp_dir / file_name

            total_size = 0
            with open(local_path, "rb") as src, open(file_path, "wb") as dst:
                while True:
                    chunk = src.read(8192)
                    if not chunk:
                        break
                    dst.write(chunk)
                    total_size += len(chunk)

            content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
            file_info = {
                "file_name": file_path.name,
                "file_size": total_size,
                "content_type": content_type,
                "download_time": datetime.now().isoformat(),
            }
            logger.info("[资源下载] 本地复制完成: %s (%d bytes)", file_path.name, total_size)
            return str(file_path), file_info