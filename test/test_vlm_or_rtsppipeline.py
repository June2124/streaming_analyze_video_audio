# -*- coding: utf-8 -*-
"""
仅测试：主控 -> A -> B -> OUT-VLM
- 强制启用 B（VLM），禁用 C（ASR）
- 通过回调直接打印 VLM 增量/完成结果
- 结束时打印 snapshot_stats() 与 skew_guard 的统计（如启用）
"""

from __future__ import annotations
import os, sys, time, threading, logging

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from streaming_analyze import StreamingAnalyze
from src.all_enum import MODEL  # 你当前的枚举是 OFFLINE / ONLINE / SECURITY

# ====== 简单日志配置（INFO 级别，便于看到主控统计与监控输出）======
def _setup_logging():
    root = logging.getLogger()
    if not root.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root.addHandler(h)
    root.setLevel(logging.INFO)

def run_test_ab_vlm(url: str, *, slice_sec: int = 5, max_secs: int | None = None):
    """
    仅测试：主控 -> A -> B -> OUT-VLM
    - B 开启（enable_b=True），C 关闭（enable_c=False）
    - 使用 MODEL.OFFLINE（离线文件场景）；如果你换 RTSP 也可复用这套测试
    - max_secs：可选 watchdog，适合 RTSP 或长任务；离线文件通常不需要
    """
    # 你的 MODEL 目前是：OFFLINE / ONLINE / SECURITY
    model = MODEL.OFFLINE

    runner = StreamingAnalyze(
        url=url,
        mode=model,
        slice_sec=slice_sec,
        enable_b=True,   # 开启 B (VLM)
        enable_c=False   # 关闭 C (ASR)
    )

    # ---- 直出到终端的回调 ----
    def on_vlm_delta(payload: dict):
        seg = payload.get("segment_index")
        seq = payload.get("seq")
        delta = (payload.get("delta") or "").strip()
        if delta:
            print(f"[TEST][VLMΔ seg#{seg} seq={seq}] {delta}", flush=True)

    def on_vlm_done(payload: dict):
        seg = payload.get("segment_index")
        text = (payload.get("full_text") or "").strip()
        print(f"[TEST][VLM✓ seg#{seg}] {text}", flush=True)

    runner.on_vlm_delta = on_vlm_delta
    runner.on_vlm_done  = on_vlm_done

    # ---- 可选：最长运行时间 watchdog（RTSP/长任务用；文件流通常 A 结束后会自然收尾）----
    stop_flag = threading.Event()
    def watchdog():
        if max_secs and max_secs > 0:
            t0 = time.time()
            while not stop_flag.is_set():
                if time.time() - t0 >= max_secs:
                    print(f"[TEST] 达到 {max_secs}s，触发优雅停止。", flush=True)
                    try:
                        runner.force_stop("timeout")
                    except Exception:
                        pass
                    break
                time.sleep(0.2)

    wd = None
    if max_secs and max_secs > 0:
        wd = threading.Thread(target=watchdog, daemon=True)
        wd.start()

    try:
        runner.start_streaming_analyze()  # 阻塞直到流程自然结束或被 force_stop
    finally:
        stop_flag.set()
        # ---- 结束时打印主控统计快照 ----
        try:
            stats = runner.snapshot_stats()  # 只启 B 时只会有 "vlm" 字段
            print("[TEST][STATS]", stats, flush=True)
        except Exception:
            pass

        # ---- 结束时打印 skew_guard 汇总（若启用）----
        if getattr(runner, "skew_guard", None):
            try:
                # 你的 skew_guard 版本我们实现了 snapshot_stats()
                print("[TEST][SKEW]", runner.skew_guard.snapshot_stats(), flush=True)
            except Exception:
                pass

        print("[TEST] 任务结束。", flush=True)


if __name__ == "__main__":
    _setup_logging()

    # === 示例 1：本地视频文件（A 切完片后自然结束）
    run_test_ab_vlm(
        url=r"D:\streaming_analyze_video_audio\static\video\RAG_video_no_sound_test.mp4",
        slice_sec=5,
        max_secs=None,  # 文件流通常不需要
    )

    # === 示例 2：RTSP 实时流（最多跑 30 秒后优雅停止）
    # run_test_ab_vlm(
    #     url="rtsp://192.168.1.10/stream",
    #     slice_sec=5,
    #     max_secs=30,
    # )
