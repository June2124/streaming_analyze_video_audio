'''
Author: 13594053100@163.com
Date: 2025-10-11 15:09:41
LastEditTime: 2025-10-11 16:38:50
'''

# -*- coding: utf-8 -*-
"""
专测：主控 -> A -> C -> OUT-ASR -> 上层消费（直接打印增量）
用法：直接运行本文件即可。按需修改 AUDIO_PATH / SLICE_SEC。
"""
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


import logging
from contextlib import ContextDecorator

# === 直接在这里改参数 ===
AUDIO_PATH = r"D:\streaming_analyze_video_audio\static\video\RAG_Video_with_sound_test_16k.wav"   # 改成你的本地音频路径（建议 wav）
SLICE_SEC  = 8     # A 侧切窗秒数

# --- 按你的工程结构调整导入 ---
from src.all_enum import MODEL
from streaming_analyze import StreamingAnalyze


class EnvPatcher(ContextDecorator):
    """设置一批环境变量，并在退出时自动恢复/清理。"""
    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping
        self._snapshot: dict[str, tuple[bool, str]] = {}

    def __enter__(self):
        for k, v in self.mapping.items():
            if k in os.environ:
                self._snapshot[k] = (True, os.environ[k])
            else:
                self._snapshot[k] = (False, "")
            os.environ[k] = v
        return self

    def __exit__(self, exc_type, exc, tb):
        for k, (existed, old_val) in self._snapshot.items():
            if existed:
                os.environ[k] = old_val
            else:
                os.environ.pop(k, None)
        return False  # 不吞异常


def _setup_logging():
    root = logging.getLogger()
    if not root.handlers:
        h = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
        h.setFormatter(fmt)
        root.addHandler(h)
    root.setLevel(logging.INFO)


def main():
    _setup_logging()

    if not os.path.exists(AUDIO_PATH):
        print(f"[TEST] 文件不存在: {AUDIO_PATH}")
        return

    # 仅本次测试设置的环境变量（会自动恢复）
    test_env = {
        "ASR_VAD_ENABLED": "1",
        "ASR_VAD_BACKEND": "auto",   # auto 优先 webrtcvad, 无则回退能量
        "ASR_VAD_FRAME_MS": "20",
        "ASR_VAD_AGGR": "2",
        "ASR_VAD_DBFS": "-40",
        "ASR_VAD_MIN_RATIO": "0.10",
        "ASR_VAD_MIN_MS": "300",
        "ASR_VAD_HANG_MS": "200",
    }

    with EnvPatcher(test_env):
        url = AUDIO_PATH
        ctrl = StreamingAnalyze(url=url, mode=MODEL.OFFLINE, slice_sec=SLICE_SEC)

        # ---- 上层：消费 OUT-ASR 的增量/收尾/无声跳过 ----
        def on_asr_delta(msg):
            seg = msg.get("segment_index")
            seq = msg.get("seq")
            delta = (msg.get("delta") or "").replace("\n", "\\n")
            print(f"[TEST][ASRΔ seg#{seg} seq={seq}] {delta}")

        def on_asr_done(msg):
            seg = msg.get("segment_index")
            text = (msg.get("full_text") or "").replace("\n", "\\n")
            usage = msg.get("usage")
            print(f"[TEST][ASR✓ seg#{seg}] {text}")
            if usage:
                vad = usage.get("vad") or {}
                s_hint = usage.get("silence_hint") or {}
                print(f"       usage.vad.backend={vad.get('backend_used')} "
                      f"active_ratio={vad.get('active_ratio')} "
                      f"applied={vad.get('applied_params')}")
                print(f"       usage.silence_hint={s_hint}")

        def on_asr_no_speech(msg):
            seg = msg.get("segment_index")
            t0, t1 = msg.get("t0"), msg.get("t1")
            try:
                dur = float(t1) - float(t0)
            except Exception:
                dur = None
            usage = msg.get("usage") or {}
            vad = usage.get("vad") or {}
            s_hint = usage.get("silence_hint") or {}
            print(f"[TEST][ASR⊘ 无人声 seg#{seg} dur={dur if dur is not None else 'n/a'}s] "
                  f"backend={vad.get('backend_used')} active_ratio={vad.get('active_ratio')} "
                  f"silence_ratio={s_hint.get('silence_ratio')}")

        ctrl.on_asr_delta = on_asr_delta
        ctrl.on_asr_done  = on_asr_done
        ctrl.on_asr_no_speech = on_asr_no_speech

        try:
            ctrl.start_streaming_analyze()  # 离线音频会自然收尾
        except KeyboardInterrupt:
            print("\n[TEST] 捕获 Ctrl+C，调用 force_stop()")
            ctrl.force_stop("KeyboardInterrupt")
        finally:
            # 如果你把统计方法放回了类里，这里打印一次
            if hasattr(ctrl, "snapshot_stats"):
                try:
                    stats = ctrl.snapshot_stats()
                    print(f"[TEST][STATS] {stats}")
                except Exception:
                    pass


if __name__ == "__main__":
    main()
