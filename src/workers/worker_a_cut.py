from __future__ import annotations

'''
Author: 13594053100@163.com
Date: 2025-10-08 14:30:47
LastEditTime: 2025-10-13 18:29:03
'''
# -*- coding: utf-8 -*-

"""
子线程A：切窗 + 标准化 + （按模式）关键帧策略
- 固定参数：url/have_audio_track/mode/slice_sec（启动时传入）
- 运行期控制：q_ctrl 收到 {type: ...}
    START / PAUSE / RESUME / MODE_CHANGE / UPDATE_SLICE / UPDATE_OVERLAP / STOP
- 输出（发往下游 B/C 的“数据”队列，A 不负责下游 STOP 哨兵）：
    (若启用B且存在视频)
    q_video.put({
        "path": <无声小视频>, "t0": <秒>, "t1": <秒>,
        "segment_index": <int>,
        "mode": "offline|online|security",
        "win": <float>, "step": <float>, "overlap_sec": <float>,
        "has_audio": <bool>,
        "keyframes": [<jpg路径>...],
        "keyframe_count": <int>,
        "small_video": <压缩后mp4路径或 None>,
        "hires": <bool>,
        "policy": {
            "significant_motion": <bool>,
            "policy_used": "small_video" | "fixed_sampling" | "keyframes" | "none",
            "interval_sec": <float>,
            "max_frames": <int|None>,
            "silence_hint": {"silence_ratio": <float>, "is_mostly_silent": <bool>}
        }
    })
    (若启用C且存在音频)
    q_audio.put({
        "path": <标准化wav>, "t0": <秒>, "t1": <秒>,
        "segment_index": <int>,
        "silence_hint": {"silence_ratio": <float>, "is_mostly_silent": <bool>}
    })
"""

from typing import Optional, Tuple, List, Dict
from queue import Queue, Empty, Full
from time import sleep
import subprocess
import numpy as np
import cv2
import time
import re
import os
import threading

from src.all_enum import MODEL
from src.utils.ffmpeg_utils import FFmpegUtils
from src.utils.logger_utils import get_logger

logger = get_logger(__name__)


# -------------------- 小工具 --------------------
def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _imwrite_jpg(path: str, img, quality: int = 90) -> str:
    """imencode 写盘，避免中文路径问题。"""
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode('.jpg', ...) failed")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    return path


# -------------------- 窗口/步长策略 --------------------
def _mode_window_policy(mode: MODEL, desire_win: float, overlap_sec: float) -> Tuple[float, float]:
    """
    返回 (win, step)
    - security: win ∈ [4, 8],   step = win
    - online  : win ∈ [5, 10],  step = win
    - offline : win ∈ [8, 12],  step = win - overlap_sec (overlap ∈ [0.0, 1.0])
    """
    if mode == MODEL.SECURITY:
        win = _clamp(float(desire_win), 4.0, 8.0)
        return win, win
    elif mode == MODEL.ONLINE:
        win = _clamp(float(desire_win), 5.0, 10.0)
        return win, win
    else:
        win = _clamp(float(desire_win), 8.0, 12.0)
        ovl = _clamp(float(overlap_sec), 0.0, 1.0)
        step = max(0.1, win - ovl)
        return win, step


# -------------------- 运动评分（探测，不写盘） --------------------
def compute_motion_score(
    video_path: str,
    sample_interval: float = 1.0,
    resize_w: int = 160,
    max_samples: int = 30,
) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"avg_diff": 0.0, "max_diff": 0.0, "samples": 0}

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps * sample_interval)))
    last_gray = None
    diffs: List[float] = []
    try:
        i = 0
        while len(diffs) < max_samples:
            ret, frame = cap.read()
            if not ret:
                break
            i += 1
            if i % step:
                continue
            h, w = frame.shape[:2]
            small_h = max(1, int(h * (resize_w / max(1, w))))
            small = cv2.resize(frame, (resize_w, small_h), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if last_gray is not None:
                diffs.append(cv2.absdiff(last_gray, gray).mean())
            last_gray = gray
    finally:
        cap.release()

    if not diffs:
        return {"avg_diff": 0.0, "max_diff": 0.0, "samples": 0}
    return {"avg_diff": float(sum(diffs) / len(diffs)), "max_diff": float(max(diffs)), "samples": len(diffs)}


def fast_motion_detect_placeholder(
    video_path: str,
    sample_interval: float = 1.0,
    diff_threshold: float = 15.0,
    max_samples: int = 30,
) -> bool:
    score = compute_motion_score(video_path, sample_interval, 160, max_samples)
    return score["samples"] > 0 and score["avg_diff"] >= diff_threshold


# -------------------- 静音探测（基于标准化后的音频 a_out） --------------------
def silence_detect_hint(
    audio_path: str,
    *,
    noise_db: float = -40.0,
    min_silence: float = 0.30,
    timeout_sec: float = 60.0,
) -> dict:
    """
    使用 FFmpeg silencedetect 对标准化后的 WAV（16kHz/mono/PCM16）做静音探测。
    返回：{"silence_ratio": <0~1>, "is_mostly_silent": <bool>, "segments": [(s,e)...]}
    """
    start_re = re.compile(r"silence_start:\s*([0-9.]+)")
    end_re = re.compile(r"silence_end:\s*([0-9.]+)\s*\|\s*silence_duration:\s*([0-9.]+)")

    try:
        duration = FFmpegUtils.ffprobe_duration(audio_path)
    except Exception:
        duration = None

    cmd = [
        "ffmpeg", "-hide_banner", "-nostats", "-v", "info",
        "-i", audio_path,
        "-af", f"silencedetect=noise={noise_db}dB:d={min_silence}",
        "-f", "null", "-"
    ]

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, bufsize=1)
    except FileNotFoundError:
        return {"silence_ratio": 0.0, "is_mostly_silent": False, "segments": []}

    silence_intervals: List[tuple] = []
    cur_start = None
    t0 = time.time()
    try:
        assert proc.stderr
        for line in iter(proc.stderr.readline, ""):
            if timeout_sec and (time.time() - t0 > timeout_sec):
                break
            m1 = start_re.search(line)
            if m1:
                try:
                    cur_start = float(m1.group(1))
                except Exception:
                    cur_start = None
                continue
            m2 = end_re.search(line)
            if m2:
                try:
                    end = float(m2.group(1))
                    dur = float(m2.group(2))
                    silence_intervals.append((cur_start, end, dur))
                except Exception:
                    pass
                cur_start = None
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            proc.wait(timeout=2)
        except Exception:
            pass
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass

    total_silence = sum([d for _, _, d in silence_intervals if d])
    if duration and duration > 0:
        silence_ratio = max(0.0, min(1.0, total_silence / duration))
    else:
        silence_ratio = 0.0

    return {
        "silence_ratio": round(silence_ratio, 3),
        "is_mostly_silent": silence_ratio > 0.8,
        "segments": [(s, e) for (s, e, _) in silence_intervals if s is not None and e is not None],
    }


# -------------------- 关键帧/快照 --------------------
def extract_keyframes_by_interval(
    video_path: str,
    out_dir: str,
    tag: str,
    interval_sec: float = 1.0,
    diff_threshold: float = 0.65,
    resize_w: int = 320,
    jpg_quality: int = 90,
    max_frames: Optional[int] = None,
    *,
    probe_only: bool = False,
    return_stats: bool = False,
    max_samples_for_probe: int = 30,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"avg_diff": 0.0, "max_diff": 0.0, "samples": 0} if (probe_only and return_stats) else []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps * interval_sec)))
    saved: List[str] = []
    last_small = None
    diffs: List[float] = []
    try:
        while True:
            for _ in range(step - 1):
                if not cap.grab():
                    if probe_only and return_stats:
                        if not diffs:
                            return {"avg_diff": 0.0, "max_diff": 0.0, "samples": 0}
                        return {
                            "avg_diff": float(sum(diffs) / len(diffs)),
                            "max_diff": float(max(diffs)),
                            "samples": len(diffs),
                        }
                    return saved
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            small_h = max(1, int(h * (resize_w / max(1, w))))
            small = cv2.resize(frame, (resize_w, small_h), interpolation=cv2.INTER_AREA)

            save = last_small is None
            if last_small is not None:
                diff = cv2.absdiff(last_small, small)
                score = float(np.count_nonzero(diff)) / diff.size
                diffs.append(score)
                save = score >= diff_threshold
            last_small = small

            if probe_only:
                if len(diffs) >= max_samples_for_probe:
                    break
                continue

            if save:
                name = f"{tag}_kf_{len(saved):04d}.jpg"
                out_path = os.path.join(out_dir, name)
                _imwrite_jpg(out_path, frame, quality=jpg_quality)
                saved.append(out_path)
                if max_frames and len(saved) >= max_frames:
                    break
    finally:
        cap.release()

    if probe_only and return_stats:
        if not diffs:
            return {"avg_diff": 0.0, "max_diff": 0.0, "samples": 0}
        return {"avg_diff": float(sum(diffs) / len(diffs)), "max_diff": float(max(diffs)), "samples": len(diffs)}
    return saved


def sample_fixed_frames(video_path: str, out_dir: str, tag: str, count: int = 2, jpg_quality: int = 90) -> List[str]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        return []
    idxs = []
    if count <= 1:
        idxs = [total // 2]
    else:
        for k in range(count):
            idxs.append(int(round((k + 1) / (count + 1) * total)))
    saved = []
    try:
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            name = f"{tag}_snap_{len(saved):04d}.jpg"
            out_path = os.path.join(out_dir, name)
            _imwrite_jpg(out_path, frame, quality=jpg_quality)
            saved.append(out_path)
    finally:
        cap.release()
    return saved


# -------------------- 关键帧/小视频策略表 --------------------
VLM_POLICY = {
    MODEL.ONLINE: {
        "interval_sec": 1.0,
        "max_frames": 3,
        "hires": False,
        "small_video": False,
        "small_video_encode": {"fps": 8, "height": 480, "crf": 28, "preset": "veryfast"},
    },
    MODEL.SECURITY: {
        "interval_sec": 1.0,
        "max_frames_signif": 5,
        "count_nonsignif": 2,
        "hires": False,
        "small_video": False,
        "small_video_encode": {"fps": 8, "height": 480, "crf": 28, "preset": "veryfast"},
    },
    MODEL.OFFLINE: {
        "interval_sec": 1.0,
        "max_frames_nonsignif": 6,
        "hires": False,
        "small_video": True,
        "small_video_encode": {"fps": 8, "height": 480, "crf": 28, "preset": "veryfast"},
    },
}


def decide_vlm_sampling(mode: MODEL, significant_motion: bool) -> Dict:
    cfg = VLM_POLICY[mode]
    policy = {
        "extract_keyframes": False,
        "interval_sec": cfg.get("interval_sec", 1.0),
        "max_frames": None,
        "use_fixed_sampling": False,
        "fixed_count": 0,
        "emit_small_video": False,
        "encode": cfg.get("small_video_encode", {"fps": 8, "height": 480, "crf": 28, "preset": "veryfast"}),
        "hires": cfg.get("hires", False),
    }
    if mode == MODEL.ONLINE:
        policy["extract_keyframes"] = True
        policy["max_frames"] = cfg.get("max_frames", 3)
    elif mode == MODEL.SECURITY:
        if significant_motion:
            policy["extract_keyframes"] = True
            policy["max_frames"] = cfg.get("max_frames_signif", 5)
        else:
            policy["use_fixed_sampling"] = True
            policy["fixed_count"] = cfg.get("count_nonsignif", 2)
    else:  # OFFLINE
        if significant_motion:
            policy["emit_small_video"] = cfg.get("small_video", True)
        else:
            policy["extract_keyframes"] = True
            policy["max_frames"] = cfg.get("max_frames_nonsignif", 6)
    return policy


def _safe_put_with_ctrl(
    q: Optional[Queue], obj: dict, q_ctrl: Queue, stop: object,
    *, timeout=0.2, max_tries=50
) -> bool:
    """
    尝试多次短超时 put；
    期间非阻塞轮询控制队列；
    一旦检测到 STOP/SHUTDOWN 或超过重试上限，返回 False；
    投递成功返回 True。
    若 q 为 None，表示该数据面未启用，直接返回 False（视为跳过投递）。
    """
    if q is None:
        return False

    tries = 0
    while True:
        # ---- 控制优先：收到 STOP 立即放弃本次投递 ----
        try:
            msg = q_ctrl.get_nowait()
            if msg is stop or (isinstance(msg, dict) and msg.get("type") in ("STOP", "SHUTDOWN")):
                logger.info("[A] 检测到控制队列 STOP，停止将本次切片传递到数据队列。")
                return False
            # 其他控制消息丢回去（不吞）
            try:
                q_ctrl.put_nowait(msg)
            except Full:
                pass
        except Empty:
            pass

        # ---- 尝试 put ----
        try:
            q.put(obj, timeout=timeout)
            return True
        except Full:
            tries += 1
            if tries >= max_tries:
                logger.warning("[A] 队列持续拥堵，放弃投递。")
                return False
        except Exception as e:
            logger.warning(f"[A] put 发生异常，将视为失败：{e}")
            return False


# -------------------- A 线程主体 --------------------
def worker_a_cut(
    url: str,
    have_audio_track: bool,
    mode: MODEL,
    slice_sec: int,
    q_audio: Optional[Queue],   # <- 可选队列
    q_video: Optional[Queue],   # <- 可选队列
    q_ctrl: Queue,
    stop: object,
):
    """
    A线程：切窗 + 标准化 + （按模式）关键帧策略。
    注意：
      - A 不向下游队列发 STOP 哨兵；只有主控在合适时机发。
      - 控制通道同时识别 stop 对象和 {"type":"STOP"} 字典。
      - 离线：切完整个文件（尾窗兜底）后退出；实时：常驻。
    """
    running = False
    paused = False
    cur_mode = mode
    desire_win = max(1, int(slice_sec))
    overlap_sec = 0.0  # offline only
    t0 = 0.0
    seg_idx = 0
    out_dir = "out"

    # 离线有总时长；RTSP/直播 None
    max_duration: Optional[float] = None
    try:
        max_duration = FFmpegUtils._probe_duration_seconds(url)
    except Exception:
        logger.warning("[A] FFmpeg 获取媒体总时长失败，按实时流常驻处理")
    tail_emitted = False  # offline 尾窗兜底只发一次

    def drain_ctrl_once() -> Optional[str]:
        nonlocal running, paused, cur_mode, desire_win, overlap_sec
        try:
            msg = q_ctrl.get_nowait()
        except Empty:
            return None

        if msg is stop:
            return "STOP"

        if isinstance(msg, dict):
            typ = msg.get("type")
            if typ == "START":
                logger.info("[A] 收到 START，开始切片")
                running = True
            elif typ == "PAUSE":
                logger.info("[A] 收到 PAUSE，暂停切片")
                paused = True
            elif typ == "RESUME":
                logger.info("[A] 收到 RESUME，继续切片")
                paused = False
            elif typ == "MODE_CHANGE":
                val = msg.get("value")
                if isinstance(val, str):
                    try:
                        cur_mode = MODEL(val)
                        logger.info(f"[A] 模式切换为：{cur_mode}")
                    except Exception:
                        pass
            elif typ == "UPDATE_SLICE":
                try:
                    desire_win = max(1, int(msg.get("value", desire_win)))
                    logger.info(f"[A] 切窗大小更新：{desire_win}")
                except Exception:
                    pass
            elif typ == "UPDATE_OVERLAP":
                try:
                    overlap_sec = float(msg.get("value", overlap_sec))
                    logger.info(f"[A] 窗口重叠更新：{overlap_sec}")
                except Exception:
                    pass
            elif typ in ("STOP", "SHUTDOWN"):
                return "STOP"
        return None

    try:
        while True:
            # 控制优先，降低延迟
            for _ in range(4):
                res = drain_ctrl_once()
                if res == "STOP":
                    logger.info("[A] 收到控制队列 STOP，退出")
                    return
                if not running or paused:
                    sleep(0.01)

            if not running or paused:
                continue

            # 窗口/步长（含 offline 重叠）
            win, step = _mode_window_policy(cur_mode, desire_win, overlap_sec)

            # 离线 EOF：用 t0 与 max_duration 判定；offline 做一次尾窗兜底
            if max_duration is not None and t0 >= max_duration:
                if cur_mode == MODEL.OFFLINE and not tail_emitted and max_duration > 0:
                    new_t0 = max(0.0, max_duration - win)
                    if new_t0 + 1e-6 < t0:
                        t0 = new_t0
                        tail_emitted = True
                    else:
                        logger.info("[A] 离线已切完，退出")
                        return
                else:
                    logger.info("[A] 离线已切完，退出")
                    return

            # ---- 切窗+标准化（输出：可能仅音频 / 可能仅视频 / 也可能都有）----
            seg = FFmpegUtils.cut_and_standardize_segment(
                src_url=url, start_time=t0, duration=win,
                output_dir=out_dir, segment_index=seg_idx, have_audio=have_audio_track
            )
            v_out = seg.get("video_path")  # 可能为 None（纯音频源）
            a_out = seg.get("audio_path")  # 可能为 None（纯视频源或无音轨）

            has_v = bool(v_out)
            has_a = bool(a_out)

            # “是否要产出音频”：需要有音轨、q_audio 可用，并且本段确实切出了 a_out
            produce_audio = bool(have_audio_track and q_audio is not None and has_a)

            # ---- 静音提示：仅当要产出音频时才做 ----
            if produce_audio:
                silence_hint = silence_detect_hint(a_out)
            else:
                silence_hint = {"silence_ratio": 0.0, "is_mostly_silent": False, "segments": []}

            # ---- 运动检测 & 关键帧/小视频策略：仅当有视频且启用了 B（q_video 可用）时才做 ----
            keyframes: List[str] = []
            small_video_path: Optional[str] = None
            policy_used = None
            hires_flag = False

            if has_v and q_video is not None:
                # 轻量运动探测
                signif = fast_motion_detect_placeholder(
                    v_out, sample_interval=1.0, diff_threshold=15.0, max_samples=30
                )
                # 关键帧/小视频策略
                vlm_policy = decide_vlm_sampling(cur_mode, signif)
                hires_flag = vlm_policy["hires"]
                tag = f"seg{seg_idx:04d}"

                if vlm_policy["emit_small_video"]:
                    small_video_path = os.path.join(out_dir, f"{tag}_small.mp4")
                    enc = vlm_policy["encode"]
                    FFmpegUtils.compress_video_for_vlm(
                        in_video=v_out, out_video=small_video_path,
                        fps=enc.get("fps", 8), height=enc.get("height", 480),
                        crf=enc.get("crf", 28), preset=enc.get("preset", "veryfast")
                    )
                    policy_used = "small_video"
                elif vlm_policy["use_fixed_sampling"]:
                    keyframes = sample_fixed_frames(v_out, out_dir, tag, count=vlm_policy["fixed_count"])
                    policy_used = "fixed_sampling"
                elif vlm_policy["extract_keyframes"]:
                    dynamic_max = vlm_policy.get("max_frames")
                    keyframes = extract_keyframes_by_interval(
                        v_out, out_dir, tag,
                        interval_sec=vlm_policy["interval_sec"],
                        diff_threshold=0.65,
                        max_frames=dynamic_max,
                    )
                    policy_used = "keyframes"
            else:
                signif = False
                policy_used = "none"
                hires_flag = False
                small_video_path = None
                keyframes = []

            # ---- 投递到视频队列：仅当有视频且 q_video 可用 ----
            if has_v and q_video is not None:
                payload_video = {
                    "path": v_out,
                    "t0": seg["t0"],
                    "t1": seg["t1"],
                    "segment_index": seg_idx,
                    "mode": cur_mode.value,
                    "win": win,
                    "step": step,
                    "overlap_sec": overlap_sec if cur_mode == MODEL.OFFLINE else 0.0,
                    "has_audio": has_a,
                    "keyframes": keyframes,
                    "keyframe_count": len(keyframes),
                    "small_video": small_video_path,
                    "hires": hires_flag,
                    "policy": {
                        "significant_motion": bool(signif),
                        "policy_used": (policy_used or "none"),
                        "interval_sec": (vlm_policy["interval_sec"] if has_v and q_video is not None else 0.0),
                        "max_frames": (vlm_policy.get("max_frames") if has_v and q_video is not None else None),
                        "silence_hint": {
                            "silence_ratio": float(silence_hint.get("silence_ratio", 0.0)),
                            "is_mostly_silent": bool(silence_hint.get("is_mostly_silent", False)),
                        },
                    },
                }
                ok = _safe_put_with_ctrl(q_video, payload_video, q_ctrl, stop)
                if not ok:
                    return  # 退出 A 线程

            # ---- 投递到音频队列：仅当“准备产出音频”且 q_audio 可用 ----
            if produce_audio and q_audio is not None:
                audio_payload = {
                    "path": a_out,
                    "t0": seg["t0"],
                    "t1": seg["t1"],
                    "segment_index": seg_idx,
                    "silence_hint": {
                        "silence_ratio": float(silence_hint.get("silence_ratio", 0.0)),
                        "is_mostly_silent": bool(silence_hint.get("is_mostly_silent", False)),
                    },
                }
                ok = _safe_put_with_ctrl(q_audio, audio_payload, q_ctrl, stop)
                if not ok:
                    return

            # ---- 步进；offline 尾窗已发过的话，把 t0 推到 EOF 以便下一轮退出 ----
            seg_idx += 1
            t0 += step
            if tail_emitted and max_duration is not None:
                t0 = max_duration

            # 让出 GIL
            sleep(0.005)

    except Exception as e:
        logger.error(f"[A] 运行时异常，线程退出：{e}")
        # 注意：A 不主动给下游发 STOP；由主控在合适时机发哨兵
    finally:
        logger.info('[A] 线程退出清理完成')
