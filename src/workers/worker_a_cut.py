from __future__ import annotations

"""
子线程A：切窗 + 标准化 + （按模式）关键帧策略（含 OpenCV/FFmpeg 双兜底）
- 固定参数：url / have_audio_track / mode / slice_sec（启动时传入）
- 运行期控制：q_ctrl 收到 {type: ...}
    START / PAUSE / RESUME / MODE_CHANGE / UPDATE_SLICE / UPDATE_OVERLAP / STOP
- 输出（发往下游 B/C 的“数据”队列，A 不负责下游 STOP 哨兵）：
    (若启用B且存在视频)
    q_video.put({
        "path": <无声小视频>, "t0": <秒>, "t1": <秒>,
        "clip_t0": <秒>, "clip_t1": <秒>,
        "segment_index": <int>,
        "mode": "offline|online|security",
        "win": <float>, "step": <float>, "overlap_sec": <float>,
        "has_audio": <bool>,
        "keyframes": [<jpg路径>...],            # 关键帧/快照（若策略选择了图像）
        "frame_pts": [<float秒>...],            # 与 keyframes 一一对应（源时间线）
        "frame_indices": [<int>...],            # 可选：帧索引（相对该片段 v_out）
        "keyframe_count": <int>,
        "small_video": <压缩后mp4路径或 None>,
        "small_video_fps": <float或None>,       # 若走小视频策略则提供
        "hires": <bool>,
        "policy": {
            "significant_motion": <bool>,
            "policy_used": "small_video" | "fixed_sampling" | "keyframes" | "none",
            "interval_sec": <float>,
            "max_frames": <int|None>,
            "silence_hint": {"silence_ratio": <float>, "is_mostly_silent": <bool>},
            "diff_method": "<gray_mean|bgr_ratio|hist|flow>",
            "diff_threshold": <float>,
            "hist_bins": <int>,
            "flow_step": <int>,
        }
    })
    (若启用C且存在音频)
    q_audio.put({
        "path": <标准化wav>, "t0": <秒>, "t1": <秒>,
        "segment_index": <int>,
        "silence_hint": {"silence_ratio": <float>, "is_mostly_silent": <bool>}
    })
"""

from typing import Optional, Tuple, List, Dict, Literal
from queue import Queue, Empty, Full
from time import sleep
import subprocess
import numpy as np
import cv2
import time
import os

from src.all_enum import MODEL
from src.utils.ffmpeg_utils import FFmpegUtils
from src.utils.logger_utils import get_logger
from src.configs.cut_config import CutConfig

logger = get_logger(__name__)
DiffMethod = Literal["gray_mean", "bgr_ratio", "hist", "flow"]  # 帧间变化度计算方法

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


def _get_video_meta(video_path: str) -> Tuple[float, int]:
    """返回 (fps, total_frames)；失败时回退 (25.0, 0)"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 25.0, 0
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    if fps <= 0:
        fps = 25.0
    return fps, total


def _file_exists_nonzero(path: Optional[str]) -> bool:
    return bool(path) and os.path.exists(path) and os.path.getsize(path) > 0


# -------------------- 窗口/步长策略 --------------------
def _mode_window_policy(mode: MODEL, desire_win: float, overlap_sec: float) -> Tuple[float, float]:
    """
    返回 (win, step)
    """
    if mode == MODEL.SECURITY:
        win = _clamp(float(desire_win), 1.0, 12.0)  # 建议 4~8
        return win, win
    elif mode == MODEL.ONLINE:
        win = _clamp(float(desire_win), 1.0, 16.0)  # 建议 5~10
        return win, win
    else:
        win = _clamp(float(desire_win), 1.0, 30.0)  # 建议 8~15
        ovl = _clamp(float(overlap_sec), 0.0, 2.0)
        step = max(0.1, win - ovl)
        return win, step


# -------------------- 运动评分 --------------------
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
    import re, subprocess
    try:
        duration = FFmpegUtils.ffprobe_duration(audio_path)
    except Exception:
        duration = None

    start_re = re.compile(r"silence_start:\s*([0-9.]+)")
    end_re = re.compile(r"silence_end:\s*([0-9.]+)\s*\|\s*silence_duration:\s*([0-9.]+)")

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


# -------------------- FFmpeg 抓帧兜底 --------------------
def _ffmpeg_grab_frames_by_indices(video_path: str, out_dir: str, tag: str, indices: List[int]) -> List[str]:
    """
    按近似帧位导出图片（用 eq(n,idx) 选择），部分容器/时间基可能不精确。
    """
    if not indices:
        return []
    os.makedirs(out_dir, exist_ok=True)
    outs: List[str] = []
    for k, idx in enumerate(indices):
        out = os.path.join(out_dir, f"{tag}_snap_{k:04d}.jpg")
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", video_path,
            "-vf", f"select='eq(n\\,{idx})'",
            "-vsync", "vfr",
            out
        ]
        try:
            subprocess.check_call(cmd)
            if _file_exists_nonzero(out):
                outs.append(out)
        except Exception:
            continue
    return outs


def _ffmpeg_grab_frames_by_interval(video_path: str, out_dir: str, tag: str, interval_sec: float) -> List[str]:
    """
    等间隔抓帧（fps=1/interval）。作为 OpenCV 间隔抽帧失败时的兜底。
    """
    if interval_sec <= 0:
        interval_sec = 1.0
    os.makedirs(out_dir, exist_ok=True)
    pattern = os.path.join(out_dir, f"{tag}_kf_%04d.jpg")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", video_path,
        "-vf", f"fps=1/{max(1e-3, float(interval_sec))}",
        pattern
    ]
    try:
        subprocess.check_call(cmd)
    except Exception:
        return []
    outs = []
    for fname in sorted(os.listdir(out_dir)):
        if fname.startswith(f"{tag}_kf_") and fname.endswith(".jpg"):
            outs.append(os.path.join(out_dir, fname))
    return outs


# -------------------- 关键帧/快照（含 PTS） --------------------
def extract_keyframes_by_interval_with_pts(
    video_path: str,
    out_dir: str,
    tag: str,
    interval_sec: float = 1.0,            # 采样间隔（候选帧节拍)
    diff_threshold: float = 0.65,         # 根据 diff_method 解释
    resize_w: int = 320,
    jpg_quality: int = 90,
    max_frames: Optional[int] = None,
    *,
    seg_t0: float,
    diff_method: DiffMethod = "bgr_ratio",
    hist_bins: int = 32,
    flow_step: int = 1,                   # 光流比较的候选帧步长
) -> Tuple[List[str], List[float], List[int]]:
    """
    返回 (paths, pts_list, frame_indices)

    diff_method:
      - "gray_mean": 灰度绝对差的平均值 / 255, 推荐阈值: 0.08~0.20  值域[0,1] 越大越不相似
      - "bgr_ratio": BGR三通道绝对差的非零比例, 推荐阈值: 0.05~0.15 值域[0,1]  越大越不相似
      - "hist": 灰度直方图相关系数的(1-corr)，推荐阈值: 0.15~0.35 值域[0,2]
      - "flow": Farneback光流幅值均值，推荐阈值: 0.5~2.0(视分辨率/场景），值域[0,+∞)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("[A] 待抽取关键帧视频，无法通过 OpenCV 打开，使用 FFmpeg 兜底")
        outs = _ffmpeg_grab_frames_by_interval(video_path, out_dir, tag, interval_sec)
        if not outs:
            return [], [], []
        fps, total = _get_video_meta(video_path)
        if fps <= 0: fps = 25.0
        pts = [float(seg_t0) + i * interval_sec for i in range(len(outs))]
        idxs = [min(int(round(i * fps * interval_sec)), max(0, total - 1)) for i in range(len(outs))]
        return outs, pts, idxs

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    if fps <= 0: fps = 25.0
    step = max(1, int(round(fps * interval_sec)))

    paths: List[str] = []
    pts: List[float] = []
    fidx: List[int] = []

    last_small_bgr = None          # bgr_ratio
    last_gray = None               # gray_mean / hist / flow
    last_hist = None               # hist
    frame_idx_from_cap = -1
    grabbed_since_last = 0         # flow_step 控制

    def _resize_keep_w(frame_bgr):
        h, w = frame_bgr.shape[:2]
        small_h = max(1, int(h * (resize_w / max(1, w))))
        return cv2.resize(frame_bgr, (resize_w, small_h), interpolation=cv2.INTER_AREA)

    def _gray(img_bgr_small):
        return cv2.cvtColor(img_bgr_small, cv2.COLOR_BGR2GRAY)

    def _gray_mean_score(g1, g2) -> float:
        diff = cv2.absdiff(g1, g2).astype("float32")
        return float(diff.mean()) / 255.0  # 0~1

    def _bgr_ratio_score(b1, b2) -> float:
        diff = cv2.absdiff(b1, b2)
        return float(np.count_nonzero(diff)) / float(diff.size)  # 0~1

    def _hist_score(g1, g2) -> float:
        hist1 = cv2.calcHist([g1], [0], None, [hist_bins], [0, 256])
        hist2 = cv2.calcHist([g2], [0], None, [hist_bins], [0, 256])
        cv2.normalize(hist1, hist1)
        cv2.normalize(hist2, hist2)
        corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)  # [-1,1]，越大越相似
        return float(1.0 - max(-1.0, min(1.0, corr)))             # 0(相同)~2(完全相反)

    def _flow_score(prev_gray, cur_gray) -> float:
        flow = cv2.calcOpticalFlowFarneback(prev_gray, cur_gray,
                                            None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return float(np.mean(mag))

    try:
        while True:
            for _ in range(step - 1):
                if not cap.grab():
                    cap.release()
                    return paths, pts, fidx
                grabbed_since_last += 1

            ret, frame = cap.read()
            if not ret:
                break
            frame_idx_from_cap = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

            small_bgr = _resize_keep_w(frame)
            need_save = False

            if diff_method == "bgr_ratio":
                if last_small_bgr is None:
                    need_save = True
                else:
                    score = _bgr_ratio_score(last_small_bgr, small_bgr)
                    need_save = (score >= diff_threshold)
                last_small_bgr = small_bgr

            elif diff_method == "gray_mean":
                g = _gray(small_bgr)
                if last_gray is None:
                    need_save = True
                else:
                    score = _gray_mean_score(last_gray, g)
                    need_save = (score >= diff_threshold)
                last_gray = g

            elif diff_method == "hist":
                g = _gray(small_bgr)
                if last_gray is None:
                    need_save = True
                    last_hist = None
                else:
                    if last_hist is None:
                        last_hist = cv2.calcHist([last_gray], [0], None, [hist_bins], [0, 256])
                        cv2.normalize(last_hist, last_hist)
                    cur_hist = cv2.calcHist([g], [0], None, [hist_bins], [0, 256])
                    cv2.normalize(cur_hist, cur_hist)
                    corr = cv2.compareHist(last_hist, cur_hist, cv2.HISTCMP_CORREL)
                    score = float(1.0 - max(-1.0, min(1.0, corr)))
                    need_save = (score >= diff_threshold)
                    last_hist = cur_hist
                last_gray = g

            elif diff_method == "flow":
                g = _gray(small_bgr)
                if (last_gray is None) or (grabbed_since_last < flow_step):
                    need_save = (last_gray is None)
                else:
                    score = _flow_score(last_gray, g)
                    need_save = (score >= diff_threshold)
                last_gray = g
                grabbed_since_last = 0

            else:
                need_save = (last_small_bgr is None and last_gray is None)

            if need_save:
                name = f"{tag}_kf_{len(paths):04d}.jpg"
                out_path = os.path.join(out_dir, name)
                _imwrite_jpg(out_path, frame, quality=jpg_quality)
                paths.append(out_path)
                pts.append(float(seg_t0) + (frame_idx_from_cap / fps))
                fidx.append(frame_idx_from_cap)
                if max_frames and len(paths) >= max_frames:
                    break

    finally:
        cap.release()

    return paths, pts, fidx


def sample_fixed_frames_with_pts(
    video_path: str,
    out_dir: str,
    tag: str,
    count: int = 2,
    jpg_quality: int = 90,
    *,
    seg_t0: float
) -> Tuple[List[str], List[float], List[int]]:
    """
    更稳妥的固定抽帧：优先“顺播”读取，避免 set(CAP_PROP_POS_FRAMES) 的随机 seek 不可靠；
    OpenCV 失败则使用 FFmpeg 按帧索引近似抓帧（或按时间等距抓帧）兜底。
    返回 (paths, pts_list, frame_indices)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("[A] 待抽取固定帧视频，无法通过 OpenCV 打开，使用 FFmpeg 兜底")
        fps, total = _get_video_meta(video_path)
        if total <= 0:
            return [], [], []
        if count <= 1:
            indices = [total // 2]
        else:
            indices = [int(round((k + 1) / (count + 1) * total)) for k in range(count)]
        outs = _ffmpeg_grab_frames_by_indices(video_path, out_dir, tag, indices)
        if not outs:
            outs = _ffmpeg_grab_frames_by_interval(video_path, out_dir, tag, max(1.0, (total / max(1, count)) / max(1.0, fps)))
        if not outs:
            return [], [], []
        if fps <= 0:
            fps = 25.0
        pts = [float(seg_t0) + (idx / fps) for idx in range(len(outs))]
        idxs = indices if outs and len(outs) == len(indices) else list(range(len(outs)))
        return outs, pts, idxs

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    if fps <= 0:
        fps = 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        return [], [], []

    if count <= 1:
        target_idxs = [total // 2]
    else:
        target_idxs = [int(round((k + 1) / (count + 1) * total)) for k in range(count)]
    target_sorted = sorted(target_idxs)

    paths, pts, fidx = [], [], []
    cur_read_idx = -1
    next_target_i = 0

    try:
        while next_target_i < len(target_sorted):
            ret, frame = cap.read()
            if not ret:
                break
            cur_read_idx += 1
            if cur_read_idx == target_sorted[next_target_i]:
                name = f"{tag}_snap_{len(paths):04d}.jpg"
                out_path = os.path.join(out_dir, name)
                _imwrite_jpg(out_path, frame, quality=jpg_quality)
                paths.append(out_path)
                pts.append(float(seg_t0) + (cur_read_idx / fps))
                fidx.append(cur_read_idx)
                next_target_i += 1
    finally:
        cap.release()

    if paths:
        return paths, pts, fidx

    logger.warning("[A] OpenCV固定抽帧未取到图片，回落到 FFmpeg 抓帧（按索引或等间隔）")
    outs = _ffmpeg_grab_frames_by_indices(video_path, out_dir, tag, target_sorted)
    if not outs:
        interval = max(1.0, (total / max(1, count)) / max(1.0, fps))
        outs = _ffmpeg_grab_frames_by_interval(video_path, out_dir, tag, interval)
    if not outs:
        return [], [], []

    pts2 = [float(seg_t0) + i * max(1.0 / fps, 0.04) for i in range(len(outs))]
    idxs2 = target_sorted if len(outs) == len(target_sorted) else list(range(len(outs)))
    return outs, pts2, idxs2


# -------------------- 关键帧/小视频策略：按模式决定「是否小视频/固定抽帧/关键帧」等 --------------------
VLM_POLICY = {
    MODEL.ONLINE:   {"max_frames": 3, "hires": False, "small_video": False, "encode": {"fps": 8, "height": 480, "crf": 28, "preset": "veryfast"}},
    MODEL.SECURITY: {"max_frames_signif": 5, "count_nonsignif": 2, "hires": False, "small_video": False, "encode": {"fps": 8, "height": 480, "crf": 28, "preset": "veryfast"}},
    MODEL.OFFLINE:  {"max_frames_nonsignif": 6, "hires": False, "small_video": True, "encode": {"fps": 8, "height": 480, "crf": 28, "preset": "veryfast"}},
}

def decide_vlm_sampling(mode: MODEL, significant_motion: bool) -> Dict:
    cfg = VLM_POLICY[mode]
    policy = {
        "extract_keyframes": False,
        "use_fixed_sampling": False,
        "fixed_count": 0,
        "emit_small_video": False,
        "encode": cfg.get("encode", {"fps": 8, "height": 480, "crf": 28, "preset": "veryfast"}),
        "hires": cfg.get("hires", False),
        "max_frames": None,
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
    if q is None:
        return False

    tries = 0
    while True:
        try:
            msg = q_ctrl.get_nowait()
            if msg is stop or (isinstance(msg, dict) and msg.get("type") in ("STOP", "SHUTDOWN")):
                logger.info("[A] 检测到控制队列 STOP，停止将本次切片传递到数据队列。")
                return False
            try:
                q_ctrl.put_nowait(msg)
            except Full:
                pass
        except Empty:
            pass

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
    cut_config: Optional[CutConfig] = None,
):
    # ---- 读取 cut_config----
    cut_config: CutConfig = cut_config or CutConfig()
    interval_sec: float = cut_config.interval_sec
    diff_method: str = cut_config.diff_method.value
    diff_threshold: float = cut_config.diff_threshold            
    hist_bins: int = cut_config.hist_bins
    flow_step: int = cut_config.flow_step
    motion_sample_interval: float = cut_config.motion_sample_interval
    motion_diff_threshold: float = cut_config.motion_diff_threshold
    out_dir: str = cut_config.out_dir

    running = False
    paused = False
    cur_mode = mode
    desire_win = max(1, int(slice_sec))
    overlap_sec = 0.0  # offline only
    t0 = 0.0
    seg_idx = 0

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
                try:
                    cur_mode = val if isinstance(val, MODEL) else MODEL(str(val))
                    logger.info(f"[A] 模式切换为：{cur_mode}")
                except Exception:
                    logger.warning("[A] MODE_CHANGE 值无法解析：%r", val)
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

            # 离线 EOF
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

            # ---- 切窗+标准化 ----
            seg = FFmpegUtils.cut_and_standardize_segment(
                src_url=url, start_time=t0, duration=win,
                output_dir=out_dir, segment_index=seg_idx, have_audio=have_audio_track
            )
            v_out = seg.get("video_path")
            a_out = seg.get("audio_path")
            has_v = _file_exists_nonzero(v_out)
            has_a = _file_exists_nonzero(a_out)

            # 是否产出音频
            produce_audio = bool(have_audio_track and q_audio is not None and has_a)

            # 静音提示
            if produce_audio:
                silence_hint = silence_detect_hint(a_out)
            else:
                silence_hint = {"silence_ratio": 0.0, "is_mostly_silent": False, "segments": []}

            # 运动 & 关键帧策略（仅当有视频且启用 B）
            keyframes: List[str] = []
            frame_pts: List[float] = []
            frame_indices: List[int] = []
            small_video_path: Optional[str] = None
            small_video_fps: Optional[float] = None
            policy_used = "none"
            hires_flag = False

            if has_v and q_video is not None:
                signif = fast_motion_detect_placeholder(
                    v_out, sample_interval=motion_sample_interval, diff_threshold=motion_diff_threshold, max_samples=30
                )
                vlm_policy = decide_vlm_sampling(cur_mode, signif)
                hires_flag = vlm_policy["hires"]
                tag = f"seg{seg_idx:04d}"

                # SECURITY/ONLINE 不会走小视频（配置已禁用）
                if vlm_policy["emit_small_video"]:
                    small_video_path = os.path.join(out_dir, f"{tag}_small.mp4")
                    enc = vlm_policy["encode"]
                    try:
                        FFmpegUtils.compress_video_for_vlm(
                            in_video=v_out, out_video=small_video_path,
                            fps=enc.get("fps", 8), height=enc.get("height", 480),
                            crf=enc.get("crf", 28), preset=enc.get("preset", "veryfast")
                        )
                        small_video_fps, _ = _get_video_meta(small_video_path)
                        policy_used = "small_video"
                    except Exception as e:
                        logger.warning("[A] 小视频压缩失败，将回落到图像策略：%s", e)
                        small_video_path = None
                        small_video_fps = None

                if not small_video_path:
                    # 优先：间隔关键帧（根据 diff_method/diff_threshold）
                    imgs, pts, idxs = extract_keyframes_by_interval_with_pts(
                        v_out, out_dir, tag,
                        interval_sec=interval_sec,
                        diff_threshold=diff_threshold,
                        max_frames=vlm_policy.get("max_frames"),
                        seg_t0=seg["t0"],
                        diff_method=diff_method,
                        hist_bins=hist_bins,
                        flow_step=flow_step,
                    )
                    if imgs:
                        policy_used = "keyframes"
                    else:
                        # 回退：固定抽帧（稳定兜底）
                        logger.warning("[A] seg#%s 关键帧抽取未取到图片，回落到固定抽帧策略。", seg_idx)
                        imgs, pts, idxs = sample_fixed_frames_with_pts(
                            v_out, out_dir, tag,
                            count=int(vlm_policy.get("fixed_count") or 2),
                            jpg_quality=90, seg_t0=seg["t0"]
                        )
                        if imgs:
                            policy_used = "fixed_sampling"

                    keyframes, frame_pts, frame_indices = imgs, pts, idxs

                # 兜底：若小视频被禁用且又没抽到任何图片→跳过视频侧
                if not small_video_path and not keyframes:
                    logger.warning("[A] seg#%s 无可用的小视频/关键帧文件，跳过该段。", seg_idx)

                logger.info(
                    "[A] seg#%s mode=%s signif=%s policy: emit_small_video=%s, use_fixed_sampling=%s, extract_keyframes=%s",
                    seg_idx, cur_mode.value, bool(signif),
                    bool(vlm_policy.get("emit_small_video")),
                    bool(vlm_policy.get("use_fixed_sampling")),
                    True,  # 现在默认优先关键帧
                )
            else:
                signif = False

            # ---- 投视频队列 ----
            if has_v and q_video is not None and (small_video_path or keyframes):
                payload_video = {
                    "path": v_out,
                    "t0": seg["t0"],
                    "t1": seg["t1"],
                    "clip_t0": seg["t0"],
                    "clip_t1": seg["t1"],
                    "segment_index": seg_idx,
                    "mode": cur_mode.value,
                    "win": win,
                    "step": step,
                    "overlap_sec": overlap_sec if cur_mode == MODEL.OFFLINE else 0.0,
                    "has_audio": has_a,
                    "keyframes": keyframes,
                    "frame_pts": frame_pts,
                    "frame_indices": frame_indices,
                    "keyframe_count": len(keyframes),
                    "small_video": small_video_path,
                    "small_video_fps": small_video_fps,
                    "hires": hires_flag,
                    "policy": {
                        "significant_motion": bool(signif),
                        "policy_used": (policy_used or "none"),
                        "interval_sec": float(interval_sec),
                        "max_frames": (vlm_policy.get("max_frames") if has_v and q_video is not None else None),
                        "silence_hint": {
                            "silence_ratio": float(silence_hint.get("silence_ratio", 0.0)),
                            "is_mostly_silent": bool(silence_hint.get("is_mostly_silent", False)),
                        },
                        "diff_method": diff_method,
                        "diff_threshold": float(diff_threshold),
                        "hist_bins": int(hist_bins),
                        "flow_step": int(flow_step),
                    },
                }
                ok = _safe_put_with_ctrl(q_video, payload_video, q_ctrl, stop)
                if not ok:
                    return  # 退出 A 线程

            # ---- 投音频队列 ----
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

            # 步进
            seg_idx += 1
            t0 += step
            if tail_emitted and max_duration is not None:
                t0 = max_duration

            sleep(0.005)

    except Exception as e:
        logger.error(f"[A] 运行时异常，线程退出：{e}")
    finally:
        logger.info('[A] 线程退出清理完成')
    
   