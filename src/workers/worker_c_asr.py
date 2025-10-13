# -*- coding: utf-8 -*-
from __future__ import annotations


import os
import time
import queue
import wave
import logging
import threading
from typing import Optional, Dict, Any, Iterable

logger = logging.getLogger("src.workers.worker_c_asr")

# ----------- 可选：WebRTC VAD -----------
try:
    import webrtcvad  # type: ignore
    _HAVE_WEBRTCVAD = True
except Exception:
    _HAVE_WEBRTCVAD = False

# ----------- Paraformer SDK -----------
try:
    import dashscope  # type: ignore
    from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult  # type: ignore
    _HAVE_PARA = True
except Exception as e:
    _HAVE_PARA = False
    _PARA_IMPORT_ERR = e


# ========== 工具：读 WAV 并转成 100ms PCM16 块 ==========
def _iter_pcm_frames_from_wav(wav_path: str, *, block_bytes: int = 3200) -> Iterable[bytes]:
    """
    读取 16kHz/mono/PCM16 WAV，按 100ms(3200B) 切块产出纯 PCM16。
    要求：A 侧已标准化为 16kHz/mono/PCM16。
    """
    with wave.open(wav_path, "rb") as wf:
        nch = wf.getnchannels()
        sbytes = wf.getsampwidth()
        rate = wf.getframerate()
        if not (nch == 1 and sbytes == 2 and rate == 16000):
            raise ValueError(f"WAV 参数不符，期待 16k/mono/PCM16，实际: ch={nch}, sw={sbytes}, sr={rate}")
        while True:
            data = wf.readframes(block_bytes // sbytes)  # 样本数 = 字节数 / 2
            if not data:
                break
            yield data


# ========== 可选 VAD：优先 WebRTC，失败用能量阈值 ==========
def _analyze_vad_active_ratio(
    wav_path: str,
    *,
    frame_ms: int = 20,
    aggr: int = 1,
    energy_dbfs_thresh: float = -45.0,
    min_speech_ms: int = 300,
    hangover_ms: int = 300,
) -> Dict[str, Any]:
    """
    返回：
      {
        "is_speech": bool,
        "active_ratio": float(0~1),
        "backend_used": "webrtcvad"|"energy"|"disabled",
        "applied_params": {...}
      }
    """
    applied = {
        "aggr": aggr,
        "energy_dbfs_thresh": energy_dbfs_thresh,
        "min_active_ratio": 0.08,
        "min_speech_ms": min_speech_ms,
        "hangover_ms": hangover_ms,
    }

    try:
        with wave.open(wav_path, "rb") as wf:
            nch, sw, sr = wf.getnchannels(), wf.getsampwidth(), wf.getframerate()
            if not (nch == 1 and sw == 2 and sr == 16000):
                # A 已标准化，这里只做兜底
                return {"is_speech": True, "active_ratio": 1.0, "backend_used": "disabled", "applied_params": applied}
            pcm = wf.readframes(wf.getnframes())
    except Exception as e:
        logger.warning(f"[C] VAD 读取 WAV 失败，降级：{e}")
        return {"is_speech": True, "active_ratio": 1.0, "backend_used": "disabled", "applied_params": applied}

    # 优先 webrtcvad
    if _HAVE_WEBRTCVAD:
        try:
            vad = webrtcvad.Vad(int(aggr))
            step = frame_ms * 16  # 16k mono 16bit -> 每毫秒 16*2=32B，每 20ms 是 640B；但我们按采样数来切
            # 以采样为粒度：frame_ms * sr / 1000 样本，每样本 2B
            frame_bytes = int(sr * frame_ms / 1000) * 2
            total = 0
            act = 0
            for i in range(0, len(pcm), frame_bytes):
                chunk = pcm[i:i + frame_bytes]
                if len(chunk) < frame_bytes:
                    break
                # webrtcvad 要求 10/20/30ms
                if vad.is_speech(chunk, sr):
                    act += 1
                total += 1
            ratio = (act / total) if total else 0.0
            return {
                "is_speech": bool(ratio >= applied["min_active_ratio"]),
                "active_ratio": float(ratio),
                "backend_used": "webrtcvad",
                "applied_params": applied
            }
        except Exception as e:
            logger.warning(f"[C] WebRTC VAD 失败，能量兜底：{e}")

    # 简单能量阈值兜底（RMS-> dBFS）
    try:
        import array, math
        pcm_i16 = array.array("h", pcm)
        if not pcm_i16:
            return {"is_speech": False, "active_ratio": 0.0, "backend_used": "energy", "applied_params": applied}
        # 分帧计算 RMS dBFS
        frame_samples = int(sr * frame_ms / 1000)
        if frame_samples <= 0:
            frame_samples = 320  # 20ms
        total = 0
        act = 0
        for i in range(0, len(pcm_i16), frame_samples):
            frm = pcm_i16[i:i + frame_samples]
            if not frm:
                break
            rms = math.sqrt(sum(int(x) * int(x) for x in frm) / len(frm))
            # 15-bit peak for PCM16 normalized to 1.0 => dbfs approx
            if rms <= 1e-6:
                dbfs = -90.0
            else:
                dbfs = 20.0 * math.log10(rms / 32768.0 * 2.0)  # 粗略
            if dbfs > energy_dbfs_thresh:
                act += 1
            total += 1
        ratio = (act / total) if total else 0.0
        return {
            "is_speech": bool(ratio >= 0.08),
            "active_ratio": float(ratio),
            "backend_used": "energy",
            "applied_params": applied
        }
    except Exception:
        return {"is_speech": True, "active_ratio": 1.0, "backend_used": "disabled", "applied_params": applied}


# ========== Paraformer 回调：收 sentence_end 句级结果 ==========
class _ParaCallback(RecognitionCallback):
    def __init__(self):
        self.sentences = []         # 收到的句对象（只存 sentence_end=True 的）
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._err: Optional[str] = None

    # 录音相关 on_open/on_close 在离线文件模式不需要
    def on_complete(self) -> None:
        self._done.set()

    def on_error(self, message) -> None:
        try:
            req_id = getattr(message, "request_id", None)
            msg = getattr(message, "message", None)
            logger.error(f"[C] Paraformer 错误: request_id={req_id}, err={msg}")
        except Exception:
            logger.error("[C] Paraformer 错误（无法解析 message）")
        self._err = "error"
        self._done.set()

    def on_event(self, result) -> None:
        try:
            sentence = result.get_sentence()  # dict
        except Exception:
            return
        # 只保留句末
        if sentence and sentence.get("sentence_end", False):
            with self._lock:
                self.sentences.append(sentence)

    # 工具
    def wait_done(self, timeout: Optional[float]) -> bool:
        return self._done.wait(timeout=timeout)

    def fetch_sentences(self):
        with self._lock:
            out = list(self.sentences)
            self.sentences.clear()
            return out

    def has_error(self) -> bool:
        return self._err is not None


# ========== 主体：C 线程 ==========
def worker_c_asr(
    q_audio: "queue.Queue[Dict[str, Any]]",
    q_asr: "queue.Queue[Dict[str, Any]]",
    q_ctrl: "queue.Queue[Dict[str, Any]]",
    stop: object,
):
    """
    - 从 q_audio 取离线 wav 段，先做 VAD 预判，再推往 Paraformer 做识别。
    - 只在 sentence_end=True 时投递句级增量，段尾投收尾。
    - 收到 STOP/SHUTDOWN 立即退出。
    """
    logger.info("[C] 线程启动: C-ASR转录")
    # 读取 API Key
    if _HAVE_PARA:
        if "DASHSCOPE_API_KEY" in os.environ:
            dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]

    # 配置
    asr_model = os.getenv("ASR_MODEL", "paraformer-realtime-v2")
    lang = os.getenv("ASR_LANG", "zh")  # 仅作标注
    sr_hz = 16000
    block_bytes = 3200  # 100ms

    # VAD 开关
    vad_enabled = os.getenv("ASR_VAD_ENABLED", "1") == "1"
    vad_aggr = int(os.getenv("ASR_VAD_AGGR", "1"))
    vad_dbfs = float(os.getenv("ASR_VAD_DBFS", "-45"))
    vad_min_ms = int(os.getenv("ASR_VAD_MIN_MS", "300"))
    vad_hang_ms = int(os.getenv("ASR_VAD_HANG_MS", "300"))

    def _q_put_with_retry(obj: Dict[str, Any], *, timeout=0.2, tries=50) -> bool:
        n = 0
        while True:
            # 控制优先
            try:
                ctrl = q_ctrl.get_nowait()
                if ctrl is stop or (isinstance(ctrl, dict) and ctrl.get("type") in ("STOP", "SHUTDOWN")):
                    logger.info("[C] 收到控制队列 STOP ，退出")
                    return False
                # 非 STOP 放回去
                try:
                    q_ctrl.put_nowait(ctrl)
                except queue.Full:
                    pass
            except queue.Empty:
                pass

            try:
                q_asr.put(obj, timeout=timeout)
                return True
            except queue.Full:
                n += 1
                if n >= tries:
                    logger.warning("[C] q_asr 持续拥堵，放弃投递。")
                    return False
            except Exception as e:
                logger.warning(f"[C] q_asr.put 异常，放弃：{e}")
                return False

    try:
        seg_auto_idx = 0
        while True:
            # 取音频任务或 STOP 哨兵
            try:
                item = q_audio.get(timeout=0.1)
            except queue.Empty:
                # 也要监听 STOP
                try:
                    msg = q_ctrl.get_nowait()
                    if msg is stop or (isinstance(msg, dict) and msg.get("type") in ("STOP", "SHUTDOWN")):
                        logger.info("[C] 收到控制队列 STOP ，退出")
                        return
                    # 其他控制丢回去
                    try:
                        q_ctrl.put_nowait(msg)
                    except queue.Full:
                        pass
                except queue.Empty:
                    pass
                continue

            if item is stop:
                logger.info("[C] 收到数据队列 STOP，退出")
                return

            a_path = item.get("path")
            t0 = float(item.get("t0", 0.0))
            t1 = float(item.get("t1", 0.0))
            seg_idx = item.get("segment_index")
            if seg_idx is None:
                seg_idx = seg_auto_idx
                seg_auto_idx += 1

            if not a_path or not os.path.exists(a_path):
                logger.warning(f"[C] 音频文件不存在，跳过 seg#{seg_idx}: {a_path}")
                continue

            # ---------- VAD 预判 ----------
            vad_info = {"is_speech": True, "active_ratio": 1.0, "backend_used": "disabled", "applied_params": {}}
            if vad_enabled:
                try:
                    vad_info = _analyze_vad_active_ratio(
                        a_path, aggr=vad_aggr, energy_dbfs_thresh=vad_dbfs,
                        min_speech_ms=vad_min_ms, hangover_ms=vad_hang_ms
                    )
                except Exception as e:
                    logger.warning(f"[C] VAD 失败，降级直接转写：{e}")

            if not vad_info.get("is_speech", True):
                ok = _q_put_with_retry({
                    "type": "asr_stream_no_speech",
                    "segment_index": seg_idx,
                    "full_text": "[SKIPPED_NO_SPEECH]",
                    "usage": {"vad": vad_info, "silence_hint": item.get("silence_hint")},
                    "model": asr_model,
                    "latency_ms": 0,
                    "lang": lang,
                    "sr_hz": sr_hz,
                    "t0": t0, "t1": t1
                })
                if not ok:
                    return
                continue

            if not _HAVE_PARA:
                logger.error(f"[C] 未安装/导入 Paraformer SDK（dashscope）失败：{_PARA_IMPORT_ERR}")
                # 直接当作无声跳过（或根据需要 raise）
                ok = _q_put_with_retry({
                    "type": "asr_stream_no_speech",
                    "segment_index": seg_idx,
                    "full_text": "[SKIPPED_SDK_NOT_AVAILABLE]",
                    "usage": {"vad": vad_info, "silence_hint": item.get("silence_hint")},
                    "model": "N/A",
                    "latency_ms": 0,
                    "lang": lang,
                    "sr_hz": sr_hz,
                    "t0": t0, "t1": t1
                })
                if not ok:
                    return
                continue

            # ---------- Paraformer：仅句级 ----------
            cb = _ParaCallback()
            recog = Recognition(
                model=asr_model,
                format='pcm',           # 我们发送纯 PCM 块
                sample_rate=sr_hz,
                semantic_punctuation_enabled=True,
                callback=cb
            )
            start_ts = time.time()
            try:
                recog.start()
                for pcm in _iter_pcm_frames_from_wav(a_path, block_bytes=block_bytes):
                    # 控制中断
                    try:
                        ctrl = q_ctrl.get_nowait()
                        if ctrl is stop or (isinstance(ctrl, dict) and ctrl.get("type") in ("STOP", "SHUTDOWN")):
                            logger.info("[C] 收到控制队列 STOP ，退出")
                            return
                        try:
                            q_ctrl.put_nowait(ctrl)
                        except queue.Full:
                            pass
                    except queue.Empty:
                        pass

                    recog.send_audio_frame(pcm)

                recog.stop()  # 发送完当前段
            except Exception as e:
                logger.warning(f"[C] Paraformer 发送/停止异常：{e}")

            # 等待 SDK 完成，取句级结果
            cb.wait_done(timeout=float(os.getenv("ASR_PARA_WAIT_DONE_S", "10.0")))
            if cb.has_error():
                logger.warning("[C] Paraformer 识别报错（已记录），继续后续段。")

            sentences = cb.fetch_sentences()  # 只包含 sentence_end=True 的句对象
            # 投句级增量
            seq = 0
            concat_texts = []
            for s in sentences:
                text = (s.get("text") or "").strip()
                if not text:
                    continue
                concat_texts.append(text)
                seq += 1
                ok = _q_put_with_retry({
                    "type": "asr_stream_delta",
                    "segment_index": seg_idx,
                    "seq": seq,
                    "delta": text,  # 句级增量 = 本句全文
                    "usage": {"vad": vad_info, "silence_hint": item.get("silence_hint")},
                    "model": asr_model,
                    "lang": lang,
                    "sr_hz": sr_hz,
                    "t0": t0, "t1": t1
                })
                if not ok:
                    return

            # 段尾收尾
            latency_ms = int((time.time() - start_ts) * 1000)
            full_text = "".join(concat_texts)
            ok = _q_put_with_retry({
                "type": "asr_stream_done",
                "segment_index": seg_idx,
                "full_text": full_text,
                "usage": {"vad": vad_info, "silence_hint": item.get("silence_hint")},
                "model": asr_model,
                "latency_ms": latency_ms,
                "lang": lang,
                "sr_hz": sr_hz,
                "t0": t0, "t1": t1
            })
            if not ok:
                return

            logger.info("[C] Paraformer 识别完成 seg#%s", seg_idx)

    except Exception as e:
        logger.exception(f"[C] 异常退出：{e}")
    finally:
        logger.info("[C] 线程退出清理完成。")
