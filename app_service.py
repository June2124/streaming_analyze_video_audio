from __future__ import annotations

import os
import json
import uuid
import time
import shutil
import threading
import subprocess
from typing import Optional, Dict, Callable
from queue import Queue, Empty
from shutil import which

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles


from streaming_analyze import StreamingAnalyze
from src.all_enum import MODEL, CLOUD_VLM_MODEL_NAME, VLM_SYSTEM_PROMPT_PRESET, DIFF_METHOD, VLM_DETECT_EVENT_LEVEL
from src.configs.vlm_config import VlmConfig
from src.configs.cut_config import CutConfig
from src.configs.asr_config import AsrConfig


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
INDEX_HTML = os.path.join(BASE_DIR, "index.html")

app = FastAPI(title="JetLinksAI Analyze Service", version="0.4.0")

# 同源部署也可以保留 CORS，便于调试跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 演示放开
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- 静态与上传目录 -----------------
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
STATIC_DIR = os.path.join(BASE_DIR, "static")
STATIC_KEYFRAMES_DIR = os.path.join(STATIC_DIR, "keyframes")   # 证据帧实际落地目录
STATIC_KEYFRAMES_URL_PREFIX = "/static/keyframes"              # 前端访问 URL 前缀

# 逐级确保存在（无则创建）
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(STATIC_KEYFRAMES_DIR, exist_ok=True)

# 只有 static 目录存在时才挂载（上面已确保存在，其实这里一定会执行）
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ----------------- SSE Broker -----------------
class SSEBroker:
    def __init__(self, max_queue_size: int = 1000):
        self._subs: Dict[str, Queue[str]] = {}
        self._filters: Dict[str, Optional[Callable[[dict], bool]]] = {}
        self._max = max_queue_size
        self._lock = threading.Lock()

    def subscribe(self, flt: Optional[Callable[[dict], bool]] = None) -> str:
        sid = uuid.uuid4().hex[:8]
        with self._lock:
            self._subs[sid] = Queue(self._max)
            self._filters[sid] = flt
        return sid

    def unsubscribe(self, sid: str):
        with self._lock:
            self._subs.pop(sid, None)
            self._filters.pop(sid, None)

    def publish(self, event: dict):
        line = f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        with self._lock:
            for sid, q in self._subs.items():
                flt = self._filters.get(sid)
                if flt is not None:
                    try:
                        if not flt(event):
                            continue
                    except Exception:
                        continue
                try:
                    q.put_nowait(line)
                except Exception:
                    # 队列满丢弃
                    pass

    def stream(self, sid: str):
        q = self._subs.get(sid)
        if q is None:
            yield "event: end\ndata: {}\n\n"
            return
        try:
            yield f"event: hello\ndata: {json.dumps({'sid': sid})}\n\n"
            while True:
                try:
                    line = q.get(timeout=0.5)
                    yield line
                except Empty:
                    continue
        finally:
            self.unsubscribe(sid)


broker = SSEBroker()


# ----------------- 任务管理 -----------------
class Job:
    def __init__(self, ctrl: StreamingAnalyze, mode: MODEL):
        self.ctrl = ctrl
        self.mode = mode
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.viewer_proc: Optional[subprocess.Popen] = None  # 本地播放器


JOBS: Dict[str, Job] = {}
JOBS_LOCK = threading.Lock()


# ----------------- 事件筛选/改写 -----------------
def _event_payload_min(job_id: str, ev: dict) -> Optional[dict]:
    typ = ev.get("type")
    seg = ev.get("segment_index")
    t0 = ev.get("clip_t0")
    t1 = ev.get("clip_t1")
    ts = ((ev.get("_meta") or {}).get("emit_iso")) or time.strftime("%Y-%m-%dT%H:%M:%S")

    if typ == "vlm_stream_delta":
        text = (ev.get("delta") or "")
        if not text:
            return None
        return {"type": "vlm_delta", "job_id": job_id, "seg": seg, "text": text, "t0": t0, "t1": t1, "ts": ts}

    if typ == "vlm_stream_done":
        text = (ev.get("full_text") or "").strip()
        if text == "":
            return None
        payload = {
            "type": "vlm_done",
            "job_id": job_id,
            "seg": seg,
            "text": text,
            "t0": t0,
            "t1": t1,
            "ts": ts,
        }
        # 把证据帧 URL 透给前端
        evidence_url = ev.get("evidence_image_url")
        if evidence_url:
            payload["evidence_image_url"] = evidence_url
        return payload

    if typ == "asr_stream_done":
        text = (ev.get("full_text") or "").strip()
        return {"type": "asr_done", "job_id": job_id, "seg": seg, "text": text, "t0": t0, "t1": t1, "ts": ts}

    return None


def _should_push(ev_min: dict, job_mode: MODEL) -> bool:
    typ = ev_min.get("type")
    if typ == "asr_done":
        return True
    if job_mode == MODEL.OFFLINE:
        return typ == "vlm_done"
    if job_mode in (MODEL.SECURITY, MODEL.ONLINE):
        return typ == "vlm_done"
    return typ == "vlm_done"


# ----------------- 本地 RTSP 播放 -----------------
def launch_local_rtsp_viewer(rtsp_url: str, title: str) -> Optional[subprocess.Popen]:
    if which("ffplay"):
        try:
            return subprocess.Popen([
                "ffplay", "-rtsp_transport", "tcp", "-autoexit", "-loglevel", "error",
                "-window_title", title, rtsp_url
            ])
        except Exception:
            pass
    if which("vlc"):
        try:
            return subprocess.Popen(["vlc", "--quiet", "--play-and-exit", rtsp_url, "--video-title", title])
        except Exception:
            pass
    return None


def kill_viewer(proc: Optional[subprocess.Popen]):
    if not proc:
        return
    try:
        proc.terminate()
    except Exception:
        pass
    try:
        proc.wait(timeout=2)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


# ----------------- 启动任务 -----------------
def _start_job(job_id: str, ctrl: StreamingAnalyze, mode: MODEL):
    job = Job(ctrl, mode)

    def _runner():
        job.running = True
        try:
            for ev in ctrl.run_stream():
                ev_min = _event_payload_min(job_id, ev)
                if ev_min is None:
                    continue
                if _should_push(ev_min, job.mode):
                    broker.publish(ev_min)
        except Exception as e:
            broker.publish({"type": "job_error", "job_id": job_id, "msg": str(e)})
        finally:
            job.running = False
            broker.publish({"type": "job_end", "job_id": job_id})

    t = threading.Thread(target=_runner, name=f"job-{job_id}", daemon=True)
    job.thread = t
    with JOBS_LOCK:
        JOBS[job_id] = job
    t.start()
    return job_id


# ----------------- 页面与 SSE -----------------
@app.get("/", response_class=HTMLResponse)
def index():
    if os.path.exists(INDEX_HTML):
        return FileResponse(INDEX_HTML, media_type="text/html; charset=utf-8")
    # 兜底：文件不存在时返回一个简单占位页面
    return HTMLResponse("<h3>index.html 未找到。请把前端页放到与 app_service.py 同级目录。</h3>", status_code=200)


@app.get("/events")
def sse_events(job_id: Optional[str] = None):
    sid = broker.subscribe((lambda e: e.get("job_id") == job_id) if job_id else None)
    return StreamingResponse(
        broker.stream(sid),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


# ----------------- 上传离线文件并分析 -----------------
@app.post("/upload")
async def upload_and_analyze(
    file: UploadFile = File(...),
    slice_sec: int = Form(5),
    vlm_system_prompt: Optional[str] = Form("请描述画面内容，突出重点"),
    vlm_model: str = Form(CLOUD_VLM_MODEL_NAME.QWEN_VL_MAX.value),
    vlm_streaming: bool = Form(True),  # OFFLINE 推荐流式（前端只收 delta）
    cut_interval_sec: float = Form(1.0),
    cut_diff_method: str = Form(DIFF_METHOD.BGR_RATIO.value),
    cut_diff_threshold: float = Form(0.50),
    asr_disfluency_removal_enabled: bool = Form(False),
    asr_semantic_punctuation_enabled: bool = Form(False),
    asr_max_sentence_silence: int = Form(1000),
    asr_punctuation_prediction_enabled: bool = Form(True),
    asr_inverse_text_normalization_enabled: bool = Form(False),
):
    if not file.filename:
        raise HTTPException(400, "空文件名")
    job_id = uuid.uuid4().hex[:8]
    dst_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")
    with open(dst_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        vlm_cfg = VlmConfig(
            vlm_system_prompt=vlm_system_prompt,
            vlm_model_name=CLOUD_VLM_MODEL_NAME(vlm_model),
            vlm_streaming=bool(vlm_streaming),
            # 把证据帧静态目录与 URL 前缀传入 B 侧
            vlm_static_keyframe_dir=STATIC_KEYFRAMES_DIR,
            vlm_static_keyframe_url_prefix=STATIC_KEYFRAMES_URL_PREFIX,
        )
        cut_cfg = CutConfig(
            interval_sec=float(cut_interval_sec),
            diff_method=DIFF_METHOD(cut_diff_method),
            diff_threshold=float(cut_diff_threshold),
        )
        asr_cfg = AsrConfig(
            disfluency_removal_enabled=bool(asr_disfluency_removal_enabled),
            semantic_punctuation_enabled=bool(asr_semantic_punctuation_enabled),
            max_sentence_silence=int(asr_max_sentence_silence),
            punctuation_prediction_enabled=bool(asr_punctuation_prediction_enabled),
            inverse_text_normalization_enabled=bool(asr_inverse_text_normalization_enabled),
        )
    except Exception as e:
        raise HTTPException(400, f"配置参数错误：{e}")

    ctrl = StreamingAnalyze(
        url=dst_path,
        mode=MODEL.OFFLINE,
        slice_sec=int(slice_sec),
        enable_b=True,
        enable_c=True,               # 离线有声/无声均可，内部自判
        skew_guard_enabled=None,
        cut_config=cut_cfg,
        vlm_config=vlm_cfg,
        asr_config=asr_cfg,
    )

    _start_job(job_id, ctrl, MODEL.OFFLINE)
    return {"ok": True, "job_id": job_id, "mode": "offline", "file": os.path.basename(dst_path)}


# ----------------- RTSP 安防（支持两个路径） -----------------
def _rtsp_handler(
    rtsp_url: str,
    slice_sec: int,
    vlm_preset: str,
    cut_interval_sec: float,
    cut_diff_method: str,
    cut_diff_threshold: float,
    open_local_player: bool,
    vlm_event_min_level: str,   # 接收字符串，后面做归一化
):
    if not (rtsp_url.startswith("rtsp://") or rtsp_url.startswith("rtsps://")):
        raise HTTPException(400, "rtsp_url 必须以 rtsp:// 或 rtsps:// 开头")
    job_id = uuid.uuid4().hex[:8]

    # 选择预设提示词（统一 JSON 输出）
    try:
        preset_enum = VLM_SYSTEM_PROMPT_PRESET[vlm_preset]
    except Exception:
        raise HTTPException(400, f"未知的 Preset：{vlm_preset}")

    # 归一化筛选等级（大小写兼容）
    try:
        level_enum = VLM_DETECT_EVENT_LEVEL[vlm_event_min_level.strip().upper()]
    except Exception:
        raise HTTPException(400, f"非法的事件等级：{vlm_event_min_level}，可选：LOW/MEDIUM/HIGH/CRITICAL")

    try:
        vlm_cfg = VlmConfig(
            vlm_system_prompt=preset_enum,                   # 直接传入 PRESET
            vlm_model_name=CLOUD_VLM_MODEL_NAME.QWEN3_VL_PLUS,
            vlm_streaming=False,                             # 非流式 → 只会有 vlm_done
            # 把证据帧静态目录与 URL 前缀传入 B 侧
            vlm_static_keyframe_dir=STATIC_KEYFRAMES_DIR,
            vlm_static_keyframe_url_prefix=STATIC_KEYFRAMES_URL_PREFIX,
            vlm_event_min_level=level_enum,                  # 归一化后的枚举
        )
        cut_cfg = CutConfig(
            interval_sec=float(cut_interval_sec),
            diff_method=DIFF_METHOD(cut_diff_method),
            diff_threshold=float(cut_diff_threshold),
        )
    except Exception as e:
        raise HTTPException(400, f"配置参数错误：{e}")

    ctrl = StreamingAnalyze(
        url=rtsp_url,
        mode=MODEL.SECURITY,
        slice_sec=int(slice_sec),
        enable_b=True,
        enable_c=False,            # 安防演示一般禁 ASR
        skew_guard_enabled=False,
        cut_config=cut_cfg,
        vlm_config=vlm_cfg,
    )

    _start_job(job_id, ctrl, MODEL.SECURITY)

    viewer = "disabled"
    if open_local_player:
        proc = launch_local_rtsp_viewer(rtsp_url, title=f"RTSP-{job_id}")
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job:
                job.viewer_proc = proc
        if proc is not None:
            viewer = "ffplay" if which("ffplay") else ("vlc" if which("vlc") else "unknown")

    return {"ok": True, "job_id": job_id, "mode": "security", "rtsp": rtsp_url, "viewer": viewer}


@app.post("/rtsp_and_analyze")
async def rtsp_and_analyze(
    rtsp_url: str = Form(...),
    vlm_preset: str = Form(...),
    slice_sec: int = Form(4),
    cut_interval_sec: float = Form(2.0),
    cut_diff_method: str = Form(DIFF_METHOD.GRAY_MEAN.value),
    cut_diff_threshold: float = Form(0.55),
    open_local_player: bool = Form(True),
    vlm_event_min_level: str = Form("LOW"),   # 字符串入参，后端归一化
):
    return _rtsp_handler(
        rtsp_url, slice_sec, vlm_preset, cut_interval_sec, cut_diff_method, cut_diff_threshold,
        open_local_player, vlm_event_min_level
    )


# 兼容老路径 /rtsp
@app.post("/rtsp")
async def rtsp_compat(
    rtsp_url: str = Form(...),
    slice_sec: int = Form(4),
    cut_interval_sec: float = Form(2.0),
    cut_diff_method: str = Form(DIFF_METHOD.GRAY_MEAN.value),
    cut_diff_threshold: float = Form(0.55),
    open_local_player: bool = Form(True),
    # 老接口无 preset，这里给一个默认更安全的合规 Preset
    vlm_preset: str = Form("OFFICE_PRODUCTIVITY_COMPLIANT"),
    vlm_event_min_level: str = Form("LOW"),   # 字符串入参，后端归一化
):
    return _rtsp_handler(
        rtsp_url, slice_sec, vlm_preset, cut_interval_sec, cut_diff_method, cut_diff_threshold,
        open_local_player, vlm_event_min_level
    )


# ----------------- 停止（不需要 job_id） -----------------
@app.post("/stop")
async def stop_all():
    """停止所有在跑的任务；无需参数。"""
    failed = []
    with JOBS_LOCK:
        ids = list(JOBS.keys())
    for jid in ids:
        try:
            job = JOBS.get(jid)
            if not job:
                continue
            try:
                job.ctrl.force_stop("manual stop")
            except Exception as e:
                failed.append(f"{jid}:{e}")
            try:
                kill_viewer(job.viewer_proc)
                job.viewer_proc = None
            except Exception:
                pass
        finally:
            with JOBS_LOCK:
                JOBS.pop(jid, None)
    if failed:
        return JSONResponse({"ok": False, "error": "; ".join(failed)}, status_code=500)
    return {"ok": True, "stopped": ids}


# ----------------- 本地启动 -----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_service:app", host="0.0.0.0", port=8000, reload=False)
