# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import time
import uuid
import shutil
from typing import List, Dict, Any, Optional, Iterator, Tuple, Callable
from queue import Queue, Empty
from collections import deque

from src.utils import logger_utils
from src.configs.vlm_config import VlmConfig
from src.all_enum import (
    VLM_SYSTEM_PROMPT_AUTO_ADD,
    CLOUD_VLM_MODEL_NAME,
    VLM_SYSTEM_PROMPT_PRESET,
    VLM_DETECT_EVENT_LEVEL,
)

logger = logger_utils.get_logger(__name__)

# ---- DashScope 原生 SDK ----
try:
    import dashscope
    from dashscope import MultiModalConversation
    _HAS_MMC = True
except Exception as _e:
    logger.error("[B] 未检测到 DashScope SDK 或导入失败，请先：pip install dashscope ；error=%s", _e)
    _HAS_MMC = False


# ----------------- 严格事件数组 Schema（任务型使用） -----------------
EVENT_ARRAY_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "EventArray",
        "schema": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "describe": {"type": "string", "maxLength": 15},
                    "level": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH", "CRITICAL"]},
                    "suggestion": {"type": "string", "maxLength": 15},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                },
                "required": ["type", "describe", "level", "suggestion", "confidence"],
                "additionalProperties": False
            }
        }
    }
}

# 等级顺序（用于筛选）
_LEVEL_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


# ----------------- 文件与URI辅助 -----------------
def _is_http_url(p: str) -> bool:
    return isinstance(p, str) and p.lower().startswith(("http://", "https://"))


def _to_file_uri(path_or_uri: str) -> Optional[str]:
    if not path_or_uri:
        return None
    if _is_http_url(path_or_uri) or path_or_uri.lower().startswith("file://"):
        return path_or_uri
    abs_path = os.path.abspath(path_or_uri).replace("\\", "/")
    return f"file://{abs_path}"


def _exists_local_from_uri(file_uri_or_path: str) -> bool:
    if not file_uri_or_path:
        return False
    if _is_http_url(file_uri_or_path):
        return True
    if file_uri_or_path.lower().startswith("file://"):
        local = file_uri_or_path[len("file://"):]
        return os.path.exists(local)
    return os.path.exists(file_uri_or_path)


def _uri_to_local_path(file_uri: str) -> Optional[str]:
    """file://... -> 本地路径；否则 None"""
    if not file_uri:
        return None
    if file_uri.lower().startswith("file://"):
        return file_uri[len("file://"):]
    return None


# ----------------- Prompt 拼装工具 -----------------
def _build_msgs_for_video(video_uri: str, user_prompt: Optional[str]) -> List[Dict[str, Any]]:
    prompt = user_prompt or "请描述视频主要内容（人物、场景、事件）。"
    return [{"role": "user", "content": [{"video": video_uri}, {"text": prompt}]}]


def _build_msgs_for_images(img_uris: List[str], user_prompt: Optional[str]) -> List[Dict[str, Any]]:
    prompt = user_prompt or "请描述这些关键帧展示的主要事件或画面要点。"
    content: List[Dict[str, Any]] = [{"text": prompt}]
    # 采用原始顺序 append（而非倒序 insert）
    for u in img_uris:
        content.append({"image": u})
    return [{"role": "user", "content": content}]


def _ensure_json_keyword(s: str) -> str:
    """确保提示词包含 JSON 关键词，满足 Qwen JSON 模式的服务端硬校验。"""
    if not s:
        return "仅输出JSON。"
    return s if ("json" in s.lower()) else (s.rstrip() + "\n\n仅输出JSON。")


# ----------------- STOP 感知 & 带 STOP 检查的安全投递 -----------------
def _ctrl_stop_requested(q_ctrl: Queue | None, stop: object | None) -> bool:
    """无阻塞探测控制队列里是否有 STOP/SHUTDOWN 指令。"""
    if q_ctrl is None or stop is None:
        return False
    try:
        while True:
            msg = q_ctrl.get_nowait()
            if (msg is stop) or (isinstance(msg, dict) and msg.get("type") in ("STOP", "SHUTDOWN")):
                logger.warning("[B] 检测到控制队列 STOP 哨兵, 停止本次投递")
                return True
            # 不是停止指令，塞回去避免丢消息
            try:
                q_ctrl.put_nowait(msg)
            except Exception:
                pass
            return False
    except Empty:
        return False


def _q_put_with_retry(
    q: Queue,
    obj: Any,
    *,
    tries: int = 3,
    timeout: float = 0.5,
    q_ctrl: Optional[Queue] = None,
    stop: object = None,
    drop_on_stop: bool = True,
    drop_on_timeout: bool = True,
) -> bool:
    """
    - 每次 put 前、以及每次超时后均检查 STOP。
    - drop_on_stop=True：检测到 STOP 立即放弃（返回 False）。
    - drop_on_timeout=True：重试耗尽后放弃（返回 False）。
    """
    if _ctrl_stop_requested(q_ctrl, stop):
        return not drop_on_stop  # 通常返回 False

    for i in range(1, tries + 1):
        try:
            q.put(obj, timeout=timeout)
            return True
        except Exception as e:
            logger.warning(f"[B] q_vlm.put 超时（第{i}/{tries}次）：{e}")
            if _ctrl_stop_requested(q_ctrl, stop):
                return not drop_on_stop

    if drop_on_timeout:
        return False

    try:
        q.put(obj, timeout=0.2)
        return True
    except Exception:
        return False


# ----------------- DashScope Streaming/Non-Streaming 辅助 -----------------
def _parse_text_and_usage_from_resp(resp: dict) -> Tuple[str, Optional[dict]]:
    out = (resp or {}).get("output") or {}
    usage = (resp or {}).get("usage") or None
    text = ""
    try:
        choices = out.get("choices") or []
        if choices:
            msg = (choices[0] or {}).get("message") or {}
            content = msg.get("content") or []
            parts = [it.get("text") for it in content if isinstance(it, dict) and it.get("text")]
            text = "".join(parts) if parts else ""
    except Exception:
        pass
    return text, usage


def call_vlm_unified(
    model_name: str,
    messages: list[dict],
    *,
    want_streaming: bool,
    on_heartbeat: Optional[Callable[[], bool]] = None,  # 返回 True 表示应中断
    open_timeout_s: float = 15.0,
    nonstream_max_tries: int = 2,
    nonstream_backoff_base: float = 1.6,
    response_format: Optional[dict] = None,  # 任务型：json_schema；描述型：None
):
    """
    - want_streaming=True：先试流式，失败则本次回退到非流式
    - want_streaming=False：直接非流式
    返回：
      - 若流式成功：("stream", iterator, None)
      - 若非流式：("nonstream", None, (full_text, usage))
    """
    if not _HAS_MMC:
        raise RuntimeError("DashScope SDK 不可用")

    # 尝试流式（仅用于描述型）
    if want_streaming:
        try:
            def _open_stream_call():
                t0 = time.time()
                while True:
                    try:
                        return MultiModalConversation.call(
                            model=model_name,
                            messages=messages,
                            stream=True,
                            incremental_output=True,
                        )
                    except Exception as e:
                        if time.time() - t0 > open_timeout_s:
                            raise e
                        time.sleep(0.2)

            stream_obj = _open_stream_call()

            def _iter() -> Iterator[Tuple[Optional[str], Optional[dict]]]:
                usage_cache: Optional[dict] = None
                for rsp in stream_obj:
                    if on_heartbeat and on_heartbeat():
                        break
                    delta, usage_part = None, None
                    try:
                        out = rsp.get("output") or {}
                        choices = out.get("choices") or []
                        if choices:
                            msg = (choices[0] or {}).get("message") or {}
                            content = msg.get("content") or []
                            parts = [it.get("text") for it in content if isinstance(it, dict) and it.get("text")]
                            if parts:
                                delta = "".join(parts)
                        u = rsp.get("usage") or {}
                        if u:
                            usage_cache = {
                                "prompt_tokens": u.get("input_tokens") or u.get("prompt_tokens"),
                                "completion_tokens": u.get("output_tokens") or u.get("completion_tokens"),
                                "total_tokens": u.get("total_tokens"),
                            }
                            usage_part = usage_cache
                    except Exception:
                        pass
                    yield delta, usage_part

            return "stream", _iter(), None
        except Exception as e:
            logger.warning("[B] 流式失败，本次回退非流式：%s", e)

    # 非流式（或回退）：直接传 response_format
    last_err = None
    for i in range(1, nonstream_max_tries + 1):
        try:
            resp = MultiModalConversation.call(
                model=model_name,
                messages=messages,
                stream=False,
                response_format=response_format if response_format else None,
            )
            ft, usage = _parse_text_and_usage_from_resp(resp)
            return "nonstream", None, (ft, usage)
        except Exception as e:
            last_err = e
            if i < nonstream_max_tries:
                sleep_s = (nonstream_backoff_base ** i) + min(0.6, 0.2 * i)
                logger.warning("[B] 非流式失败 %d/%d，%.1fs后重试：%s", i, nonstream_max_tries, sleep_s, e)
                time.sleep(sleep_s)
    raise last_err


# ----------------- 文本去重/归一化 -----------------
def _normalize_lines(s: str) -> List[str]:
    if not s:
        return []
    raw_lines = [ln.strip() for ln in s.replace("\r", "").split("\n")]
    out = []
    for ln in raw_lines:
        if not ln:
            continue
        for p in ("- ", "• ", "* ", "· "):
            if ln.startswith(p):
                ln = ln[len(p):].strip()
                break
        ln = ln.strip(" \u3000")
        if ln:
            out.append(ln)
    return out


def _join_as_bullets(lines: List[str]) -> str:
    return "\n".join("- " + ln for ln in lines)


def _build_history_context_text(history: List[str], max_chars: int) -> str:
    if not history:
        return ""
    buf = []
    remain = max_chars
    for one in reversed(history):
        if not one:
            continue
        t = one.strip()
        if not t:
            continue
        if len(t) + 1 > remain:
            break
        buf.append(t)
        remain -= (len(t) + 1)
    if not buf:
        return ""
    return "以下为“历史小结”（近到远，最多 N 段）：\n" + ("\n---\n".join(buf))


# ----------------- 统一事件发包（已加 STOP 感知） -----------------
def _emit_delta(
    q_vlm: Queue,
    seg_idx: int,
    delta: str,
    seq: int,
    model: str,
    item: dict,
    *,
    streaming: bool,
    q_ctrl: Optional[Queue] = None,
    stop: object = None,
) -> None:
    _q_put_with_retry(
        q_vlm,
        {
            "type": "vlm_stream_delta",
            "segment_index": seg_idx,
            "delta": delta,
            "seq": seq,
            "model": model,
            "streaming": bool(streaming),
            "produce_ts": time.time(),
            "clip_t0": item.get("t0"), "clip_t1": item.get("t1"),
            "frame_pts": item.get("frame_pts") or [],
            "frame_indices": item.get("frame_indices") or [],
            "small_video_fps": item.get("policy", {}).get("encode", {}).get("fps"),
            "origin_policy": (item.get("policy") or {}).get("policy_used"),
        },
        q_ctrl=q_ctrl,
        stop=stop,
        drop_on_stop=True,
        drop_on_timeout=True,
    )


def _emit_done(
    q_vlm: Queue,
    seg_idx: int,
    full_text: str,
    model: str,
    item: dict,
    *,
    usage: dict | None,
    latency_ms: int,
    streaming: bool,
    suppressed_dup: bool | None = None,
    ctx_rounds: int | None = None,
    evidence_image: Optional[str] = None,      # file:// 源URI（可选）
    evidence_image_url: Optional[str] = None,  # 静态URL（可选）
    q_ctrl: Optional[Queue] = None,
    stop: object = None,
) -> None:
    payload = {
        "type": "vlm_stream_done",
        "segment_index": seg_idx,
        "full_text": full_text or "",
        "usage": usage,
        "model": model,
        "streaming": bool(streaming),
        "latency_ms": latency_ms,
        "clip_t0": item.get("t0"), "clip_t1": item.get("t1"),
        "frame_pts": item.get("frame_pts") or [],
        "frame_indices": item.get("frame_indices") or [],
        "small_video_fps": item.get("policy", {}).get("encode", {}).get("fps"),
        "origin_policy": (item.get("policy") or {}).get("policy_used"),
        "produce_ts": time.time(),
        # 诊断
        "suppressed_dup": bool(suppressed_dup) if suppressed_dup is not None else None,
        "ctx_rounds": ctx_rounds,
    }
    if evidence_image:
        payload["evidence_image"] = evidence_image
    if evidence_image_url:
        payload["evidence_image_url"] = evidence_image_url

    _q_put_with_retry(
        q_vlm,
        payload,
        q_ctrl=q_ctrl,
        stop=stop,
        drop_on_stop=True,
        drop_on_timeout=True,
    )


# ----------------- 证据帧选择（窗口级一张） -----------------
def _pick_evidence_frame(keyframes: List[str]) -> Optional[str]:
    """
    简单稳妥策略：优先取靠后的关键帧（减少转场/空镜）。
    返回 file:// URI；若无可用则 None。
    """
    if not keyframes:
        return None
    for p in reversed(keyframes):
        u = _to_file_uri(p)
        if u and _exists_local_from_uri(u):
            return u
    for p in keyframes:
        u = _to_file_uri(p)
        if u and _exists_local_from_uri(u):
            return u
    return None


# ----------------- 将证据帧导出到静态目录并生成 URL -----------------
def _export_evidence_to_static(
    evidence_uri: Optional[str],
    seg_idx: int,
    cfg: VlmConfig,
) -> Optional[str]:
    """
    把 file:// 证据帧拷贝到 cfg.vlm_static_keyframe_dir 下，返回可对外访问的 URL：
      <cfg.vlm_static_keyframe_url_prefix>/<unique_filename>
    若配置缺失/源文件不存在/拷贝失败 -> 返回 None。
    """
    if not evidence_uri:
        return None

    static_dir = getattr(cfg, "vlm_static_keyframe_dir", None)
    url_prefix = getattr(cfg, "vlm_static_keyframe_url_prefix", None)
    if not static_dir or not url_prefix:
        return None

    src = _uri_to_local_path(evidence_uri)
    if not src or not os.path.exists(src):
        return None

    try:
        os.makedirs(static_dir, exist_ok=True)
    except Exception as e:
        logger.warning("[B] 创建静态目录失败：%s", e)
        return None

    base = os.path.basename(src)
    # 生成唯一文件名（避免覆盖）：seg3_1690000000000_abcd1234_base.jpg
    unique = f"seg{seg_idx}_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}_{base}"
    dst = os.path.join(static_dir, unique)

    try:
        shutil.copy2(src, dst)
    except Exception as e:
        logger.warning("[B] 复制证据帧到静态目录失败：%s", e)
        return None

    url = f"{url_prefix.rstrip('/')}/{unique}"
    return url.replace("\\", "/")


# ----------------- 任务型：按阈值筛选事件数组 -----------------
def _filter_events_by_level_json_text(json_text: str, min_level: Optional[str]) -> Tuple[str, Optional[str]]:
    """
    输入：模型返回的 JSON 文本（数组），以及最小等级（如 "MEDIUM"）。
    输出：(过滤后的 JSON 文本, max_level)；如果解析失败则原样返回、不做筛选。
    """
    if not min_level or min_level not in _LEVEL_ORDER:
        return json_text, None
    try:
        import json
        arr = json.loads(json_text)
        if not isinstance(arr, list):
            return json_text, None
        keep = []
        max_level_name = None
        max_level_val = -1
        th = _LEVEL_ORDER[min_level]
        for it in arr:
            if not isinstance(it, dict):
                continue
            lv = str(it.get("level", "")).upper()
            if _LEVEL_ORDER.get(lv, -1) >= th:
                keep.append(it)
                v = _LEVEL_ORDER[lv]
                if v > max_level_val:
                    max_level_val = v
                    max_level_name = lv
        import json as _json
        return (_json.dumps(keep, ensure_ascii=False), max_level_name)
    except Exception:
        return json_text, None


# ----------------- B 线程主体 -----------------
def worker_b_vlm(q_video: Queue, q_vlm: Queue, q_ctrl: Queue, stop: object, vlm_config: Optional[VlmConfig] = None):
    """
    VLM B 线程（视觉语义解析）
    - 使用 compose_vlm_system_prompt 构造 system 提示词
    - 当上游未提供提示词时自动设置默认 system prompt
    - 描述型：按 vlm_streaming 决定是否流式；流式失败当次回退非流式
    - 任务型 preset：强制非流式 + 严格 json_schema，默认不拼历史、不做去重/改写（保证纯 JSON）
      · 可通过 vlm_config.vlm_task_history_enabled 开启“拼历史”
      · 可通过 vlm_config.vlm_event_min_level（如 "MEDIUM"）按等级筛选事件
      · 事件非空则挑选一张证据帧并随 done 一起下发 evidence_image 与 evidence_image_url
    """

    running = False
    paused = False
    max_frames_cap: Optional[int] = None

    # VLM配置类（先兜底实例化，再校验/取值）
    vlm_config = vlm_config or VlmConfig()
    if not isinstance(vlm_config.vlm_model_name, CLOUD_VLM_MODEL_NAME):
        raise TypeError("vlm_model_name 必须是 CLOUD_VLM_MODEL_NAME 枚举实例")
    model_name = vlm_config.vlm_model_name.value

    # 是否任务型（Preset）
    is_task = isinstance(vlm_config.vlm_system_prompt, VLM_SYSTEM_PROMPT_PRESET)

    # 任务型是否拼历史（唯一事实来源：配置字段）
    task_history_enabled = bool(getattr(vlm_config, "vlm_task_history_enabled", False))

    # 任务型最小等级筛选（低于阈值的事件将被丢弃，不作为分析结果对外输出）
    raw_min_level = getattr(vlm_config, "vlm_event_min_level", None)
    if isinstance(raw_min_level, VLM_DETECT_EVENT_LEVEL):
        event_min_level: Optional[str] = raw_min_level.name
    elif isinstance(raw_min_level, str):
        event_min_level = raw_min_level.strip().upper()
    else:
        event_min_level = None
    if event_min_level and event_min_level not in _LEVEL_ORDER:
        logger.warning("[B] 配置的 vlm_event_min_level=%r 无效，将忽略筛选。", raw_min_level)
        event_min_level = None

    # -------- 构造最终 system prompt --------
    if not vlm_config.vlm_system_prompt:
        # 未传入 → 走“描述型”默认 + DESCRIPTIVE 后缀
        default_prompt = "请对当前视频或关键帧进行描述。"
        sys_text = default_prompt + "\n\n" + VLM_SYSTEM_PROMPT_AUTO_ADD.DESCRIPTIVE.value
    else:
        if is_task and not task_history_enabled:
            # 任务型 + 不拼历史 → 只用 Preset 原文，不附加任何模板/规则
            sys_text = vlm_config.vlm_system_prompt.value.strip()
        else:
            # 描述型；或 任务型+允许拼历史 → 使用模板拼接
            sys_text = compose_vlm_system_prompt(vlm_config)

    logger.info(f"[B] 最终 system prompt:\n{sys_text}")
    logger.info(f"[B] is_task={is_task}, task_history_enabled={task_history_enabled}, event_min_level={event_min_level}")

    # 历史上下文存储 & 去重
    if is_task:
        # 任务型：不做去重/改写；是否拼历史由开关控制
        dedup_enabled = False
        if task_history_enabled:
            hist_max_rounds = 30
            hist_max_chars = 4000
            history_deque: deque[str] = deque(maxlen=hist_max_rounds)
        else:
            hist_max_rounds = 0
            hist_max_chars = 0
            history_deque = deque(maxlen=0)
    else:
        # 描述型：允许历史与去重
        dedup_enabled = True
        hist_max_rounds = 30
        hist_max_chars = 4000
        history_deque: deque[str] = deque(maxlen=hist_max_rounds)

    def _poll_ctrl_heartbeat() -> bool:
        try:
            msg = q_ctrl.get_nowait()
        except Empty:
            return False
        if msg is stop:
            return True
        if isinstance(msg, dict) and msg.get("type") in ("STOP", "SHUTDOWN"):
            return True
        try:
            q_ctrl.put_nowait(msg)
        except Exception:
            pass
        return False

    # 主循环
    try:
        while True:
            if not running or paused:
                try:
                    msg = q_ctrl.get(timeout=0.2)
                except Empty:
                    continue
                if msg is stop:
                    logger.info("[B] 收到 STOP，退出")
                    return
                if isinstance(msg, dict):
                    typ = msg.get("type")
                    if typ in ("START", "RESUME"):
                        running, paused = True, False
                        logger.info("[B] 启动视觉解析")
                    elif typ == "PAUSE":
                        paused = True
                        logger.info("[B] 暂停视觉解析")
                    elif typ == "STOP":
                        logger.info("[B] 收到 STOP，退出。")
                        return
                continue

            # -------- 取片段 --------
            try:
                item = q_video.get(timeout=0.1)
            except Empty:
                continue

            try:
                if item is stop:
                    logger.info("[B] 数据队列 STOP，退出")
                    return

                seg_idx = int(item.get("segment_index", -1))
                small_video = item.get("small_video")
                keyframes: List[str] = item.get("keyframes") or []
                t0, t1 = item.get("t0"), item.get("t1")

                messages: List[Dict[str, Any]] = []

                if small_video:
                    vuri = _to_file_uri(small_video)
                    if vuri and _exists_local_from_uri(vuri):
                        messages = _build_msgs_for_video(vuri, user_prompt=None)
                if not messages:
                    cap = max_frames_cap or 12
                    img_uris = []
                    for p in keyframes[:cap]:
                        u = _to_file_uri(p)
                        if u and _exists_local_from_uri(u):
                            img_uris.append(u)
                    if not img_uris:
                        logger.warning(f"[B] seg#{seg_idx} 无可用的视频/图片 URI，跳过该段。")
                        continue
                    messages = _build_msgs_for_images(img_uris, user_prompt=None)

                # 在消息最前插入 system prompt
                messages.insert(0, {"role": "system", "content": [{"text": sys_text}]})

                # user级历史上下文（任务型默认不拼，除非显式开启）
                if (not is_task) or (is_task and task_history_enabled):
                    history_text = _build_history_context_text(list(history_deque), hist_max_chars)
                    if history_text:
                        messages.append({"role": "user", "content": [{"text": history_text}]})

                # ---- 选择调用模式 ----
                want_streaming = bool(vlm_config.vlm_streaming) and (not is_task)
                response_fmt = EVENT_ARRAY_SCHEMA if is_task else None

                t_start = time.time()
                try:
                    mode, iter_pair, nonstream_pair = call_vlm_unified(
                        model_name,
                        messages,
                        want_streaming=want_streaming,
                        on_heartbeat=_poll_ctrl_heartbeat,
                        response_format=response_fmt,
                    )
                except Exception as e:
                    logger.error(f"[B] VLM 调用失败（seg#{seg_idx}）：{e}")
                    _emit_done(
                        q_vlm, seg_idx,
                        full_text=f"[VLM_BACKEND_ERROR] {e}",
                        model=model_name, item=item,
                        usage=None,
                        latency_ms=int((time.time() - t_start) * 1000),
                        streaming=False,
                        suppressed_dup=None,
                        ctx_rounds=len(history_deque),
                        q_ctrl=q_ctrl, stop=stop
                    )
                    continue

                usage = None
                final_text = ""
                suppressed = False

                if mode == "stream":
                    # 流式路径
                    seq = 1
                    buf: List[str] = []
                    for delta, usage_part in iter_pair:  # type: ignore
                        if delta:
                            buf.append(delta)
                            _emit_delta(q_vlm, seg_idx, delta, seq, model_name, item,
                                        streaming=True, q_ctrl=q_ctrl, stop=stop)
                            seq += 1
                        if usage_part:
                            usage = usage_part
                    final_text = "".join(buf)
                    streaming_flag = True
                else:
                    # 非流式路径
                    final_text, usage = nonstream_pair or ("", None)  # type: ignore
                    streaming_flag = False

                # ---- 收尾：任务型可选做“按等级筛选 + 证据帧”；描述型才做去重/小结 ----
                evidence_uri: Optional[str] = None
                evidence_url: Optional[str] = None
                out_text = final_text

                if is_task:
                    # 任务型：按阈值过滤
                    if event_min_level:
                        filtered_text, _max_level = _filter_events_by_level_json_text(final_text, event_min_level)
                        if filtered_text != final_text:
                            logger.info("[B] seg#%d 事件按阈值筛选：min=%s", seg_idx, event_min_level)
                        out_text = filtered_text

                    # 若筛后仍有事件 -> 选证据帧并导出到静态目录
                    try:
                        import json
                        arr = json.loads(out_text)
                        if isinstance(arr, list) and len(arr) > 0:
                            evidence_uri = _pick_evidence_frame(keyframes)  # 窗口级一张
                            evidence_url = _export_evidence_to_static(evidence_uri, seg_idx, vlm_config)
                        else:
                            logger.info("[B] seg#%d 事件为空，跳过证据帧导出", seg_idx)
                    except Exception as e:
                        logger.warning("[B] seg#%d 事件解析失败（保持原样透传）：%s", seg_idx, e)

                    suppressed = False  # 任务型不判重
                else:
                    # 描述型：允许历史与去重
                    if dedup_enabled and history_deque:
                        cur_lines = _normalize_lines(final_text)
                        hist_lines = [ln for h in history_deque for ln in _normalize_lines(h)]
                        new_lines = [ln for ln in cur_lines if ln not in set(hist_lines)]
                        out_text = _join_as_bullets(new_lines)
                        suppressed = (not out_text.strip())
                        if out_text.strip():
                            history_deque.append(out_text)

                _emit_done(
                    q_vlm, seg_idx,
                    full_text=out_text,
                    model=model_name, item=item,
                    usage=usage,
                    latency_ms=int((time.time() - t_start) * 1000),
                    streaming=streaming_flag,
                    suppressed_dup=suppressed,
                    ctx_rounds=len(history_deque),
                    evidence_image=evidence_uri,          # 本地关键帧存放路径 file://
                    evidence_image_url=evidence_url,      # 供前端访问的关键帧静态 URL
                    q_ctrl=q_ctrl, stop=stop
                )

            finally:
                try:
                    q_video.task_done()
                except Exception:
                    pass
    finally:
        logger.info("[B] 线程退出清理完成。")


def compose_vlm_system_prompt(vlm_config: VlmConfig) -> str:
    """
    根据传入的 VlmConfig 构造完整的 system 级提示词。
    1. 若上游传入的 system_prompt 是字符串 → 视为“描述型”，自动在末尾追加 DESCRIPTIVE 规则模板。
    2. 若上游传入的是 VLM_SYSTEM_PROMPT_PRESET 枚举 → 视为“任务型”，自动在末尾追加 TASK_ORIENTED 规则模板。
       *若 RTSP 任务型且 vlm_task_history_enabled=False, 则外层逻辑会绕过本函数, 直接使用 Preset 原文*
    3. 若上游传入为空串或 None → 自动 fallback 为 “请对当前视频或关键帧进行描述。” + DESCRIPTIVE。
    """
    prompt = vlm_config.vlm_system_prompt

    # 情况1：空串 → 自动默认
    if not prompt:
        base = "请对当前视频或关键帧进行描述。"
        addon = VLM_SYSTEM_PROMPT_AUTO_ADD.DESCRIPTIVE.value
        return f"{base}\n\n{addon}"

    # 情况2：上游传入字符串 → 描述型
    if isinstance(prompt, str):
        base = prompt.strip()
        addon = VLM_SYSTEM_PROMPT_AUTO_ADD.DESCRIPTIVE.value
        return f"{base}\n\n{addon}"

    # 情况3：任务型枚举 → 拼接任务型后缀
    if isinstance(prompt, VLM_SYSTEM_PROMPT_PRESET):
        base = prompt.value.strip()
        addon = VLM_SYSTEM_PROMPT_AUTO_ADD.TASK_ORIENTED.value
        return f"{base}\n\n{addon}"

    # 情况4：兜底
    return VLM_SYSTEM_PROMPT_AUTO_ADD.DESCRIPTIVE.value
