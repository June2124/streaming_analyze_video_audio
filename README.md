# Streaming Analyze (A/B/C) — 端到端音视频流式理解

> 切片（A）→ 视觉解析（B, VLM）→ 语音转写（C, ASR）  
> 支持离线文件与实时 RTSP，提供 **同步流式生成器**、**一次性汇总**、与 **自定义回调** 三种使用方式。

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue.svg">
  <img alt="ffmpeg" src="https://img.shields.io/badge/FFmpeg-required-green.svg">
  <img alt="license" src="https://img.shields.io/badge/license-Apache--2.0-lightgrey.svg">
</p>

---

## ✨ 功能亮点

- **多模态并行**：视频送 VLM、音频送 ASR，主控统一对齐与节流  
- **三种调用模式**：
  - `run_stream()`：边跑边拿 **原始事件**（推荐）
  - `run_and_return()`：任务完成后一次性返回结果（离线友好）
  - **自定义回调**：接入你自己的 UI/消息总线
- **时间对齐**：事件携带 `_meta.emit_ts/emit_iso` 与媒体时间 `t0/t1`，便于拼接上下文  
- **可配置策略**：离线/在线/安防三套关键帧/小视频策略可切换  
- **稳健收尾**：慢停/快停机制，异常自动广播 STOP

---

## 🧩 体系结构

```
            +---------------------------+
            |   StreamingAnalyze 主控   |
            |  (管控/监控/快慢停/节流)  |
            +-----+---------------------+
                  | START / STOP / MODE / SLICE
       ┌──────────┴──────────┐
       |                      |
   [A 切片/标准化]        输出消费者(异步)
       |                      |
  q_video ───────────────► OUT-VLM → 回调/打印/汇总/事件队列
  q_audio ───────────────► OUT-ASR → 回调/打印/汇总/事件队列
       |                      ^
       v                      |
   [B VLM 解析]          run_stream() 迭代器
   [C ASR 转写]
```

- **A**：切窗 + 标准化，按策略产出“小视频/关键帧（给 B）”与“WAV（给 C）”
- **B**：视觉模型流式增量（delta）与收尾（done）
- **C**：ASR 句级收尾（默认；如需字级可改 C 侧策略）
- **主控**：统一管理线程、控制面消息、STOP 策略、对齐/节流（`TranscriptPlaybackSkewController`）

---

## 📦 目录结构（核心文件）

```
.
├─ streaming_analyze.py          # 主控：A/B/C 管线、三种使用方式、便捷回调
├─ run_stream_example.py        # 使用示例
├─ src/
│  ├─ workers/
│  │  ├─ worker_a_cut.py        # A: 切片/标准化/策略/关键帧/小视频
│  │  ├─ worker_b_vlm.py        # B: VLM 流式解析（DashScope MMC）
│  │  └─ worker_c_asr.py        # C: ASR 句级识别（Paraformer）
│  ├─ utils/
│  │  ├─ ffmpeg_utils.py        # FFmpeg音视频标准化、静音探测、切窗等工具
│  │  ├─ skew_guard.py          # TranscriptPlaybackSkewController
│  │  └─ logger_utils.py
│  └─ all_enum.py               # MODEL / SOURCE_KIND
```

---

## 🔧 前置依赖

- **Python** 3.10+
- **FFmpeg**（必须）  
  - Windows：把 `ffmpeg.exe` 加入 PATH  
  - macOS：`brew install ffmpeg`  
  - Linux：`apt/yum` 安装
- **SDK**
  - 视觉：`pip install dashscope`
  - ASR（Paraformer，可选）：`pip install dashscope`（与上相同）
  - 可选 VAD：`pip install webrtcvad-wheels`（Windows 可用的轮子）

---

## ⚙️ 环境变量（常用）

> 可写入 `.env` 或直接在环境中设置

| 变量 | 说明 | 默认 |
|---|---|---|
| `DASHSCOPE_API_KEY` | 阿里云百炼 API Key | — |
| `VLM_MODEL_NAME` | VLM 模型名（B） | `qwen3-vl-plus` |
| `VLM_USER_PROMPT` | 全局提示词（若不设，按模式默认） | — |
| `VLM_USER_PROMPT_OFFLINE/ONLINE/SECURITY` | 分模式提示词 | 内置默认 |
| `ASR_MODEL` | Paraformer 模型名（C） | `paraformer-realtime-v2` |
| `ASR_VAD_ENABLED` | 是否启用内置 VAD 预判 | `1` |
| `EMIT_MAX_SKEW_S` | 跨通道最大视觉领先 | `3.0` |
| `EMIT_RATE_LIMIT_HZ` | 消费限速（Hz） | `8.0` |

---

## 🚀 快速开始

### 1) 边跑边拿：`run_stream()`（推荐）

```python
from streaming_analyze import StreamingAnalyze
from src.all_enum import MODEL

# ❗ 示例 RTSP（占位）：请替换为你的真实相机地址
# e.g. "rtsp://user:pass@192.168.1.10:554/Streaming/Channels/101"
RTSP_URL = "rtsp://user:pass@camera-host:554/live/stream"  # <--- 替换为真实地址

ctrl = StreamingAnalyze(
    url=RTSP_URL,
    mode=MODEL.ONLINE,   # ONLINE / OFFLINE / SECURITY
    slice_sec=5,
    enable_b=True,       # 开 VLM
    enable_c=True,        # 开 ASR
    ew_guard_enabled=None #关闭对齐与节流
)

for ev in ctrl.run_stream(print_vlm=False, print_asr=False, max_secs=60):
    # 每条事件都包含 _meta.emit_ts / _meta.emit_iso（事件产生时间）
    # VLM: type in {"vlm_stream_delta","vlm_stream_done"}，含 seg 媒体时间 t0/t1
    # ASR: type == "asr_stream_done"（句级），含 seg 媒体时间 t0/t1 与（如可用）句内时间戳
    print(ev)

# 可在任意时刻：
# ctrl.force_stop("manual stop")
```

### 2) 一次性返回：`run_and_return()`（离线友好）

```python
from streaming_analyze import StreamingAnalyze
from src.all_enum import MODEL

ctrl = StreamingAnalyze(
    url=r"D:\streaming_analyze_video_audio\static\video\RAG_Video_with_sound_test.mp4",
    mode=MODEL.OFFLINE,
    slice_sec=5,
    enable_b=True,
    enable_c=True,
    ew_guard_enabled=None #关闭对齐与节流
)

result = ctrl.run_and_return(print_vlm=False, print_asr=False)
# 结构：
# {
#   "vlm": {"deltas": [...], "dones": [...]},
#   "asr": {"dones": [...]}
# }
print(result)
```

### 3) 简单打印：`run_simple()`（演示/调试）

```python
from streaming_analyze import StreamingAnalyze
from src.all_enum import MODEL

StreamingAnalyze(
    url=r"D:\path\video.mp4",
    mode=MODEL.OFFLINE,
    slice_sec=5,
    enable_b=True,
    enable_c=True,
    ew_guard_enabled=None #关闭对齐与节流
).run_simple(print_vlm=True, print_asr=True)
```

---

## 🧪 事件规范（片段化媒体时间 + 产生时间）

**公共字段：**

| 字段 | 含义 |
|---|---|
| `segment_index` | 片段序号（A 切片产生） |
| `t0` / `t1` | 片段媒体时间（秒），来自 A |
| `_meta.emit_ts` / `_meta.emit_iso` | 事件产生的时钟时间（用于 UI 排序等） |
| `model` | 后端模型名（B/C） |

**VLM（B）事件：**

- 增量：  
  `{"type":"vlm_stream_delta","delta":"...","seq":1,"media_kind":"video|images","t0":..,"t1":..,"_meta":{...}}`
- 收尾：  
  `{"type":"vlm_stream_done","full_text":"...","latency_ms":1234,"media_used":[...],"origin_policy":"...","t0":..,"t1":..,"_meta":{...}}`

**ASR（C）事件（默认句级，仅收尾）：**
- 句级收尾：  
  `{"type":"asr_stream_done","full_text":"...","latency_ms":1234,"sentence_times":[{"start_ts":..,"end_ts":..}, ...],"t0":..,"t1":..,"_meta":{...}}`

> 需要**字级增量**？在 `worker_c_asr.py` 中切换策略（示例已留注释，默认是句级）。

 **历史摘要回带（B 线程）**  
  - 从第 2 段起，把上一轮（最多 N 轮，默认 30）VLM 的“段级摘要”拼接进提示词，让模型在理解上下文的基础上工作。
   - 推荐的提示词片段（可按业务微调）：
     ```
     你正在按时间顺序总结视频/关键帧。以下为历史摘要（可能为空）：
     ---
     ${HISTORY}
     ---
     规则：
     - 仅输出 **相较历史的新增信息**；若没有新增，输出“无新增”或留空。
     - 使用要点化/项目符号；避免重复历史描述；不要复述相同场景不变的元素。
     - 如果出现新的动作、文字、水印、数值、场景或人物变化，请明确指出。
     ```

---

## ⏱️ 对齐与节流（Skew Guard）

- `TranscriptPlaybackSkewController` 控制不同通道（VLM/ASR）向外发射的**速率**与**相对领先**，避免 UI 抖动/过载  
- 核心环境变量：
  - `EMIT_MAX_SKEW_S`：VLM 可领先播放进度的最大秒数  
  - `EMIT_RATE_LIMIT_HZ`：每通道事件外发的上限频率（Hz）

---

## ⚙️ 性能 & 参数建议

- `slice_sec` 过小：上下文短、VLM/ASR 开销增加；过大：时延变高。一般 **5~10s** 较均衡  
- OFFLINE 策略默认：**显著运动 → 小视频**；**非显著 → 关键帧**（更省流）  
- SECURITY 模式：静场固定抽帧，运动时加密关键帧  
- 设备受限时：降低小视频 `height/fps`，或减少关键帧 `max_frames`

---

## 🛠️ 故障排查

- **没有任何输出？**  
  - 检查 FFmpeg 是否在 PATH  
  - 本地路径/RTSP 是否可读  
  - DashScope SDK 是否安装，`DASHSCOPE_API_KEY` 是否可用
- **B/C 线程提前退出**：主控会快停并打印原因  
- **监控线程失败**：降级无监控模式，不影响主流程

---

## 🔒 停止策略

- **离线**：A 切到 EOF → 主控发 STOP → 等 B/C 自然退出 → 关消费者与监控（慢停）  
- **实时**：A 常驻；`force_stop()` 任意时刻触发快停  
- **异常**：B/C 任一早死 → 主控巡检后广播 STOP（快停）

---

## 🧭 路线图

- [ ] ASR 字级/句级切换 & 两路同时产出（可配置）  
- [ ] 多后端 VLM 适配层  
- [ ] 关键帧抽取策略自动调优（结合静音/运动）

---

## 🤝 贡献

欢迎提交 Issue / PR！  
建议提交前先跑 `run_stream_example.py`，并附上系统/日志片段。
技术交流也可发送到作者邮箱: 13594053100@163.com

---

## 📜 许可

Apache-2.0

---

## 📎 附：最小化演示（可直接复制运行）

```python
# mini_demo.py
from streaming_analyze import StreamingAnalyze
from src.all_enum import MODEL

ctrl = StreamingAnalyze(
    url=r"D:\streaming_analyze_video_audio\static\video\RAG_Video_with_sound_test.mp4",
    mode=MODEL.OFFLINE,
    slice_sec=5,
    enable_b=True,
    enable_c=True,
    ew_guard_enabled=None #关闭对齐与节流
)

for ev in ctrl.run_stream(print_vlm=False, print_asr=False):
    print(ev)  # 每条都带 _meta.emit_iso 与媒体 t0/t1
```

> RTSP 示例（占位）：`rtsp://user:pass@camera-host:554/live/stream`  
> 请替换为你自己的相机地址进行测试。
