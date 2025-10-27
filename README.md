# 🚀 StreamingAnalyze — 智能流式音视频分析及RTSP安防系统

> 端到端多线程流式音视频理解系统   
> 支持 **离线文件音频转录和视觉理解** 与 **RTSP 实时安防巡检**；内置 **FastAPI 后端**、**原生前端页面**、**RTSP内置提示词**。

---

## 目录（ToC）

- [项目简介](#项目简介)
- [架构与数据流](#架构与数据流)
- [目录结构](#目录结构)
- [安装与运行](#安装与运行)
- [环境变量](#环境变量)
- [后端接口（FastAPI）](#后端接口fastapi)
- [前端页面说明](#前端页面说明)
- [监控与日志](#监控与日志)
- [License](#license)

---

## 🌟 项目简介

**StreamingAnalyze** 将音视频源切为窗口片段并并行处理：

- **A 线程（`worker_a_cut`）**：基于 FFmpeg 切片并标准化输出  
  - 视频：无声 MP4（直接用于 VLM）  
  - 音频：16kHz 单声道 WAV（供 ASR）  
  - 失败时自动从“流拷贝”回退到“重编码”，提升鲁棒性
- **B 线程（`worker_b_vlm`）**：视觉理解（VLM），支持**流式增量**与**完整段**输出
- **C 线程（`worker_c_asr`）**：语音识别（ASR），支持**增量转写**、**静音跳过**与**完整文本**
- **主控（`StreamingAnalyze`）**：调度 A/B/C、状态监控、通过 **SSE** 实时推送到浏览器三大面板

---

## 🧩 架构与数据流

```
            ┌──────────────────────────────┐
            │        StreamingAnalyze       │
            │   主控：调度 / SSE / 统计 / 对齐  │
            └───────┬───────────┬──────────┘
                    │                           (SSE 事件)
                    │                           ─────────► 前端三面板
           ┌────────┘           └───────────┐
           ▼                                  ▼
   [A] worker_a_cut.py                 [C] worker_c_asr.py
   FFmpeg 切片/标准化                     语音识别（增量/整段）
           │
           ▼
   [B] worker_b_vlm.py
   视觉理解（流式/非流式）
```

事件通过 **SSE** 推送到前端：**语音转录** / **视觉理解** / **RTSP 安防检测** 三个面板分栏显示。

---

## 📦 目录结构

```
.
├─ app_service.py                # FastAPI 入口
├─ index.html                    # 前端页面（原生 HTML + SSE）
├─ streaming_analyze.py          # 主控（A/B/C 管线 + SSE 事件总线）
├─ run_stream_example.py         # 仅后端调用示例
├─ src/
│  ├─ workers/
│  │  ├─ worker_a_cut.py
│  │  ├─ worker_b_vlm.py
│  │  └─ worker_c_asr.py
│  ├─ configs/
│  │  ├─ asr_config.py           # AsrConfig（过滤语气词/语义断句/逆文本正则化等）
│  │  ├─ cut_config.py           # CutConfig（切片/采样策略/阈值等）
│  │  └─ vlm_config.py           # VlmConfig（提示词/流式/事件级别等）
│  ├─ utils/
│  │  ├─ ffmpeg_utils.py 
│  │  ├─ logger_utils.py
│  │  ├─ backpressure.py         # 可选：背压控制
│  │  └─ skew_guard.py           # 可选：VLM/ASR 发射对齐/限速
│  └─ all_enum.py                # RTSP安防内置提示词等枚举
├─ requirements.txt
└─ uploads/                      # 运行期上传/切片输出
```

---

## ⚙️ 安装与运行

### 1) 克隆与安装依赖

```bash
git clone https://github.com/June2124/streaming_analyze_video_audio
cd streaming_analyze_video_audio
pip install -r requirements.txt
```

### 2) 启动服务

```bash
uvicorn app_service:app --host 0.0.0.0 --port 8000
```

（也可在本地将 `--host` 改为 `127.0.0.1`）

---

## 🔑 环境变量

```bash
# DASHSCOPE 模型平台 API Key
export DASHSCOPE_API_KEY="sk-xxxxxxxx"

# （可选）指定自定义 ffmpeg/ffprobe，可不设（默认走系统 PATH）
export FFMPEG_BIN="$(which ffmpeg)"
export FFPROBE_BIN="$(which ffprobe)"
```

> Windows（PowerShell）请使用：
>
> ```powershell
> $env:DASHSCOPE_API_KEY="sk-xxxxxxxx"
> $env:FFMPEG_BIN="C:\path\to\ffmpeg.exe"
> $env:FFPROBE_BIN="C:\path\to\ffprobe.exe"
> ```

---

## 🔌 后端接口（FastAPI）

| 路径        | 方法   | 说明                                   |
|-------------|--------|----------------------------------------|
| `/upload`   | POST   | 上传离线文件并启动分析（音/视频均可）     |
| `/rtsp`     | POST   | 启动 RTSP 实时分析                       |
| `/stop`     | POST   | 停止当前任务                             |
| `/events`   | GET    | SSE 事件流（前端使用 EventSource 订阅）  |

---

## 🖥️ 前端页面说明

- **左栏**：上传离线文件或启动 RTSP  
- **右上**：视频播放器（仅离线文件可本地预览）  
- **右下三栏**：  
  - **语音转录（ASR）**：增量 + 完整文本  
  - **视觉理解（VLM）**：打字机效果渲染；RTSP 安防为结构化事件卡片  
  - **RTSP 安防检测**：事件等级 / 置信度 + 证据帧缩略图  

页面使用原生 `EventSource('/events?...')` 订阅 **SSE**，无需前端框架。

---

## 🧭 监控与日志

- 周期打印：`[监控] 队列水位 VIDEO=… AUDIO=… VLM=… ASR=… | 线程存活={…}`  
- **A 线程**：每个片段的视频/音频产物路径  
- **B/C 线程**：增量/完成事件与统计汇总（平均/最大延迟、字符数、段数等）

---

## 📜 License

Apache-2.0  
© 2025 June2124
