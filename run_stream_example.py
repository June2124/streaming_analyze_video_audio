'''
Author: 13594053100@163.com
Date: 2025-10-13 18:55:39
LastEditTime: 2025-10-22 18:23:26
'''

from streaming_analyze import StreamingAnalyze
from src.all_enum import MODEL,VLM_SYSTEM_PROMPT_PRESET,DIFF_METHOD,CLOUD_VLM_MODEL_NAME
from src.configs.vlm_config import VlmConfig
from src.configs.asr_config import AsrConfig
from src.configs.cut_config import CutConfig

"""
1.如果能够确定url的资源类型, 建议: 音频文件: enable_b=False enable_c=True;  无声视频文件: enable_b=True enable_c=False; 音视频文件: enable_b=True enable_c=True
实时RTSP: 根据具体需求和流音视频轨道情况自行选择。
2.如果无法确定url的资源类型, 统一将enable_b=True enable_c=True 代价是可能会多开1个子线程
"""

# vlm_system_prompt = "如果发现有人带了安全帽，告诉我戴帽时间"
# vlm_system_prompt = "这个视频是白天还是夜晚拍摄的，回答时间信息，其他的不要回答"
# vlm_system_prompt = "请描述画面内容，突出重点"

#离线音视频: VLM一般为画面描述型任务, 可以不传入系统提示词, 也可以自定义系统提示词. 但是当vlm_system_prompt传入你自定义的字符串提示词时, 
# 不保证VLM模型一定能返回JSON数据格式, 很有可能只是无规则文本。

# 注意: 当你开启了流式输出vlm_streaming=True，此时调用run_stream(), 会推送vlm_stream_delta和vlm_stream_done, 正常情况下我们只消费vlm_stream_delta, vlm_stream_done只是对当前轮所有vlm_stream_delta的总结。
#       当你关闭流式输出vlm_streaming=False, 此时调用run_stream(), 只会推送vlm_stream_done, 正常消费即可。
#       C侧的ASR是句级转录结果，永远只会推ASR_stream_done, 正常消费即可

# ctrl = StreamingAnalyze(
#     url=r"D:\JetLinksAI_Analyze_Video_Audio\static\video_audio\RAG_Video_with_sound_test.mp4",
#     mode=MODEL.OFFLINE,
#     slice_sec=5,
#     enable_b=True,  
#     enable_c=True,
#     skew_guard_enabled=None, #关闭对齐与节流
#     cut_config=CutConfig(interval_sec=1.0,diff_method=DIFF_METHOD.BGR_RATIO,diff_threshold=0.50),
#     vlm_config=VlmConfig(vlm_system_prompt=vlm_system_prompt,vlm_model_name=CLOUD_VLM_MODEL_NAME.QWEN_VL_MAX,vlm_streaming=True),
#     asr_config=AsrConfig(disfluency_removal_enabled=False,semantic_punctuation_enabled=False,max_sentence_silence=1000
#                          ,punctuation_prediction_enabled=True,inverse_text_normalization_enabled=False)
# )

# -------------------------------------------
# 实时 RTSP: VLM一般为任务类: 1.此时采用系统内置提示词枚举类VLM_SYSTEM_PROMPT_PRESET里的枚举提示词对象, 一定保证统一返回严格JSON格式数据, 键定义请看VLM_SYSTEM_PROMPT_PRESET源码。
#                           2. 除了会议转录等场景(本质是画面描述型VLM任务), 工厂安防/小区安防/员工效率检测等各自检测场景建议关闭音频侧, 除非你有特殊需求。
ctrl = StreamingAnalyze(
    url="rtsp://admin:p@ssw0rd@192.168.33.152:554/Streaming/Channels/101",  
    mode=MODEL.SECURITY,   # 实时流建议 ONLINE 或 SECURITY
    slice_sec=4,
    enable_b=True,       
    enable_c=False,        
    skew_guard_enabled=False, #关闭对齐与节流
    cut_config=CutConfig(interval_sec=2.0,diff_method=DIFF_METHOD.GRAY_MEAN,diff_threshold=0.55),
    # 当你选择系统内置提示词进行安防等任务时, 系统会禁止你采用流式输出，run_stream()只会推送vlm_stream_done, 它的full_text键对应严格JSON字符串, 请对应解析。
    vlm_config= VlmConfig(vlm_system_prompt=VLM_SYSTEM_PROMPT_PRESET.OFFICE_PRODUCTIVITY_COMPLIANT,vlm_streaming=False,vlm_model_name=CLOUD_VLM_MODEL_NAME.QWEN3_VL_PLUS),
)

# 边跑边拿事件：VLM(增量+收尾)；ASR(仅收尾 full_text)
for ev in ctrl.run_stream(print_vlm=False, print_asr=False):
    print(ev)  # 关键字段：ev["type"], ev["segment_index"], ev.get("delta")/ev.get("full_text"), ev["_meta"]["emit_iso"]
    # 可按条件手动停止直播：
    # if need_stop: ctrl.force_stop("manual stop")

