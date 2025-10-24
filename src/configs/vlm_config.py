'''
Author: 13594053100@163.com
Date: 2025-10-21 15:54:47
LastEditTime: 2025-10-23 19:38:39
'''
from typing import Optional, Union
from pydantic import BaseModel, field_validator, model_validator
from src.all_enum import VLM_SYSTEM_PROMPT_PRESET, CLOUD_VLM_MODEL_NAME, VLM_DETECT_EVENT_LEVEL


class VlmConfig(BaseModel):
    vlm_system_prompt: Optional[Union[VLM_SYSTEM_PROMPT_PRESET,str]] = ""
    vlm_model_name: CLOUD_VLM_MODEL_NAME = CLOUD_VLM_MODEL_NAME.QWEN3_VL_PLUS
    vlm_streaming: bool = True
    vlm_task_history_enabled: bool = False # VLM是否在RTSP任务型中拼接历史回复信息
    vlm_event_min_level: VLM_DETECT_EVENT_LEVEL = VLM_DETECT_EVENT_LEVEL.LOW # RTSP任务型VLM检测的事件挑选最低等级
    vlm_static_keyframe_dir: Optional[str] = None# 本地目录, 控制挑选出的关键帧存放的本地路径
    vlm_static_keyframe_url_prefix: Optional[str] = "/static/keyframes"       # 前端访问的关键帧URL前缀

    @field_validator("vlm_system_prompt")
    @classmethod
    def check_length(cls, v):
        if isinstance(v, str) and len(v) > 300:
            raise ValueError("系统提示词需少于300字")
        return v
    
    
    @model_validator(mode="after")
    def check_consistency(self):
        if isinstance(self.vlm_system_prompt,VLM_SYSTEM_PROMPT_PRESET) and self.vlm_streaming:
            raise ValueError('内置提示词用于任务类场景, 返回统一JSON格式字符串, 该场景下不允许使用流式输出')
        return self