'''
Author: 13594053100@163.com
Date: 2025-10-22 10:22:45
LastEditTime: 2025-10-22 11:12:21
'''

from pydantic import BaseModel, Field

class AsrConfig(BaseModel):
    disfluency_removal_enabled: bool = False # 是否开启过滤语气词
    semantic_punctuation_enabled: bool = True # True开启语义断句, False开启VAD断句; 语义断句准确性更高，适合会议转写场景; VAD断句延迟较低，适合交互场景。
    max_sentence_silence: int = Field(default=800,ge=200,le=6000) # 设置VAD断句的静音时长阈值(单位为ms), 当一段语音后的静音时长超过该阈值时，
                                                                  # 系统会判定该句子已结束。参数范围为200ms至6000ms，默认值为800ms。
    punctuation_prediction_enabled: bool = True # 是否在识别结果中自动添加标点
    inverse_text_normalization_enabled: bool = True # 设置是否开启逆文本正则化(INT), 开启后，中文数字将转换为阿拉伯数字。
