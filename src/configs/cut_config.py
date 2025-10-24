'''
Author: 13594053100@163.com
Date: 2025-10-22 14:26:38
LastEditTime: 2025-10-22 16:31:15
'''

from pydantic import BaseModel, Field, model_validator
from typing import Optional
from src.all_enum import DIFF_METHOD

class CutConfig(BaseModel):
    # 关键帧候选采样间隔(s)，作用于所有帧间变化分数计算方法
    interval_sec: float = Field(default=1.0, ge=1.0, le=60.0)

    # 帧间变化分数计算方法: GRAY_MEAN | BGR_RATIO | HIST | FLOW
    diff_method: DIFF_METHOD = DIFF_METHOD.BGR_RATIO

    # 不同的 diff_method 通常对应不同的分数；允许 None，由校验器按方法补默认
    diff_threshold: Optional[float] = None

    # 直方图柱数
    hist_bins: int = Field(default=32, ge=1, le=256)

    # 光流比较的候选帧步长
    flow_step: int = Field(default=1, ge=1, le=5)

    # 运动检测的采样间隔
    motion_sample_interval: float = Field(default=1.0, ge=1.0, le=5.0)

    # 运动检测阈值
    motion_diff_threshold: float = Field(default=15.0, ge=1.0, le=200.0)

    # worker_a_cut() 输出的根目录
    out_dir: str = "out"

    @model_validator(mode="after")
    def _fill_defaults_and_check(self):
        # 1) 按方法补默认阈值
        if self.diff_threshold is None:
            if self.diff_method in (DIFF_METHOD.GRAY_MEAN, DIFF_METHOD.BGR_RATIO):
                self.diff_threshold = 0.65
            elif self.diff_method == DIFF_METHOD.HIST:
                self.diff_threshold = 0.22
            else:  # DIFF_METHOD.FLOW
                self.diff_threshold = 1.5

        # 2) 按方法做区间校验
        if self.diff_method in (DIFF_METHOD.GRAY_MEAN, DIFF_METHOD.BGR_RATIO):
            if not (0.0 <= self.diff_threshold <= 1.0):
                raise ValueError("GRAY_MEAN 或 BGR_RATIO 的分数值域为 [0,1]，请设置正确的 diff_threshold")
        elif self.diff_method == DIFF_METHOD.HIST:
            if not (0.0 <= self.diff_threshold <= 2.0):
                raise ValueError("HIST 的分数值域为 [0,2]，请设置正确的 diff_threshold")
        elif self.diff_method == DIFF_METHOD.FLOW:
            if self.diff_threshold < 0.0:
                raise ValueError("FLOW 的分数值域为 [0,+∞)，请设置正确的 diff_threshold")

        return self
