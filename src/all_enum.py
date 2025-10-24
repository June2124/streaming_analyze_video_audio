'''
Author: 13594053100@163.com
Date: 2025-10-08 08:41:48
LastEditTime: 2025-10-23 16:41:10
'''

from enum import Enum

class MODEL(Enum):
    # 不同的模式对应不同的切窗大小等策略
    OFFLINE = "offline"
    ONLINE = "online"
    SECURITY = "security"

class SOURCE_KIND(Enum):
    # 离线本地文件
    AUDIO_FILE = "audio_file"
    VIDEO_FILE = "video_file"
    # 实时流
    RTSP = "rtsp"

class VLM_SYSTEM_PROMPT_PRESET(Enum):
    """
    VLM 系统级提示词文本，用于任务类场景。
    统一输出规范：
      - 严格输出 JSON 数组；每个对象恰含 5 个键：
        type, describe, level, suggestion, confidence
      - level ∈ {LOW, MEDIUM, HIGH, CRITICAL}
      - confidence ∈ [0.00, 1.00]（数字，保留两位小数）
      - describe/suggestion 为中文≤15字
      - 无事件则输出 []
      - 只输出 JSON, 不要任何额外文本/注释/换行说明
    
    单事件JSON对象示例:
    {
    "type":"事件标签",
    "describe":"15字内描述",
    "level":HIGH,
    "suggestion":15字内建议,
    "confidence":0.87
    }
    """

    # 1) 工厂安防
    FACTORY_SECURITY = (
        "角色：工业园区安防巡检。识别：入侵、翻越围栏、闯禁区、徘徊、斗殴、烟雾/明火、资产破坏、化学泄漏迹象、电箱敞开、夜间可疑车辆/人员等。"
        "只输出 JSON 数组，元素字段固定为(type, describe, level, suggestion, confidence)。"
        "level 取 {LOW, MEDIUM, HIGH, CRITICAL}；confidence 为 0~1 数字，保留两位小数；describe/suggestion 中文≤15字。"
        "无事件输出[]。按风险由高到低排序。避免身份推断。"
        "可用 type ∈ {INTRUSION, RESTRICTED_AREA, LOITERING, FIGHTING, SMOKE_FIRE, PROPERTY_DAMAGE, HAZMAT_LEAK_SUSPECT, ELECTRICAL_PANEL_OPEN, UNKNOWN}。"
    )

    # 2) 小区安防
    RESIDENTIAL_SECURITY = (
        "角色：住宅小区安防助手。识别：翻越围墙、尾随进门、可疑徘徊、深夜聚集喧哗、乱扔烟头引发火险、车辆逆行/占消防通道、非法摆摊等。"
        "仅输出 JSON 数组；对象字段(type, describe, level, suggestion, confidence)；describe/suggestion≤15字；confidence 两位小数；无事件[]。"
        "禁止身份/隐私推断。按风险从高到低排序。"
        "可用 type ∈ {FENCE_CLIMB, TAILGATING, LOITERING, NOISE, SMOKE_FIRE, TRAFFIC_VIOLATION, EVAC_ROUTE_BLOCK, ILLEGAL_STALL}。"
    )

    # 3) 办公区秩序与占用（合规）
    OFFICE_PRODUCTIVITY_COMPLIANT = (
        "角色：办公区秩序与占用监测（不做身份/隐私推断）。检测：长时离岗工位、会议室超时、公共区拥堵、随地吸烟/明火、电气隐患、睡觉等。"
        "只输出 JSON 数组，元素含(type, describe, level, suggestion, confidence)；describe/suggestion≤15字；confidence 两位小数；无事件[]。"
        "type ∈ {SEAT_IDLE, ROOM_OVERTIME, CROWDING, SMOKE_FIRE, ELECTRICAL_RISK}。"
    )

    # 4) 施工现场 PPE 与三违
    CONSTRUCTION_PPE = (
        "角色：施工现场安全巡检。检测：安全帽/反光背心/防护鞋/护目镜/安全带佩戴，识别三违，高处未系绳、临边洞口无防护、脚手架超载、物料凌乱、焊接明火等。"
        "仅输出 JSON 数组；字段(type, describe, level, suggestion, confidence)；describe/suggestion≤15字；confidence 两位小数；无事件[]。"
        "type ∈ {PPE_MISS, UNSAFE_OPERATION, EDGE_UNPROTECTED, OVERLOAD, HOUSEKEEPING_ISSUE, SMOKE_FIRE}。"
    )

    # 5) 零售门店 防损/陈列合规
    RETAIL_LOSS_PREVENTION = (
        "角色：零售门店巡检。防损：高价值区长留、撬防盗扣、可疑藏匿；陈列：价签缺失/错位、堆头倾倒风险、通道被占、冷柜门长开等。"
        "仅输出 JSON 数组；固定字段(type, describe, level, suggestion, confidence)；describe/suggestion≤15字；confidence 两位小数；无事件[]。"
        "避免动机/身份推断。type ∈ {LOITERING_HIGH_VALUE, ANTI_THEFT_TAMPER, HIDING_BEHAVIOR, PRICE_TAG_MISS, OBSTRUCTION, DISPLAY_COLLAPSE_RISK, DOOR_OPEN_TOO_LONG}。"
    )

    # 6) 交通路口 态势与风险
    TRAFFIC_INTERSECTION = (
        "角色：路口态势观察。识别：闯红灯、逆行、占压斑马线、机非混行、拥堵、事故/擦碰、积水、摔倒风险等。"
        "仅输出 JSON 数组；字段(type, describe, level, suggestion, confidence)；describe/suggestion≤15字；confidence 两位小数；无事件[]。"
        "type ∈ {REDLIGHT_JUMP, WRONG_WAY, CROSSWALK_BLOCK, MIXED_TRAFFIC, CONGESTION, ACCIDENT, WATERLOGGING, FALL_RISK}。"
    )

    # 7) 仓储作业 安全与效率
    WAREHOUSE_SAFETY = (
        "角色：仓库/物流中心风控。检测：叉车高速/急转、人与车混行、货架超高/超载、托盘破损、消防通道被占、烟雾/明火；效率：码放无序、装卸久候、关键位空置。"
        "仅输出 JSON 数组；字段(type, describe, level, suggestion, confidence)；describe/suggestion≤15字；confidence 两位小数；无事件[]。"
        "type ∈ {FORKLIFT_RISK, MIXED_TRAFFIC, OVERLOAD, RACK_RISK, FIRE_AISLE_BLOCK, SMOKE_FIRE, INEFFICIENT_FLOW}。"
    )

    # 8) 实验室/机房 安全合规
    LAB_DATACENTER_SAFETY = (
        "角色：实验室/机房安全观察。关注：烟雾/明火、水渍近供电、线缆过热/打火、机柜未关/未上锁、化学品/气瓶摆放不合规等。"
        "仅输出 JSON 数组；字段(type, describe, level, suggestion, confidence)；describe/suggestion≤15字；confidence 两位小数；无事件[]。避免个人信息推断。"
        "type ∈ {SMOKE_FIRE, WATER_NEAR_POWER, CABLE_OVERHEAT, CABINET_OPEN, CHEM_STORAGE_ISSUE}。"
    )

    # 9) 生产线 外观质检/工位异常（安防风格）
    MANUFACTURING_QA = (
        "角色：制造产线外观质检与工位异常检测。检测：划痕、裂纹、脏污、缺件、装配错位、贴标偏移；异常：堆积、节拍紊乱、传送带卡滞、工装缺失等。"
        "仅输出 JSON 数组；字段(type, describe, level, suggestion, confidence)；describe/suggestion≤15字；confidence 两位小数；无事件[]。"
        "type ∈ {SCRATCH, CRACK, STAIN, MISSING_PART, MISALIGN, LABEL_OFFSET, STATION_BLOCK, JAM}。"
    )

    # 10) 医疗/养老机构 公共区域安全
    HEALTHCARE_PUBLIC_SAFETY = (
        "角色：医院/养老机构公共区域安全（不做身份/隐私推断）。关注：跌倒风险、地面湿滑、吸烟/明火、疏散通道被占、设备报警/泄漏等。"
        "仅输出 JSON 数组；字段(type, describe, level, suggestion, confidence)；describe/suggestion≤15字；confidence 两位小数；无事件[]。"
        "type ∈ {FALL_RISK, SLIPPERY_FLOOR, SMOKE_FIRE, EVAC_ROUTE_BLOCK, EQUIP_ALARM}。"
    )
    

class CLOUD_VLM_MODEL_NAME(Enum):
    # 当前代码基于阿里qwen系列
    QWEN3_VL_PLUS = "qwen3-vl-plus"
    QWEN_VL_MAX = "qwen-vl-max"
    QWEN_VL_PLUS = "qwen-vl-plus"
    

class VLM_SYSTEM_PROMPT_AUTO_ADD(Enum):
    """
    用于B线程自动拼接到VLM系统提示词的模板。
    分为两类：
      - DESCRIPTIVE：描述型（上游传入字符串 → 自由任务说明）
      - TASK_ORIENTED：任务型（VLM_SYSTEM_PROMPT_PRESET 场景）
    """

    # 描述型（上游字符串提示词场景）
    DESCRIPTIVE = (
        "仅分析当前传入的新增视觉片段，不要复述或总结历史内容。"
        "历史语义小结将由 user 提示词提供。"
        "你只需对新增内容执行语义分析、状态理解或事件识别。"
        "请避免重复描述先前画面。"
    )

    # 任务型（系统级任务预设场景）
    TASK_ORIENTED = (
        "仅分析当前传入的新增视觉片段，不要重复历史画面或总结。"
        "历史总结由 user 提示词提供。"
        "请仅针对新增片段执行事件识别、对象状态与风险判断，"
        "并严格按规范输出结构化 JSON 数组。"
        "输出字段固定为(type, describe, level, suggestion, confidence)，"
        "不得添加额外文本、解释或格式说明。"
    )

class DIFF_METHOD(Enum):
    # 帧间变化分数计算方法
    GRAY_MEAN = "gray_mean" # 灰度绝对差的平均值
    BGR_RATIO = "bgr_ratio" # BGR三通道绝对差的非零比例
    HIST = "hist" # 灰度直方图相关系数的(1-corr)
    FLOW = "flow" # Farneback光流幅值均值

class VLM_DETECT_EVENT_LEVEL(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"