'''
Author: 13594053100@163.com
Date: 2025-10-08 08:41:48
LastEditTime: 2025-10-13 16:21:06
'''
'''
Author: 13594053100@163.com
Date: 2025-10-08 08:41:48
LastEditTime: 2025-10-08 11:06:28
'''
from enum import Enum

class MODEL(Enum):
    OFFLINE = "offline"
    ONLINE = "online"
    SECURITY = "security"

class SOURCE_KIND(Enum):
    # 离线本地文件
    AUDIO_FILE = "audio_file"
    VIDEO_FILE = "video_file"
    # 实时流
    RTSP = "rtsp"
    

