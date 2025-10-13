'''
Author: 13594053100@163.com
Date: 2025-09-30 09:43:44
LastEditTime: 2025-09-30 09:43:53
'''

import asyncio
import random
import time
from src.utils.logger_utils import get_logger

logger = get_logger(__name__)

class InternetUtils:

    @staticmethod
    async def async_backoff_sleep(attempt: int = 4,
                               base: float = 2.0,
                               factor: float = 1.7,
                               max_sec: float = 20.0,
                               jitter: float = 0.5) -> None:
        """异步指数退避等待, 可选抖动
        """
        if not(attempt >=0 and base >0 and factor >0 and max_sec >0 and jitter >=0):
            raise ValueError("调用异步指数退避等待参数错误")
        
        sleep = base * (factor ** attempt)
        if jitter:
            sleep += random.uniform(0, jitter)
        sleep = min(sleep, max_sec)

        logger.info(f"异步指数退避等待 {sleep:.2f} 秒")
        await asyncio.sleep(max(0.0, sleep))
    
    @staticmethod
    def backoff_sleep(attempt: int = 4,
                      base: float = 2.0,
                      factor: float = 1.7,
                      max_sec: float = 20.0,
                      jitter: float = 0.5) -> None:
        """同步指数退避等待, 可选抖动
        """
        if not(attempt >=0 and base >0 and factor >0 and max_sec >0 and jitter >=0):
            raise ValueError("调用同步指数退避等待参数错误")
        
        sleep = base * (factor ** attempt)
        if jitter:
            sleep += random.uniform(0, jitter)
        sleep = min(sleep, max_sec)
        
        logger.info(f"同步指数退避等待 {sleep:.2f} 秒")
        time.sleep(max(0.0, sleep))
    
    