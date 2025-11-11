import subprocess
import os
from PIL import Image
from typing import List, Dict, Any, Optional


import asyncio
import shutil
if __name__ == "__main__":
    async def _test():

        video_path = rf"F:\project\agent\jd\hzz\jd_multi\test\采销介绍.mp4"


        # # 示例 2：简单问题回答
        # prompt2 = "用户加入购物车的第一个商品内存版本比相同配色的最小内存版本大了多少？单位GB，直接输出数字"
        # answer = await  video_understanding(video_path, prompt2)
        # # answer = video_understanding(video_path, prompt2, start_time=30, end_time=32)
        # print(answer)
        frames = extract_keyframes_smart_delay(video_path,f".\keyframes",    max_frames=30
)
        # save_frames_to_folder(frames,f".\keyframes",)

    asyncio.run(_test())
