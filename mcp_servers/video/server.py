import cv2
import os
import base64
import time
from typing import List, Optional, Dict, Any, Tuple
from openai import OpenAI
import numpy as np
import os
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
import asyncio
import logging, time
log = logging.getLogger(__name__)

mcp = FastMCP()
load_dotenv("./.env")



'''
需要环境
uv pip install opencv-python
uv pip install scikit-image
uv pip install pypinyin
'''

# 假设你已经配置好 OpenAI API key
api_key = os.getenv("VL_KEY")
base_url = os.getenv("VL_URL")
model =os.getenv("VL_MODEL")
print(api_key)
client = OpenAI(api_key=api_key, base_url=base_url)
SSIM_TH = 0.9

from pydantic import Field

import base64
import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from skimage.metrics import structural_similarity as ssim
import os
def _compute_ssim(img1_bgr, img2_bgr) -> float:
    gray1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY)
    return ssim(gray1, gray2)

from pathlib import Path

# ---------------- 可视化 / 保存同之前 ----------------
def visualize_keyframes(keyframes: List[Dict[str, Any]], win_name: str = "Keyframes"):
    print(f"共保留 {len(keyframes)} 帧，按任意键下一张，q 退出")
    for idx, kf in enumerate(keyframes):
        buf = np.frombuffer(base64.b64decode(kf["image_b64"]), np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        info = f"FRAME {idx}  TS={kf['timestamp']:.2f}s"
        cv2.putText(img, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow(win_name, img)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

import shutil, os

def save_keyframe_images_simple(
    keyframes: List[Dict[str, Any]],
    output_dir: str,
    image_format: str = "jpg",
    filename_prefix: str = "frame"
) -> None:
    """
    简化版：先清空输出目录，用_extract_frames中的时间戳命名保存关键帧图片
    """
    import os
    import shutil
    import cv2
    import numpy as np
    
    # 1. 如果目录已存在，整个删掉
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    # 2. 重新创建空目录
    os.makedirs(output_dir, exist_ok=False)

    saved_count = 0
    for i, frame_info in enumerate(keyframes):
        try:
            import base64
            image_data = base64.b64decode(frame_info["image_b64"])
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is not None:
                # 使用_extract_frames中记录的timestamp
                timestamp = frame_info["timestamp"]
                
                # 将秒转换为时分秒格式
                hours = int(timestamp // 3600)
                minutes = int((timestamp % 3600) // 60)
                seconds = int(timestamp % 60)
                milliseconds = int((timestamp - int(timestamp)) * 1000)
                
                # 创建文件名
                if hours > 0:
                    # 如果视频超过1小时，包含小时信息
                    filename = f"{filename_prefix}_{hours:02d}h{minutes:02d}m{seconds:02d}s{milliseconds:03d}ms.{image_format}"
                else:
                    # 如果视频在1小时内，只显示分钟和秒
                    filename = f"{filename_prefix}_{minutes:02d}m{seconds:02d}s{milliseconds:03d}ms.{image_format}"
                
                filepath = os.path.join(output_dir, filename)

                print(filepath)
                success = cv2.imwrite(str(filepath), img)
                if success:
                    saved_count += 1
                    # print(f"保存: {filename} (时间戳: {timestamp:.3f}s)")
                else:
                    print(f"保存失败: {filename}")
            else:
                print(f"图像解码失败: 第{i}帧")

        except Exception as e:
            print(f"处理第{i}帧时出错: {str(e)}")

    print(f"成功保存 {saved_count} 张图片到 {output_dir}")
# def extract_keyframes_with_ssim(
#     video_path: str,
#     start_time: Optional[float] = None,
#     end_time: Optional[float] = None,
#     fps: float = 1.0,
#     ssim_th: float = SSIM_TH,
#     stable_delay: int = 2  # 新增参数：跳过变化帧后的延迟帧数
# ) -> List[Dict[str, Any]]:
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         raise ValueError("无法打开视频文件")

#     total_fps = cap.get(cv2.CAP_PROP_FPS)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     duration = total_frames / total_fps

#     start = start_time if isinstance(start_time, (int, float)) else 0
#     end = end_time if isinstance(start_time, (int, float)) else duration
#     frame_interval = max(1, int(total_fps / fps))
#     keyframes: List[Dict[str, Any]] = []
#     last_frame = None
#     change_detected = False
#     stable_counter = 0
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
            
#         frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
#         timestamp = frame_idx / total_fps
        
#         if timestamp < start:
#             continue
#         if timestamp > end:
#             break

#         if frame_idx % frame_interval == 0:
#             # 第一帧直接保留
#             if last_frame is None:
#                 _, buffer = cv2.imencode(".jpg", frame)
#                 image_b64 = base64.b64encode(buffer).decode("utf-8")
#                 keyframes.append({
#                     "timestamp": timestamp,
#                     "image_b64": image_b64,
#                     "ssim": None
#                 })
#                 last_frame = frame
#             else:
#                 sim = _compute_ssim(last_frame, frame)
                
#                 if sim < ssim_th:
#                     # 检测到变化，设置标志但不立即保存
#                     change_detected = True
#                     stable_counter = 0
#                     last_frame = frame  # 更新参考帧
#                 else:
#                     # 没有变化或变化很小
#                     if change_detected:
#                         # 在变化后等待一段时间再保存
#                         stable_counter += 1
#                         if stable_counter >= stable_delay:
#                             # 保存稳定的中间帧
#                             _, buffer = cv2.imencode(".jpg", frame)
#                             image_b64 = base64.b64encode(buffer).decode("utf-8")
#                             keyframes.append({
#                                 "timestamp": timestamp,
#                                 "image_b64": image_b64,
#                                 "ssim": sim
#                             })
#                             last_frame = frame
#                             change_detected = False
#                             stable_counter = 0
#                     else:
#                         # 正常情况，没有检测到变化，直接保存
#                         _, buffer = cv2.imencode(".jpg", frame)
#                         image_b64 = base64.b64encode(buffer).decode("utf-8")
#                         keyframes.append({
#                             "timestamp": timestamp,
#                             "image_b64": image_b64,
#                             "ssim": sim
#                         })
#                         last_frame = frame
#     print(len(keyframes))

#     cap.release()
#     return keyframes


def calculate_ssim(frame1, frame2):
        """计算两帧之间的SSIM相似度"""
        # 调整帧大小为统一尺寸以提高计算效率
        size = (256, 256)
        frame1_resized = cv2.resize(frame1, size)
        frame2_resized = cv2.resize(frame2, size)
        
        # 转换为灰度图
        gray1 = cv2.cvtColor(frame1_resized, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2_resized, cv2.COLOR_BGR2GRAY)
        
        # 计算SSIM
        score, _ = ssim(gray1, gray2, full=True)
        return score


####111
# def extract_keyframes_with_ssim(
#     video_path: str,
#     start_time: Optional[float] = None,
#     end_time: Optional[float] = None,
#     file_name: str= None,
#     fps: float = 2.0,
#     ssim_th: float = SSIM_TH,
#     stable_delay: int = 4,
#     target_frame_count: Tuple[int, int] = (12, 25),  # 目标帧数范围
#     max_iterations: int = 10  # 最大迭代次数，防止无限循环
# ) -> List[Dict[str, Any]]:
        
#     def _extract_frames():
#         """提取视频帧（不带阈值筛选）"""
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             raise ValueError("无法打开视频文件")

#         total_fps = cap.get(cv2.CAP_PROP_FPS)
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         duration = total_frames / total_fps

#         start = start_time if isinstance(start_time, (int, float)) else 0
#         end = end_time if isinstance(end_time, (int, float)) else duration
#         frame_interval = max(1, int(total_fps / fps))
        
#         frames: List[Dict[str, Any]] = []
        
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
                
#             frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
#             timestamp = frame_idx / total_fps
            
#             if timestamp < start:
#                 continue
#             if timestamp > end:
#                 break

#             if frame_idx % frame_interval == 0:
#                 _, buffer = cv2.imencode(".jpg", frame)
#                 image_b64 = base64.b64encode(buffer).decode("utf-8")
#                 frames.append({
#                     "timestamp": timestamp,
#                     "image_b64": image_b64,
#                     "frame": frame  # 保留原始帧数据用于后续处理
#                 })
        
#         cap.release()
#         return frames
#     def filter_frames_by_ssim(frames, threshold):
#         """根据SSIM阈值筛选关键帧，保留场景变化后的稳定帧"""
#         if not frames:
#             return []
        
#         keyframes = [frames[0]]  # 总是保留第一帧
        
#         for i in range(1, len(frames)):
#             prev_frame = keyframes[-1]["frame"]
#             curr_frame = frames[i]["frame"]
            
#             # 计算与上一关键帧的相似度
#             similarity = calculate_ssim(prev_frame, curr_frame)
            
#             # 如果相似度低于阈值，说明场景变化，保留变化后的帧（考虑稳定延迟）
#             if similarity < threshold:
#                 # 确保不会超出frames范围
#                 stable_index = min(i + stable_delay, len(frames) - 1)
#                 stable_frame = frames[stable_index]
                
#                 # 检查是否已经包含了这个稳定帧
#                 if stable_frame not in keyframes:
#                     keyframes.append(stable_frame)
        
#         return keyframes
#     min_target, max_target = target_frame_count
#     current_threshold = ssim_th
#     iteration = 0
#     frames = _extract_frames()
#     save_keyframe_images_simple(frames,output_dir=f".\keyframes\{file_name}_all")
#     print(f"帧数量: {len(frames)}")

#     best_keyframes = []
#     while iteration < max_iterations:
#         print(f"\n迭代 {iteration + 1}, 使用SSIM阈值: {current_threshold:.3f}")
#         keyframes = filter_frames_by_ssim(frames, current_threshold)
#         keyframe_count = len(keyframes)
        
#         print(f"筛选后关键帧数量: {keyframe_count}")

        
#         # 检查是否在目标范围内
#         if min_target <= keyframe_count <= max_target:
#             print(f"达到目标帧数范围: {keyframe_count}")
#             best_keyframes = keyframes
#             break
#         elif keyframe_count < min_target:
#             # 帧数太少，降低阈值（更宽松）
#             current_threshold *= 1.1
#             print(f"帧数过少 ({keyframe_count} < {min_target})，降低阈值至: {current_threshold:.3f}")
#         else:
#             # 帧数太多，提高阈值（更严格）
#             current_threshold *= 0.8 
#             print(f"帧数过多 ({keyframe_count} > {max_target})，提高阈值至: {current_threshold:.3f}")
        
#         # 防止阈值超出合理范围
#         current_threshold = max(0.1, min(0.95, current_threshold))
#         iteration += 1
    
#     # 如果迭代结束仍未达到目标，使用最后一次的结果
#     if not best_keyframes and keyframes:
#         print(f"达到最大迭代次数，使用当前结果: {len(keyframes)} 帧")
#         best_keyframes = keyframes
    
#     # 保存最终关键帧
#     if best_keyframes:
#         save_keyframe_images_simple(best_keyframes, output_dir=f"./keyframes/{file_name}")
    

    
#     return best_keyframes

####111
import os
import cv2
import base64
import re
from typing import List, Dict, Any, Optional, Tuple

def extract_keyframes_with_ssim(
    video_path: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    file_name: str = None,
    fps: float = 2.0,
    ssim_th: float = SSIM_TH,
    stable_delay: int = 2,
    target_frame_count: Tuple[int, int] = (15, 25),
    max_iterations: int = 25
) -> List[Dict[str, Any]]:
    
    def _extract_frames():
        """提取视频帧（不带阈值筛选）"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("无法打开视频文件")

        total_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / total_fps

        start = start_time if isinstance(start_time, (int, float)) else 0
        end = end_time if isinstance(end_time, (int, float)) else duration
        frame_interval = max(1, int(total_fps / fps))
        
        frames: List[Dict[str, Any]] = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            timestamp = frame_idx / total_fps
            
            if timestamp < start:
                continue
            if timestamp > end:
                break

            if frame_idx % frame_interval == 0:
                _, buffer = cv2.imencode(".jpg", frame)
                image_b64 = base64.b64encode(buffer).decode("utf-8")
                frames.append({
                    "timestamp": timestamp,
                    "image_b64": image_b64,
                    "frame": frame  # 保留原始帧数据用于后续处理
                })
        
        cap.release()
        return frames

    def filter_frames_by_ssim(frames, threshold):
        """根据SSIM阈值筛选关键帧，保留场景变化后的稳定帧"""
        if not frames:
            return []
        
        keyframes = [frames[0],frames[-1]]  # 总是保留第一帧
        
        for i in range(1, len(frames)):
            prev_frame = keyframes[-1]["frame"]
            curr_frame = frames[i]["frame"]
            
            # 计算与上一关键帧的相似度
            similarity = calculate_ssim(prev_frame, curr_frame)
            
            # 如果相似度低于阈值，说明场景变化，保留变化后的帧（考虑稳定延迟）
            if similarity < threshold:
                # 确保不会超出frames范围
                stable_index = min(i + stable_delay, len(frames) - 1)
                stable_frame = frames[stable_index]
                
                # 检查是否已经包含了这个稳定帧
                if stable_frame not in keyframes:
                    keyframes.append(stable_frame)
        
        return keyframes

    def parse_timestamp_from_filename(filename: str) -> float:
        """从文件名解析时间戳"""
        # 匹配格式：00m28s000ms, 01m30s500ms 等
        match = re.search(r'(\d+)m(\d+)s(\d+)ms', filename)
        if match:
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            milliseconds = int(match.group(3))
            return minutes * 60 + seconds + milliseconds / 1000.0
        
        # 尝试其他可能的格式
        # 格式：128.5 (纯数字)
        match_float = re.search(r'(\d+\.\d+)', filename)
        if match_float:
            return float(match_float.group(1))
        
        # 格式：128 (整数)
        match_int = re.search(r'(\d+)', filename)
        if match_int:
            return float(match_int.group(1))
            
        raise ValueError(f"无法从文件名解析时间戳: {filename}")

    def load_cached_keyframes(output_dir: str) -> List[Dict[str, Any]]:
        """从已保存的图片加载关键帧数据"""
        if not os.path.exists(output_dir):
            return None
            
        keyframe_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.jpg')])
        if not keyframe_files:
            return None
            
        keyframes = []
        for filename in keyframe_files:
            try:
                # 从文件名提取时间戳
                timestamp = parse_timestamp_from_filename(filename)
                
                # 读取图片文件
                img_path = os.path.join(output_dir, filename)
                frame = cv2.imread(img_path)
                
                if frame is not None:
                    # 重新编码为base64
                    _, buffer = cv2.imencode(".jpg", frame)
                    image_b64 = base64.b64encode(buffer).decode("utf-8")
                    
                    keyframes.append({
                        "timestamp": timestamp,
                        "image_b64": image_b64,
                        "frame": frame
                    })
                else:
                    print(f"无法读取图片: {img_path}")
            except (ValueError, Exception) as e:
                print(f"加载缓存图片 {filename} 时出错: {e}")
                continue
                
        return keyframes if keyframes else None

    # 检查是否已有缓存的关键帧图片
    output_dir = f"./keyframes/{file_name}"
    if  start_time:
        output_dir += f"_{start_time}"
    if  end_time:
        output_dir += f"_{end_time}"
    cached_keyframes = load_cached_keyframes(output_dir)
    
    if cached_keyframes is not None:
        print(f"使用缓存的关键帧，数量: {len(cached_keyframes)}")
        return cached_keyframes

    # 如果没有缓存，进行正常处理
    print("未找到缓存，开始处理视频...")
    min_target, max_target = target_frame_count
    current_threshold = ssim_th
    iteration = 0
    frames = _extract_frames()
    
    # 保存所有帧（可选）
    all_frames_dir = f"./keyframes/{file_name}_all"
    save_keyframe_images_simple(frames, output_dir=all_frames_dir)
    print(f"帧数量: {len(frames)}")

    best_keyframes = []
    while iteration < max_iterations:
        print(f"\n迭代 {iteration + 1}, 使用SSIM阈值: {current_threshold:.3f}")
        keyframes = filter_frames_by_ssim(frames, current_threshold)
        keyframe_count = len(keyframes)
        
        print(f"筛选后关键帧数量: {keyframe_count}")

        # 检查是否在目标范围内
        if min_target <= keyframe_count <= max_target:
            print(f"达到目标帧数范围: {keyframe_count}")
            best_keyframes = keyframes
            break
        elif keyframe_count < min_target:
            # 帧数太少，降低阈值（更宽松）
            current_threshold += 0.01
            print(f"帧数过少 ({keyframe_count} < {min_target})，提升阈值至: {current_threshold:.3f}")
        else:
            # 帧数太多，提高阈值（更严格）
            current_threshold *= 0.9
            print(f"帧数过多 ({keyframe_count} > {max_target})，降低阈值至: {current_threshold:.3f}")
        
        # 防止阈值超出合理范围
        current_threshold = max(0.1, min(1, current_threshold))
        if current_threshold > 1 or current_threshold < 0.1:
            break
        iteration += 1
    
    # 如果迭代结束仍未达到目标，使用最后一次的结果
    if not best_keyframes and keyframes:
        print(f"达到最大迭代次数，使用当前结果: {len(keyframes)} 帧")
        best_keyframes = keyframes
    
    # 保存最终关键帧（作为缓存）
    if best_keyframes:
        save_keyframe_images_simple(best_keyframes, output_dir=output_dir)

    return best_keyframes


# ==========================
# 工具函数：调用 VLM 分析帧
# ==========================
def analyze_frame(image_b64: str, prompt: str, file_name: str, file_description: str) -> str:
    """
    使用 OpenAI GPT-4V 分析单帧图像内容。
    """
    response = client.chat.completions.create(
        model= model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                    }
                ]
            }
        ],
    )
    return response.choices[0].message.content.strip()

# ==========================
# 工具函数：LLM 总结文本
# ==========================
def summarize_text(texts: List[str], prompt: str) -> str:
    """
    使用 LLM 对多段文本进行总结。
    """
    combined = "\n\n".join(texts)
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": f"{prompt}\n\n以下是多段图像分析结果：\n{combined}"
            }
        ],
        max_tokens=500
    )
    return response.choices[0].message.content.strip()

# VIDEO_QA_PROMPT = """
# There is a video named {file_name}, here is its description {file_description}
# Use the key‑frames and (optional) transcription to answer.

# Transcription (may be empty):
# {transcription}

# Question:
# {question}
# """.strip()


VIDEO_QA_PROMPT = """
You are an expert video analysis assistant. Analyze the provided video content to answer the user's question accurately.

**VIDEO CONTEXT**
- Video Title: {file_name}
- Video Description: {file_description}

**AVAILABLE DATA**
1. Key Frames: Visual frames extracted from the video
2. Transcription (Optional): {transcription}

**QUESTION TO ANSWER**
{question}

**INSTRUCTIONS**
- Carefully examine all available visual and textual information
- If the transcription is empty, rely primarily on key frame analysis
- Provide specific, evidence-based answers
- If information is insufficient, clearly state what cannot be determined
""".strip()


# ==========================
# 工具 1：video_summary
# ==========================
# @mcp.tool(description="分析视频并生成摘要（通过关键帧提取和LLM总结）")
# async def video_summary(
# def video_summary(
#     path: str = Field(description="本地视频文件路径"),
#     vlm_prompt: Optional[str] = Field(default="请详细分析该页面内容", description="可选：视频帧vlm分析提示词，用于指定关注的关键信息如手机价格、详细页面等"),
#     llm_prompt: Optional[str] = Field(default="请基于上述关键帧内容分析总结", description="可选：llm总结提示词，用于指定需要解决问题例如多个不同页面商品间价格比较"),
#     start_time: Optional[float] = Field(default=None, description="可选：开始时间（秒），默认从视频开头"),
#     end_time: Optional[float] = Field(default=None, description="可选：结束时间（秒），默认到视频结尾"),
# ):
#     """
#     在指定时间段抽取关键帧读取内容，输出包括帧时间点，最后用LLM生成摘要。
#     适用于复杂任务，如"用户加入购物车的第一个商品内存版本比相同配色的最小内存版本大了多少？"
#     """
#     frames = extract_keyframes_with_ssim(
#         video_path=path,
#         start_time=start_time._default if hasattr(start_time, '_default') else start_time,
#         end_time=end_time._default if hasattr(end_time, '_default') else end_time
#      )
#     save_keyframe_images_simple(frames,output_dir=".\keyframes",)
#     transcription = ""

#     if not frames:
#         return "未提取到任何帧。"

#     print(len(frames))

#     user_message = [
#         {
#             "type": "text",
#             "text": VIDEO_QA_PROMPT.format(transcription=transcription, question=llm_prompt,file_name = file_name, file_description = file_description),
#         },
#         *(
#             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame["image_b64"]}"}}
#             for frame in frames
#         ),
#     ]

#     chat =  client.chat.completions.create(
#         model=model,
#         messages=[{"role": "user", "content": user_message}],
#         max_tokens=512,
#     )
#     print(chat)
    # results = []
    # for frame in frames:
    #     ts = frame["timestamp"]
    #     content = analyze_frame(frame["image_b64"], vlm_prompt)
    #     results.append(f"时间 {ts:.2f}s: {content}")

    # summary = summarize_text(results, llm_prompt)
    # return summary

from datetime import datetime
import json
@mcp.tool(description="获取视频基础信息,包括video_file，file_size_mb，duration_seconds，resolution，frame_rate，total_frames，aspect_ratio，file_modified")
async def get_basic_video_info(video_path: str) -> dict:
    """
    获取视频基础信息
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return {"error": "无法打开视频文件"}
    
    # 基础信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    # 视频尺寸
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 文件信息
    file_stats = os.stat(video_path)
    file_size_mb = file_stats.st_size / (1024 * 1024)
    
    cap.release()
    return {
        "video_file": os.path.basename(video_path),
        "file_size_mb": round(file_size_mb, 2),
        "duration_seconds": round(duration, 2),
        "resolution": f"{width}x{height}",
        "frame_rate": round(fps, 2),
        "total_frames": total_frames,
        "aspect_ratio": f"{width}:{height}",
        "file_modified": datetime.fromtimestamp(file_stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
    }

# ==========================
# 工具 2：video_understanding
# ==========================
@mcp.tool(description="抽取关键帧，理解视频内容并直接回答问题")
async def video_understanding(
# def video_understanding(
    path: str = Field(description="本地视频文件路径"),
    vlm_prompt: Optional[str] = Field(default=None, description="可选：问题提示词，用于指定需要回答的具体问题"),
    start_time: Optional[float] = Field(default=None, description="可选：开始时间（秒），默认从视频开头"),
    end_time: Optional[float] = Field(default=None, description="可选：结束时间（秒），默认到视频结尾"),
):
    """
    在指定时间段抽帧并直接回答问题，适用于简单任务。
    如"在第30秒到第32秒中，搜索框中的文本的第二个汉字是什么？"
    """
    from pypinyin import lazy_pinyin
    from pathlib import Path

    file_name = "".join(lazy_pinyin(Path(path).stem))   
    log.info(f"file_name:{file_name}")
    file_description = await get_basic_video_info(path)

    path_str = str(path).strip()
    if path_str:
        if path_str.startswith('./') or (not os.path.isabs(path_str) and '/' not in path_str):
            here = Path(__file__).resolve()
            root_path = here.parent.parent.parent
            if root_path.exists():
                path = str(root_path / path_str.lstrip('./'))
    log.info(f"path:{path}")

    transcription = ""
    try:
        start_time = float(start_time) if start_time is not None else None
    except (ValueError, TypeError):
        start_time = None

    try:
        end_time = float(end_time) if end_time is not None else None
    except (ValueError, TypeError):
        end_time = None
    frames = extract_keyframes_with_ssim(path, start_time, end_time, file_name)  # 低频抽一帧即可
    log.info(f"frames num：{len(frames)}")



    if not frames:
        return "未提取到任何帧。"




    # user_message = [
    #     {
    #         "type": "text",
    #         "text": VIDEO_QA_PROMPT.format(transcription=transcription, question=vlm_prompt,file_name = file_name, file_description = file_description),
    #     },
    #     *(
    #         {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame['image_b64']}"}}
    #         for frame in frames
    #     ),
    # ]
    
    # resp =  client.chat.completions.create(
    #     model=model,
    #     messages=[{"role": "user", "content": user_message}],
    # )

    return {
        "status": "success",
        # "analysis": resp.choices[0].message.content,
        "analysis": "resp.choices[0].message.content",
    }


# ==========================
# 使用示例
# ==========================
if __name__ == "__main__":
    mcp.run()
    # async def _test():
    #     for i in ['采销介绍','逛京东_副本','买iphone_副本','grgww5','iphone','kadj4','kxjs3','qkvn6']:
    #         video_path = rf"F:\project\agent\jd\hzz\jd_multi\test\{i}.mp4"


    #         # 示例 2：简单问题回答
    #         prompt2 = "用户加入购物车的第一个商品内存版本比相同配色的最小内存版本大了多少？单位GB，直接输出数字"
    #         answer = await  video_understanding(video_path, prompt2)
    #         # answer = video_understanding(video_path, prompt2, start_time=30, end_time=32)
    #         print(answer)
    # asyncio.run(_test())
