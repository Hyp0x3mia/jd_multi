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
SSIM_TH = 0.5

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
    简化版：先清空输出目录，再用索引命名保存关键帧图片
    """
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
                filename = f"{filename_prefix}_{i:05d}.{image_format}"
                filepath = os.path.join(output_dir, filename)

                print(filepath)
                # 转换为字符串并保存
                success = cv2.imwrite(str(filepath), img)
                # success = cv2.imwrite(filepath, img)
                if success:
                    saved_count += 1
                    print(f"保存: {filename} (时间戳: {frame_info['timestamp']:.3f}s)")
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





####111
def extract_keyframes_with_ssim(
    video_path: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    fps: float = 1.0,
    ssim_th: float = SSIM_TH,
    stable_delay: int = 5,
    target_frame_count: Tuple[int, int] = (12, 20)  # 目标帧数范围
) -> List[Dict[str, Any]]:
    
    def _extract_frames(current_threshold):
        """内部函数：使用指定阈值提取关键帧"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("无法打开视频文件")

        total_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / total_fps

        start = start_time if isinstance(start_time, (int, float)) else 0
        end = end_time if isinstance(end_time, (int, float)) else duration
        frame_interval = max(1, int(total_fps / fps))
        keyframes: List[Dict[str, Any]] = []
        last_frame = None
        change_detected = False
        stable_counter = 0
        
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
                # 第一帧直接保留
                if last_frame is None:
                    _, buffer = cv2.imencode(".jpg", frame)
                    image_b64 = base64.b64encode(buffer).decode("utf-8")
                    keyframes.append({
                        "timestamp": timestamp,
                        "image_b64": image_b64,
                        "ssim": None
                    })
                    last_frame = frame
                else:
                    sim = _compute_ssim(last_frame, frame)
                    
                    if sim < current_threshold:
                        # 检测到变化，设置标志但不立即保存
                        change_detected = True
                        stable_counter = 0
                        last_frame = frame  # 更新参考帧
                    else:
                        # 没有变化或变化很小
                        if change_detected:
                            # 在变化后等待一段时间再保存
                            stable_counter += 1
                            if stable_counter >= stable_delay:
                                # 保存稳定的中间帧
                                _, buffer = cv2.imencode(".jpg", frame)
                                image_b64 = base64.b64encode(buffer).decode("utf-8")
                                keyframes.append({
                                    "timestamp": timestamp,
                                    "image_b64": image_b64,
                                    "ssim": sim
                                })
                                last_frame = frame
                                change_detected = False
                                stable_counter = 0
                        else:
                            # 正常情况，没有检测到变化，直接保存
                            _, buffer = cv2.imencode(".jpg", frame)
                            image_b64 = base64.b64encode(buffer).decode("utf-8")
                            keyframes.append({
                                "timestamp": timestamp,
                                "image_b64": image_b64,
                                "ssim": sim
                            })
                            last_frame = frame
        
        cap.release()
        return keyframes
    
    def _resample_frames(frames, target_count):
        """对关键帧进行二次抽帧"""
        if len(frames) <= target_count:
            return frames
            
        # 按时间戳排序
        sorted_frames = sorted(frames, key=lambda x: x["timestamp"])
        
        # 保留第一帧和最后一帧
        result = [sorted_frames[0]]
        
        if target_count > 2:
            # 计算需要额外选择的帧数
            remaining_count = target_count - 2
            
            # 计算时间间隔
            total_duration = sorted_frames[-1]["timestamp"] - sorted_frames[0]["timestamp"]
            interval = total_duration / (remaining_count + 1)
            
            # 选择时间上均匀分布的帧
            target_timestamps = [sorted_frames[0]["timestamp"] + (i + 1) * interval 
                               for i in range(remaining_count)]
            
            # 为每个目标时间戳找到最接近的帧
            for target_ts in target_timestamps:
                closest_frame = min(sorted_frames[1:-1], 
                                  key=lambda x: abs(x["timestamp"] - target_ts))
                if closest_frame not in result:
                    result.append(closest_frame)
            
        result.append(sorted_frames[-1])
        return sorted(result, key=lambda x: x["timestamp"])
    
    # 第一次提取
    keyframes = _extract_frames(ssim_th)
    frame_count = len(keyframes)
    
    min_target, max_target = target_frame_count
    
    print(f"初始提取帧数: {frame_count}, 使用阈值: {ssim_th}")
    
    # 如果帧数不在目标范围内，调整阈值
    if frame_count < min_target:
        # 帧数太少，降低阈值（更敏感）
        adjusted_threshold = ssim_th * 0.8  # 降低阈值
        print(f"帧数过少，调整阈值到: {adjusted_threshold}")
        keyframes = _extract_frames(adjusted_threshold)
        frame_count = len(keyframes)
        
    elif frame_count > max_target:
        # 帧数太多，提高阈值（更严格）
        adjusted_threshold = ssim_th * 1.2  # 提高阈值
        print(f"帧数过多，调整阈值到: {adjusted_threshold}")
        keyframes = _extract_frames(adjusted_threshold)
        frame_count = len(keyframes)
    
    # 如果调整阈值后仍然太多，进行二次抽帧
    if frame_count > max_target:
        print(f"调整阈值后仍有 {frame_count} 帧，进行二次抽帧")
        keyframes = _resample_frames(keyframes, max_target)
    
    print(f"最终帧数: {len(keyframes)}")
    return keyframes


####111



def deduplicate_keyframes_by_ssim(
    keyframes: List[Dict[str, Any]], 
    ssim_threshold: float = 0.85
) -> List[Dict[str, Any]]:
    """
    根据SSIM值去除相似的关键帧
    
    Args:
        keyframes: 原始关键帧列表
        ssim_threshold: SSIM阈值，高于此值认为帧相似
        
    Returns:
        去重后的关键帧列表
    """
    if not keyframes:
        return []
    
    # 第一帧总是保留
    deduplicated = [keyframes[0]]
    
    for i in range(1, len(keyframes)):
        current_frame = keyframes[i]
        last_kept_frame = deduplicated[-1]
        
        # 如果当前帧与上一保留帧的SSIM值低于阈值，则保留
        if current_frame["ssim"] is None or last_kept_frame["ssim"] is None:
            # 如果任一帧没有SSIM值，默认保留
            deduplicated.append(current_frame)
        elif current_frame["ssim"] < ssim_threshold:
            deduplicated.append(current_frame)
        # 如果SSIM高于阈值，跳过当前帧（不保留）
    
    print(f"去重前: {len(keyframes)} 帧, 去重后: {len(deduplicated)} 帧")
    return deduplicated


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
# @mcp.tool(description="获取视频基础信息,包括video_file，file_size_mb，duration_seconds，resolution，frame_rate，total_frames，aspect_ratio，file_modified")
def get_basic_video_info(video_path: str) -> dict:
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
    file_name = "".join(lazy_pinyin(path.split('\\')[-1].split('.')[0]))
    file_description = get_basic_video_info(path)

    path_str = str(path).strip()
    if path_str:
        if path_str.startswith('./') or (not os.path.isabs(path_str) and '/' not in path_str):
            here = Path(__file__).resolve()
            root_path = here.parent.parent.parent
            if root_path.exists():
                path = str(root_path / path_str.lstrip('./'))
    log.info(f"path_str：{path_str}")

    transcription = ""

    print(VIDEO_QA_PROMPT.format(transcription=transcription, question=vlm_prompt,file_name = file_name, file_description = file_description))

    frames = extract_keyframes_with_ssim(path, start_time, end_time, fps=0.5)  # 低频抽一帧即可
    log.info(f"frames num：{len(frames)}")

    save_keyframe_images_simple(frames,output_dir=f".\keyframes\{file_name}",)
    # save_keyframe_images_simple(frames,output_dir=f".\keyframes\\",)


    if not frames:
        return "未提取到任何帧。"
    # 取中间一帧
    frame = frames[len(frames) // 2]



    user_message = [
        {
            "type": "text",
            "text": VIDEO_QA_PROMPT.format(transcription=transcription, question=vlm_prompt,file_name = file_name, file_description = file_description),
        },
        *(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame['image_b64']}"}}
            for frame in frames
        ),
    ]

    resp =  client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user_message}],
        max_tokens=512,
    )

    return {
        "status": "success",
        "analysis": resp.choices[0].message.content,
    }


# ==========================
# 使用示例
# ==========================
if __name__ == "__main__":
    mcp.run()
    # async def _test():

    #     video_path = "./cache_dir/uploads/20251106114244_买iphone_副本.mp4"


    #     # 示例 2：简单问题回答
    #     prompt2 = "用户加入购物车的第一个商品内存版本比相同配色的最小内存版本大了多少？单位GB，直接输出数字"
    #     answer = await  video_understanding(video_path, prompt2)
    #     # answer = video_understanding(video_path, prompt2, start_time=30, end_time=32)
    #     print(answer)
    # asyncio.run(_test())
