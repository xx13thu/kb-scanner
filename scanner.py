import os
import json
import time
import requests
import numpy as np
import cv2
from rapidocr_onnxruntime import RapidOCR
ocr_engine = RapidOCR()

# ==========================
# 环境变量配置 (在 GitHub Actions 中设置)
# ==========================
TG_TOKEN = os.environ.get("TG_TOKEN")
TG_CHAT_ID = os.environ.get("TG_CHAT_ID")

BASE_URL_TEMPLATE = "https://i.kfs.io/album/global/{},0v3/fit/1000x1000.jpg"
# https://i.kfs.io/album/global/298469722,0v3/fit/800x800.jpg
STATE_FILE = "scan_state.json"
MAX_CONSECUTIVE_EMPTY = 100

def load_state():
    with open(STATE_FILE, 'r') as f: return json.load(f)

def save_state(state):
    with open(STATE_FILE, 'w') as f: json.dump(state, f, indent=4)

def check_target_text_tesseract(img):
    """使用深度学习引擎 RapidOCR，无需繁琐的图像二值化预处理"""
    try:
        # 直接把彩色原图丢进去识别
        result, _ = ocr_engine(img)
        if result:
            for line in result:
                text = line[1] # 拿到识别出的文本字符串
                # 只要包含 46、坂道罗马音 zaka、或者汉字“坂”
                if "46" in text or "zaka" in text.lower() or "坂" in text:
                    print(f"  -> 命中特征文字: {text}")
                    return True
        return False
    except Exception as e:
        print(f"OCR Error: {e}")
        return False

def push_to_telegram(image_url, index):
    print(f"发现目标封面 [序号 {index}]，正在推送...")
    api_url = f"https://api.telegram.org/bot{TG_TOKEN}/sendPhoto"
    caption = f"发现包含 46 的新封面！\n序号: {index}\n{image_url}"
    try:
        requests.post(api_url, data={"chat_id": TG_CHAT_ID, "photo": image_url, "caption": caption}, timeout=10)
    except Exception as e:
        print(f"推送失败: {e}")

def run_scanner():
    # 新增：记录启动时间，设置最大运行时长为 2小时50分钟
    # 确保在 3 小时的定时周期到来前主动下班，留出时间把进度 git push 回仓库
    start_time = time.time()
    MAX_RUNTIME = 2*60*60 + 50*60  

    state = load_state()
    historical_empties = set(state.get("historical_empties", []))
    next_start_index = state.get("next_start_index", 1)
    
    print(f"🚀 启动扫描任务 | 待重扫历史空号: {len(historical_empties)} 个 | 本次推进起点: {next_start_index}")

    # ==========================================
    # 阶段一：优先重扫历史空缺序号
    # ==========================================
    if historical_empties:
        print("--- 开始阶段一：历史空号回填扫描 ---")
        
        # 遍历历史空号的副本，方便在原集合中实时剔除已出图的序号
        for idx in sorted(list(historical_empties)):
            # 检查是否到了下班时间
            if time.time() - start_time > MAX_RUNTIME:
                print("运行已满设定分钟，阶段一暂停，准备保存进度交接给下一班...")
                break

            url = BASE_URL_TEMPLATE.format(idx)
            try:
                r = requests.get(url, stream=True, timeout=10)
                if r.status_code == 200 and "noimg/fit" not in r.url:
                    print(f"[回填成功] 历史空号 {idx} 已更新图片！")
                    img_array = np.asarray(bytearray(r.content), dtype=np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    
                    if check_target_text_tesseract(img):
                        push_to_telegram(url, idx)
                    
                    # 出图后立刻从历史空号集合中移除，并实时保存
                    historical_empties.discard(idx)
                    state["historical_empties"] = sorted(list(historical_empties))
                    save_state(state)
            except Exception as e:
                print(f"[网络异常] 历史空号 {idx}: {e}")
                
            time.sleep(0.3)

    # ==========================================
    # 阶段二：从高水位线起点向前推进扫描新图
    # ==========================================
    print(f"--- 开始阶段二：从 {next_start_index} 向前推进 ---")
    current_index = next_start_index
    consecutive_empty = 0
    temp_empties = []
    
    while consecutive_empty < MAX_CONSECUTIVE_EMPTY:
        # 每次循环前检查是否到了设定下班时间
        if time.time() - start_time > MAX_RUNTIME:
            print(f"运行已满设定时间，主动停止扫描！当前已安全推进至 {state.get('next_start_index')}。")
            break

        url = BASE_URL_TEMPLATE.format(current_index)
        try:
            r = requests.get(url, stream=True, timeout=10)
            
            if r.status_code == 200 and "noimg/fit" not in r.url:
                print(f"[发现新图] 序号 {current_index}")
                
                if temp_empties:
                    historical_empties.update(temp_empties)
                    state["historical_empties"] = sorted(list(historical_empties))
                    temp_empties = []
                
                img_array = np.asarray(bytearray(r.content), dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if check_target_text_tesseract(img):
                    push_to_telegram(url, current_index)
                
                consecutive_empty = 0
                state["next_start_index"] = current_index + 1
                save_state(state)
                
            else:
                consecutive_empty += 1
                temp_empties.append(current_index)
                print(f"[空图] 序号 {current_index} (连续为空: {consecutive_empty}/{MAX_CONSECUTIVE_EMPTY})")
                
        except Exception as e:
            print(f"[网络异常] 序号 {current_index}: {e}")
            consecutive_empty += 1
            temp_empties.append(current_index)
            
        current_index += 1
        time.sleep(0.3)
        
    print(f"本次任务结束 | 累计历史空号: {len(state.get('historical_empties', []))} 个 | 下次推进起点: {state.get('next_start_index')}")

if __name__ == "__main__":
    run_scanner()
