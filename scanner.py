import os
import json
import time
import requests
import numpy as np
import cv2
import pytesseract

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

# 在文件最顶部引入并初始化（全局只加载一次模型，速度极快）
from rapidocr_onnxruntime import RapidOCR
ocr_engine = RapidOCR()

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
                    print(f"  -> 🎯 命中特征文字: {text}")
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
    state = load_state()
    # 读取历史空缺列表（转为集合方便去重）和下次扫描起点
    historical_empties = set(state.get("historical_empties", []))
    next_start_index = state.get("next_start_index", 1)
    
    print(f"启动扫描任务 | 待重扫历史空号: {len(historical_empties)} 个 | 本次推进起点: {next_start_index}")

    # ==========================================
    # 阶段一：优先重扫历史空缺序号 (如 10, 15, 21)
    # ==========================================
    if historical_empties:
        print("--- 开始阶段一：历史空号回填扫描 ---")
        still_empty = set()
        
        for idx in sorted(list(historical_empties)):
            url = BASE_URL_TEMPLATE.format(idx)
            try:
                r = requests.get(url, stream=True, timeout=10)
                # 状态码为 200 且没有被跳转到 noimg 占位图，说明空号终于更新图片了
                if r.status_code == 200 and "noimg/fit" not in r.url:
                    print(f"[回填成功] 历史空号 {idx} 已更新图片！")
                    img_array = np.asarray(bytearray(r.content), dtype=np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    
                    if check_target_text_tesseract(img):
                        push_to_telegram(url, idx)
                    # 成功出图后，不再放回 still_empty，相当于从待扫列表中彻底移除
                else:
                    # 依然是空图，继续保留，留给下次任务重扫
                    still_empty.add(idx)
            except Exception as e:
                print(f"[网络异常] 历史空号 {idx}: {e}")
                still_empty.add(idx) # 网络报错也先保留，防止漏扫
                
            time.sleep(0.3)
            
        # 阶段一结束，保存最新的历史空号列表
        historical_empties = still_empty
        state["historical_empties"] = sorted(list(historical_empties))
        save_state(state)

    # ==========================================
    # 阶段二：从高水位线起点向前推进扫描新图
    # ==========================================
    print(f"--- 开始阶段二：从 {next_start_index} 向前推进 ---")
    current_index = next_start_index
    consecutive_empty = 0
    temp_empties = [] # 临时缓冲池：记录当前连续遇到的空号
    
    while consecutive_empty < MAX_CONSECUTIVE_EMPTY:
        url = BASE_URL_TEMPLATE.format(current_index)
        try:
            r = requests.get(url, stream=True, timeout=10)
            
            # 发现有效新图
            if r.status_code == 200 and "noimg/fit" not in r.url:
                print(f"[发现新图] 序号 {current_index}")
                
                # 1. 如果在这张有效图之前遇到过零星的空号（未满100个），将它们正式归入历史空缺列表
                if temp_empties:
                    historical_empties.update(temp_empties)
                    state["historical_empties"] = sorted(list(historical_empties))
                    temp_empties = [] # 清空临时缓冲池
                
                # 2. 解码并进行 OCR 识别
                img_array = np.asarray(bytearray(r.content), dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if check_target_text_tesseract(img):
                    push_to_telegram(url, current_index)
                
                # 3. 重置连续空图计数器，并将下一次的起点推进到 (当前有效图序号 + 1)
                consecutive_empty = 0
                state["next_start_index"] = current_index + 1
                save_state(state)
                
            else:
                # 遇到空图（404 或被跳转到 noimg 占位图）
                consecutive_empty += 1
                temp_empties.append(current_index) # 先放进临时缓冲池观望
                print(f"[空图] 序号 {current_index} (连续为空: {consecutive_empty}/{MAX_CONSECUTIVE_EMPTY})")
                
        except Exception as e:
            print(f"[网络异常] 序号 {current_index}: {e}")
            consecutive_empty += 1
            temp_empties.append(current_index)
            
        current_index += 1
        time.sleep(0.3)
        
    # ==========================================
    # 触及熔断收尾
    # ==========================================
    # 注意：最后连续碰壁导致熔断的这 100 个 temp_empties 不会写入 historical_empties，
    # 因为 state["next_start_index"] 依然停留在 (最后一张有效图 + 1) 的位置，下次启动阶段二时会自然重扫它们。
    print(f"触及连续 {MAX_CONSECUTIVE_EMPTY} 个空图熔断，当前任务结束。")
    print(f"状态已保存 | 累计历史空号: {len(state.get('historical_empties', []))} 个 | 下次推进起点: {state.get('next_start_index')}")

if __name__ == "__main__":
    run_scanner()
