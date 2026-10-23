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

def check_target_text_tesseract(img):
    """增强版 Tesseract 识别，增加预处理并放宽白名单"""
    try:
        # 1. 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. 将图片放大 2 倍，显著提升 Tesseract 对小字体的识别率
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        # 3. 强力二值化：把背景彻底变黑，文字彻底变白
        # 这里的 150 是阈值，可以根据实际情况微调 (100~180之间)
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        # 4. 放宽白名单：允许识别大小写字母和数字，避免连笔字被强制丢弃
        # --psm 11 表示按稀疏文本查找
        custom_config = r'--oem 3 --psm 11 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        
        # 将处理后黑白分明的图像送入引擎
        text = pytesseract.image_to_string(binary, config=custom_config)
        
        if "46" in text:
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
    next_start_index = state.get("next_start_index", 1)
    
    print(f"GitHub 虚拟机启动 | 从 {next_start_index} 向前推进")
    current_index = next_start_index
    consecutive_empty = 0
    
    while consecutive_empty < MAX_CONSECUTIVE_EMPTY:
        url = BASE_URL_TEMPLATE.format(current_index)
        try:
            # 流式下载避免爆内存
            r = requests.get(url, stream=True, timeout=10)
            if r.status_code == 200:
                print(f"[发现新图] 序号 {current_index}")
                img_array = np.asarray(bytearray(r.content), dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if check_target_text_tesseract(img):
                    push_to_telegram(url, current_index)
                
                consecutive_empty = 0
                state["next_start_index"] = current_index + 1
                save_state(state)
            else:
                consecutive_empty += 1
                print(f"[空图] 序号 {current_index} (连续为空: {consecutive_empty}/50)")
        except Exception as e:
            print(f"网络异常: {e}")
            
        current_index += 1
        time.sleep(0.3)
        
    print("触及连续空图熔断，当前任务结束，进度已保存。")

if __name__ == "__main__":
    run_scanner()
