import os
import json
import time
import requests
from requests.adapters import HTTPAdapter
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
from rapidocr_onnxruntime import RapidOCR

# ==========================
# 基础与并发配置
# ==========================
TG_TOKEN = os.environ.get("TG_TOKEN")
TG_CHAT_ID = os.environ.get("TG_CHAT_ID")

SCAN_URL_TEMPLATE = "https://i.kfs.io/album/global/{},0v1/fit/800x800.jpg"
HIGH_RES_URL_TEMPLATE = "https://i.kfs.io/album/global/{},0v1/fit/1000x1000.jpg"

STATE_FILE = "scan_state.json"
MAX_CONSECUTIVE_EMPTY = 500   # 应对大面积空洞，调大熔断阈值
MAX_WORKERS = 8              # RapidOCR 线程数适配 7GB 内存
BATCH_SIZE = 160              # 每批次扫描数量
MAX_RUNTIME = 2*60*60      # 2小时安全下班闹钟

# # 全局高速连接池
# session = requests.Session()
# adapter = HTTPAdapter(pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS, max_retries=1)
# session.mount('https://', adapter)
# session.mount('http://', adapter)
# session.headers.update({"User-Agent": "Mozilla/5.0"})

# 初始化全局 RapidOCR 引擎 (只需加载一次)
ocr_engine = RapidOCR()

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {"historical_empties": [], "next_start_index": 1}

def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=4)

def check_target_text(img):
    """极简 RapidOCR 识别，自带高抗干扰能力"""
    try:
        # 深度学习引擎无需手动做黑白/对比度预处理，直接传原图
        result, _ = ocr_engine(img)
        if result:
            for line in result:
                text = line[1]
                # 兼容 46、坂道罗马音以及汉字
                if "46" in text or "zaka" in text.lower() or "坂" in text:
                    return True
        return False
    except Exception as e:
        print(f"OCR Error: {e}")
        return False

def push_to_telegram(index):
    hd_url = HIGH_RES_URL_TEMPLATE.format(index)
    print(f"发现目标封面 [序号 {index}]，正在推送高清图...")
    api_url = f"https://api.telegram.org/bot{TG_TOKEN}/sendPhoto"
    caption = f"发现包含 46 的新封面！\n序号: {index}\n{hd_url}"
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # 👇 这里改成了 requests.post
            response = requests.post(
                api_url, 
                data={"chat_id": TG_CHAT_ID, "photo": hd_url, "caption": caption}, 
                timeout=15
            )
            if response.status_code == 200:
                print(f"序号 {index} 推送成功！")
                break
            else:
                print(f"推送异常: {response.text}")
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                print(f"推送失败 [序号 {index}]: {e}")

def process_single_index(idx):
    url = SCAN_URL_TEMPLATE.format(idx)
    try:
        # 核心提速修改：
        # 1. 直接用 requests.get 替代 session，避免底层连接池锁死
        # 2. 加上 "Connection": "close"，查完立刻断开，不占用对方服务器资源，极大降低被限流概率
        # 3. 超时时间设为 (连接 1.5秒, 读取 1.5秒)。GitHub 网速极快，只要 1.5 秒没下完，直接判定为空跳过！
        r = requests.get(
            url, 
            allow_redirects=False, 
            timeout=(1.5, 1.5), 
            headers={"User-Agent": "Mozilla/5.0", "Connection": "close"}
        )
        
        if r.status_code == 200 and "noimg" not in r.url:
            img_array = np.asarray(bytearray(r.content), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is not None:
                is_target = check_target_text(img)
                return (idx, True, is_target)
        return (idx, False, False)
    except Exception:
        # 只要超时或者网络卡顿，直接返回没图，后续会有连续空图兜底，不要死等
        return (idx, False, False)

def run_scanner():
    start_time = time.time()
    state = load_state()
    historical_empties = set(state.get("historical_empties", []))
    next_start_index = state.get("next_start_index", 1)
    
    print(f"RapidOCR 并发引擎启动 ({MAX_WORKERS}线程) | 起点: {next_start_index}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # ================== 阶段一：回填空号 ==================
        if historical_empties:
            print("--- 开始阶段一：历史空号并发回填 ---")
            sorted_empties = sorted(list(historical_empties))
            
            for i in range(0, len(sorted_empties), BATCH_SIZE):
                if time.time() - start_time > MAX_RUNTIME:
                    print("运行已满设定时间，阶段一暂停...")
                    break
                    
                batch = sorted_empties[i : i + BATCH_SIZE]
                results = list(executor.map(process_single_index, batch))
                
                for idx, has_img, is_target in results:
                    if has_img:
                        print(f"[回填成功] 历史空号 {idx} 已出图！")
                        if is_target: push_to_telegram(idx)
                        historical_empties.discard(idx)
                
                state["historical_empties"] = sorted(list(historical_empties))
                save_state(state)

        # ================== 阶段二：向前推进 ==================
        print(f"--- 开始阶段二：从 {next_start_index} 向前推进 ---")
        current_index = next_start_index
        consecutive_empty = 0
        temp_empties = []
        
        while consecutive_empty < MAX_CONSECUTIVE_EMPTY:
            if time.time() - start_time > MAX_RUNTIME:
                print(f"运行已满设定时间，安全下班！")
                break
                
            batch_indices = list(range(current_index, current_index + BATCH_SIZE))
            results = list(executor.map(process_single_index, batch_indices))
            
            hit_circuit_breaker = False
            for idx, has_img, is_target in results:
                if has_img:
                    if temp_empties:
                        historical_empties.update(temp_empties)
                        state["historical_empties"] = sorted(list(historical_empties))
                        temp_empties = []
                    
                    if is_target: push_to_telegram(idx)
                        
                    consecutive_empty = 0
                    state["next_start_index"] = idx + 1
                else:
                    consecutive_empty += 1
                    temp_empties.append(idx)
                    if consecutive_empty >= MAX_CONSECUTIVE_EMPTY:
                        hit_circuit_breaker = True
                        break
            
            save_state(state)
            print(f"[批次完成] 扫描至 {batch_indices[-1]} | 当前连续为空: {consecutive_empty}/{MAX_CONSECUTIVE_EMPTY}")
            
            if hit_circuit_breaker:
                print(f"触及连续 {MAX_CONSECUTIVE_EMPTY} 个空图熔断。")
                break
                
            current_index += BATCH_SIZE

    print(f"任务结束 | 下次起点: {state.get('next_start_index')}")

if __name__ == "__main__":
    run_scanner()
