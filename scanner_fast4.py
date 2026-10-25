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
MAX_RUNTIME = 2*60*60 + 50*60        # 2小时50分钟安全下班闹钟

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
            data = json.load(f)
            
            raw_empties = data.get("historical_empties", {})
            new_empties = {}
            
            # 兼容老版本的列表，并强制清洗掉堆积的几万个死链，只留最后 1000 个
            if isinstance(raw_empties, list):
                for idx in raw_empties[-1000:]:
                    new_empties[int(idx)] = 0  # 初始重试次数设为 0
            else:
                # 已经是新版字典，强制将键转为整数，将值(重试次数)转为整数
                for k, v in raw_empties.items():
                    new_empties[int(k)] = int(v)
                    
            return {
                "historical_empties": new_empties,
                "next_start_index": int(data.get("next_start_index", 1))
            }
    return {"historical_empties": {}, "next_start_index": 1}

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
    max_retries = 2  # 削减局部重试次数，把长效容错交给外层的 10次/JSON 淘汰制
    
    # 启用会话级别的 keep-alive，避免频繁 TCP 握手
    headers = {"User-Agent": "Mozilla/5.0", "Connection": "keep-alive"}

    for attempt in range(max_retries):
        try:
            # 🌟 核心提速：stream=True 意味着只请求“报文头”，瞬间返回，绝不提前下载图片内容！
            with requests.get(url, allow_redirects=False, stream=True, timeout=(1.5, 2.5), headers=headers) as r:
                
                final_url = url
                
                # 场景一：发生跳转 (301/302)
                if r.status_code in (301, 302):
                    final_url = r.headers.get("Location", "")
                    r.close() # 立刻关闭旧的探路连接
                    
                    if "noimg" in final_url or not final_url:
                        return (idx, False, False, "")
                        
                    # 是目标变体 (例如 0v2)，去下载真图
                    r_real = requests.get(final_url, timeout=(1.5, 3.0), headers=headers)
                    if r_real.status_code == 200:
                        img_array = np.asarray(bytearray(r_real.content), dtype=np.uint8)
                        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                        if img is not None:
                            is_target = check_target_text(img)
                            return (idx, True, is_target, final_url)
                            
                # 场景二：直接出图 (200 OK，没有跳转)
                elif r.status_code == 200:
                    if "noimg" in url:
                        return (idx, False, False, "")
                        
                    # 🌟 核心提速：既然已经探明是 200 真图，直接从之前的流里读取全部内容，绝不进行第二次请求！
                    content = r.content 
                    img_array = np.asarray(bytearray(content), dtype=np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if img is not None:
                        is_target = check_target_text(img)
                        return (idx, True, is_target, final_url)

            # 非 200/301/302 统统判定失败
            return (idx, False, False, "")
            
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            # 去掉 1 秒硬休眠，仅做极短的毫秒级喘息，不让线程罚站
            if attempt < max_retries - 1:
                time.sleep(0.5)
                continue
            else:
                return (idx, False, False, "")
        except Exception:
            return (idx, False, False, "")
            
    return (idx, False, False, "")

def run_scanner():
    start_time = time.time()
    state = load_state()
    
    historical_empties = state["historical_empties"]
    next_start_index = state["next_start_index"]
    
    print(f"极速并发引擎启动 | 待重扫空号: {len(historical_empties)} 个 | 起点: {next_start_index}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # ==========================================
        # 阶段一：历史空号并发回填 (带 10 次淘汰制)
        # ==========================================
        if historical_empties:
            print("--- 开始阶段一：历史空号回填 (智能淘汰) ---")
            sorted_empties = sorted(historical_empties.keys())
            
            for i in range(0, len(sorted_empties), BATCH_SIZE):
                if time.time() - start_time > MAX_RUNTIME:
                    print("运行已满设定时间，阶段一暂停...")
                    break
                    
                batch = sorted_empties[i : i + BATCH_SIZE]
                results = list(executor.map(process_single_index, batch))
                
                for idx, has_img, is_target, final_url in results:
                    if has_img:
                        print(f"[回填成功] 历史空号 {idx} 已出图！")
                        if is_target: push_to_telegram(idx)
                        historical_empties.pop(idx, None) # 出图后彻底删除
                    else:
                        historical_empties[idx] += 1
                        # 核心机制：重试超过 10 次永久抹除
                        if historical_empties[idx] > 10:
                            historical_empties.pop(idx, None)
                
                state["historical_empties"] = historical_empties
                save_state(state)

        # ==========================================
        # 阶段二：向高水位线推进
        # ==========================================
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
            for idx, has_img, is_target, final_url in results:
                if has_img:
                    if temp_empties:
                        for empty_idx in temp_empties:
                            historical_empties[empty_idx] = 0
                        state["historical_empties"] = historical_empties
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

    print(f"任务结束 | 剩余挂起空号: {len(state.get('historical_empties', {}))} 个 | 下次起点: {state.get('next_start_index')}")

if __name__ == "__main__":
    run_scanner()