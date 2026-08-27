# Using LLM to filter distinct persona comments (batched with tqdm)


from dotenv import load_dotenv
from tqdm import tqdm
import os
import json
import argparse
import re
import math
from openai import AsyncOpenAI
import asyncio
from tqdm.asyncio import tqdm as tqdm_asyncio

load_dotenv()

semaphore = asyncio.Semaphore(20)


parser = argparse.ArgumentParser(description="construct reply-answer pairs of comments (batched)")
parser.add_argument("--person_name", type=str, default="Influencer_A")
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--model", type=str, default="gpt-4o-mini")
args = parser.parse_args()

target = args.person_name
batch_size = args.batch_size
model_name = args.model

azure_api_key = os.environ['AZURE_OPENAI_API_KEY']
azure_api_version=os.environ['AZURE_API_VERSION']
azure_endpoint = os.environ['AZURE_ENDPOINT']


data_path = f"../data/{target}/for_sample_dataset/test_set_comment_reply_pairs.json"
output_folder_path = f"../data/{target}/for_sample_dataset"

with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)


def build_prompt(fan: str, reply: str) -> str:
    return f"""
你是文本分類與評分助手，負責判斷「{target} 的人設/個人風格展現程度」。請閱讀一段粉絲留言與 {target} 的回覆，並根據以下標準，以 0~1 區間給出分數：

### 評分說明：
- **0 分**：回覆完全中性、制式、無個人色彩，或是包含股票知識問答，例如：「謝謝你」、「這支股票可以繼續抱」
- **1 分**：回覆充分展現個人風格，可能包含：
  - 明確的情緒（開心、生氣、感動等）
  - 個人語氣或口頭禪
  - 口語化用詞、幽默、貼近粉絲語言
  - 主觀觀點、價值判斷
  - Emoji（若搭配文字，才會加分；**單獨出現不加分**）

### 注意事項：
- **只包含 Emoji 或一兩個詞的回覆（如「❤️」、「謝謝支持！」）不得高於 0.6 分**
- **純粹回答知識/資訊問題、無個人情緒或語氣的回覆不得高於 0.4 分**
- 請考慮回覆的 **長度、情緒強度、語言風格**，這些都會影響人設的展現程度

### 回傳格式：
只輸出 JSON，格式如下：
{{"score": 0.xx}}

### 評分資料：
粉絲: {fan}
{target}: {reply}
""".strip()

json_score_re = re.compile(r'\{\s*"score"\s*:\s*([01](?:\.\d+)?|\.\d+)\s*\}', re.IGNORECASE)
num_re = re.compile(r'([01](?:\.\d+)?|\.\d+)')

def clamp01(x: float) -> float:
    if math.isnan(x): return 0.0
    return max(0.0, min(1.0, x))

def parse_score(text: str) -> float:

    m = json_score_re.search(text)
    if m:
        try:
            return clamp01(float(m.group(1)))
        except:
            pass

    m2 = num_re.search(text)
    if m2:
        try:
            return clamp01(float(m2.group(1)))
        except:
            pass

    if "1" in text:
        return 1.0
    return 0.0

async def score_persona_async(pair, max_tries=3):
    system_prompt = "你是文本分類與評分助手，負責判斷角色回覆是否有展現人設風格。"
    user_prompt = build_prompt(pair["fan"], pair["reply"])
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    client = AsyncOpenAI(
        api_key=azure_api_key,
        base_url=azure_endpoint,
    )

    for _ in range(max_tries):
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=20,
                top_p=1.0,
            )
            output = response.choices[0].message.content.strip()
            score = parse_score(output)
            return score
        except Exception as e:
            print("Retrying due to error:", e)
            await asyncio.sleep(2)

    return 0.0

async def score_pair_with_limit(pair):
    async with semaphore:
        return await score_persona_async(pair)
    
async def process_persona_scores_async(pairs):
    scores = []
    pbar = tqdm_asyncio(total=len(pairs), desc="Scoring", unit="pair")

    async def score_and_update(pair):
        score = await score_pair_with_limit(pair)
        scores.append(score)
        pbar.update(1)
        return score
    tasks = [score_and_update(pair) for pair in pairs]
    scores = await asyncio.gather(*tasks)
    pbar.close()
    result = {"scores": scores}
    return result

flat_pairs = []
index_map = []  # (article_idx, pair_idx)
for a_idx, comment_thread in enumerate(data):
    for p_idx, pair in enumerate(comment_thread["pairs"]):
        fan_text = pair["Fans"]
        reply_text = pair[target]
        flat_pairs.append({"fan": fan_text, "reply": reply_text})
        index_map.append((a_idx, p_idx))



result = asyncio.run(process_persona_scores_async(flat_pairs))



for (a_idx, p_idx), lbl in zip(index_map, result["scores"]):
    data[a_idx]["pairs"][p_idx]["persona_score"] = lbl

# output
output_path = f"{output_folder_path}/test_set_comment_reply_pairs_labeled_{model_name}.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

# print(f"Saved labeled data to: {output_path}")
