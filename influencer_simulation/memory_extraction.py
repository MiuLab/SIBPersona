import argparse
import json
from dotenv import load_dotenv, find_dotenv
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
import os
from prompt import get_extract_memory_prompt, get_extract_speaking_style_prompt, get_extract_SIB_triplet_prompt
import asyncio
from typing import Literal
from tqdm.asyncio import tqdm
import time
import random
from datetime import datetime, timedelta
from google import genai
from google.genai import types
import re
load_dotenv(find_dotenv())

parser = argparse.ArgumentParser("Extracting memory components from text data")
parser.add_argument("--model", type=str, default="gpt-4o-mini")
parser.add_argument("--recognizing_model", type=str, default="gpt-4o-mini")
parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--person_name", type=str, default="Influencer_A")
parser.add_argument("--file_name", type=str, default="recognized_data_example_for_persona_bank_construction")
args = parser.parse_args()

model = args.model
person_name = args.person_name
batch_size = args.batch_size
file_name = args.file_name
recognizing_model = args.recognizing_model

semaphore = asyncio.Semaphore(batch_size)

TOTAL_INPUT_TOKEN = 0
TOTAL_OUTPUT_TOKEN = 0
azure_api_key = os.environ.get('AZURE_OPENAI_API_KEY', '')
azure_api_version = os.environ.get('AZURE_API_VERSION', '')
azure_endpoint = os.environ.get('AZURE_ENDPOINT', '')
claude_api_key = os.environ.get('ANTHROPIC_API_KEY', '')
gemini_key = os.environ.get('GEMINI_API_KEY', '')

# Initialize Gemini client if needed
gemini_client = None
if model.startswith("gemini"):
    gemini_client = genai.Client(api_key=gemini_key)

path = f"./data/{person_name}/recognized/{file_name}_{recognizing_model}.json"

def get_module_type_prompt(module_type: str, person_name: str, data: list | str) -> str:
    if module_type == "Memory":
        return get_extract_memory_prompt(person_name, data)
    elif module_type == "Speaking_Style":
        return get_extract_speaking_style_prompt(person_name, data)
    elif module_type == "SIB":
        return get_extract_SIB_triplet_prompt(person_name, data)
    else:
        raise ValueError(f"Unknown memory type: {module_type}")

def check_output_format(output: dict, module_type: str, pairs: list) -> bool:
    if "items" not in output:
        return False
    if not isinstance(output["items"], list):
        return False
    if len(output["items"]) == 0:
        return True  # Allow empty items
    for idx, item in enumerate(output["items"]):
        if not isinstance(item, dict):
            return False
        if module_type == "Memory" or module_type == "Speaking_Style":
            if "evidence_pair_id" not in item or "claim" not in item or "confidence" not in item:
                return False
            if not isinstance(item["evidence_pair_id"], list) or not isinstance(item["claim"], str) or not isinstance(item["confidence"], float):
                return False
            id_in_pairs = True
            for id in item["evidence_pair_id"]:
                if id not in [pair.get("pair_id") for pair in pairs]:
                    id_in_pairs = False
                    break
            if not id_in_pairs:
                return False
        elif module_type == "SIB":
            if "situation" not in item or "internal_state" not in item or "behavior" not in item or "evidence_pair_id" not in item or "confidence" not in item:
                return False
            if not isinstance(item["situation"], str) or not isinstance(item["internal_state"], str) or not isinstance(item["behavior"], str) or not isinstance(item["evidence_pair_id"], list) or not isinstance(item["confidence"], float):
                return False
            id_in_pairs = True
            for id in item["evidence_pair_id"]:
                if id not in [pair.get("pair_id") for pair in pairs]:
                    id_in_pairs = False
                    break
            if not id_in_pairs:
                return False
    return True

with open(path, "r", encoding="utf-8") as f:
    datas = json.load(f)


async def extract_memory_component_async(data, module_type, max_tries=5):
    global TOTAL_INPUT_TOKEN, TOTAL_OUTPUT_TOKEN
    await asyncio.sleep(random.uniform(0.1, 0.3))
    data = data["raw_text"]

    system_prompt = "你是一個專門萃取文字中包含特定對象資訊的模型"
    user_prompt = get_module_type_prompt(module_type=module_type, person_name=person_name, data=data)
    messages_for_model = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ]
    # print(messages_for_model)
    for _ in range(max_tries):
        try:
            if model.startswith("gemini"):
                # Gemini inference
                gemini_contents = [
                    types.Content(
                        role="user",
                        parts=[types.Part(text=user_prompt)]
                    )
                ]
                if model == "gemini-3-pro-preview":
                    thinking_level = "low"
                elif model == "gemini-3-flash-preview":
                    thinking_level = "minimal"
                generate_config = types.GenerateContentConfig(
                    temperature=0.2,
                    thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
                    system_instruction=system_prompt
                )
                
                response = await gemini_client.aio.models.generate_content(
                    model=model,
                    contents=gemini_contents,
                    config=generate_config,
                )
                
                # Extract tokens from Gemini response
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    batch_input_tokens = response.usage_metadata.prompt_token_count or 0
                    batch_output_tokens = response.usage_metadata.candidates_token_count or 0
                    if hasattr(response.usage_metadata, 'thoughts_token_count'):
                        batch_output_tokens += response.usage_metadata.thoughts_token_count or 0
                    TOTAL_INPUT_TOKEN += batch_input_tokens
                    TOTAL_OUTPUT_TOKEN += batch_output_tokens
                
                # Extract text from Gemini response
                output = response.text
                if not output:
                    print(f"Attempt {_ + 1}: No valid Gemini response content")
                    continue
            elif model.startswith("claude"):
                # Claude inference
                client = AsyncAnthropic(api_key=claude_api_key)
                system_prompt_for_claude = messages_for_model[0]['content']
                messages_for_model = [msg for msg in messages_for_model if msg['role'] != 'system']
                response = await client.messages.create(
                    model=model,
                    system=system_prompt_for_claude,
                    messages=messages_for_model,
                    thinking={
                        "type": "adaptive"
                    },
                    output_config={
                        "effort": "low"
                    },
                    max_tokens=4096,
                )
                if hasattr(response, 'usage') and response.usage:
                    batch_input_tokens = response.usage.input_tokens
                    batch_output_tokens = response.usage.output_tokens

                    TOTAL_INPUT_TOKEN += batch_input_tokens
                    TOTAL_OUTPUT_TOKEN += batch_output_tokens
                output = "".join(
                    block.text for block in response.content
                    if block.type == "text" and block.text.strip()
                )
            else:
                # Azure OpenAI inference
                client = AsyncOpenAI(api_key=azure_api_key, base_url=azure_endpoint)
                if "deepseek" in model.lower():
                    response = await client.chat.completions.create(
                                    model=model,
                                    messages=messages_for_model,
                                )
                elif "gpt-5" in model:
                    response = await client.chat.completions.create(
                                    model=model,
                                    messages=messages_for_model,
                                    reasoning_effort = "low"
                                )
                else:
                    response = await client.chat.completions.create(
                                        model=model,
                                        messages=messages_for_model,
                                        temperature=0.2,
                                        top_p=0.8,
                                        frequency_penalty=0.0,
                                        presence_penalty=0.0,
                                    )
                if hasattr(response, 'usage') and response.usage:
                    batch_input_tokens = response.usage.prompt_tokens
                    batch_output_tokens = response.usage.completion_tokens
                    
                    TOTAL_INPUT_TOKEN += batch_input_tokens
                    TOTAL_OUTPUT_TOKEN += batch_output_tokens
                output = response.choices[0].message.content
            
            output = output.strip()
            output = output.replace("```json", "")
            output = output.replace("```", "")
            output = output.replace("\n", "")
            pattern = re.compile(r"\{[\s\S]*\}$")
            match = pattern.search(output)
            output = match.group() if match else output
            try:
                output_dic = json.loads(output)
                if not check_output_format(output_dic, module_type, data["pairs"]):
                    print(f"Invalid output format: {output}")
                    # print(f"raw data: {data}")
                    continue
                output = {
                    "raw_text": data,
                    "recognized_result": output_dic,
                    "module_type": module_type,
                }
                return output
            except:
                print(f"Failed to decode JSON from output: {output}")
                continue
        except Exception as e:
            # print(f"❌ API error (attempt {_ + 1}): {e}")
            if "rate" in str(e).lower() or "429" in str(e):
                wait_time = (_ + 1) * 30  # 30s, 60s, 90s, ...
                print(f"⏱️ Rate limit detected, waiting {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                await asyncio.sleep(random.uniform(1, 3))
            continue
    print(f"Invalid output format after {max_tries}")
    return {
                "raw_text": data,
                "recognized_result": {},
                "module_type": module_type,
            }  # Return an empty dic if no valid output is found

async def extract_memory_component_with_limit(data, module_type):
    async with semaphore:
        result = await extract_memory_component_async(data, module_type)
        return result
    
async def run_all():
    tasks = []
    for item in datas:
        cats = (item.get("recognized_result", {}) or {}).get("categorys", [])

        if not cats:
            cats = ["Memory", "Speaking_Style", "SIB"]
        for cat in cats:
            if cat not in {"Memory", "Speaking_Style", "SIB"}:
                continue
            tasks.append(extract_memory_component_with_limit(item, cat))

    if not tasks:
        print("No tasks to run. Check input data.")
        return [], [], []

    print(f"Processing {len(tasks)} extract tasks...")
    results = await tqdm.gather(*tasks, desc="Extracting")

    # 依類別分桶
    mem_bucket, speak_bucket, SIB_bucket = [], [], []
    for r in results:
        if r.get("module_type") == "Memory":
            mem_bucket.append(r)
        elif r.get("module_type") == "Speaking_Style":
            speak_bucket.append(r)
        elif r.get("module_type") == "SIB":
            SIB_bucket.append(r)

    return mem_bucket, speak_bucket, SIB_bucket

mem_bucket, speak_bucket, SIB_bucket = asyncio.run(run_all())

output_path = f"./data/{person_name}/persona_module/{model}"
os.makedirs(output_path, exist_ok=True)

with open(f"{output_path}/memory.json", "w", encoding="utf-8") as f:
    json.dump(mem_bucket, f, ensure_ascii=False, indent=4)
with open(f"{output_path}/speaking_style.json", "w", encoding="utf-8") as f:
    json.dump(speak_bucket, f, ensure_ascii=False, indent=4)
with open(f"{output_path}/SIB.json", "w", encoding="utf-8") as f:
    json.dump(SIB_bucket, f, ensure_ascii=False, indent=4)


timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
try:
    with open(f"./token_report.json", "r", encoding="utf-8") as f:
        token_report = json.load(f)


    token_report[f"{person_name}_{file_name}_memoryExtraction_{model}_{timestamp}"] = {
        "input_tokens": TOTAL_INPUT_TOKEN,
        "output_tokens": TOTAL_OUTPUT_TOKEN,
    }
except:
    token_report = {f"{person_name}_{file_name}_memoryExtraction_{model}_{timestamp}": {
        "input_tokens": TOTAL_INPUT_TOKEN,
        "output_tokens": TOTAL_OUTPUT_TOKEN,
    }}
with open(f"./token_report.json", "w", encoding="utf-8") as f:
    json.dump(token_report, f, ensure_ascii=False, indent=4)
