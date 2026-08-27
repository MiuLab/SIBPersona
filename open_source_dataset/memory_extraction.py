import argparse
import json
from dotenv import load_dotenv, find_dotenv
from openai import AsyncOpenAI
import os
from prompt import get_extract_memory_prompt, get_extract_speaking_style_prompt, get_extract_SIB_triplet_prompt
import asyncio
from typing import Literal
from tqdm.asyncio import tqdm
from datetime import datetime
load_dotenv(find_dotenv())

parser = argparse.ArgumentParser("Extracting memory components from text data")
parser.add_argument("--model", type=str, default="gpt-4o-mini")
parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--person_name", type=str, required=True)
parser.add_argument("--dataset", type=str, required=True, help="Specify the target dataset", choices=["RoleAgentBench", "RoleBench", "CharacterEval"])
args = parser.parse_args()

model = args.model
person_name = args.person_name
batch_size = args.batch_size
dataset = args.dataset

semaphore = asyncio.Semaphore(batch_size)

azure_api_key = os.environ['AZURE_OPENAI_API_KEY']
azure_api_version=os.environ['AZURE_API_VERSION']
azure_endpoint = os.environ['AZURE_ENDPOINT']


recog_path = f"./data/{dataset}/character_data/{person_name}/recognized/recognized_dialogue.json"
character_path = f"./data/{dataset}/character_data/{person_name}"
with open(f"{character_path}/profile.json", "r", encoding="utf-8") as f:
    profiles = json.load(f)
profile = profiles["profile"]
lang = profiles["lang"]
TOTAL_INPUT_TOKEN = 0
TOTAL_OUTPUT_TOKEN = 0

def get_module_type_prompt(module_type: str, person_name: str, data: list | str, lang: str) -> str:
    if module_type == "Memory":
        return get_extract_memory_prompt(person_name, data, lang)
    elif module_type == "Speaking_Style":
        return get_extract_speaking_style_prompt(person_name, data, lang)
    elif module_type == "SIB":
        return get_extract_SIB_triplet_prompt(person_name, data, lang)
    else:
        raise ValueError(f"Unknown memory type: {module_type}")

def check_output_format(output: dict, module_type: str) -> bool:
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
            if "claim" not in item or "confidence" not in item:
                return False
            if not isinstance(item["claim"], str) or not isinstance(item["confidence"], float):
                return False
        elif module_type == "SIB":
            if "situation" not in item or "internal_state" not in item or "behavior" not in item or "confidence" not in item:
                return False
            if not isinstance(item["situation"], str) or not isinstance(item["internal_state"], str) or not isinstance(item["behavior"], str) or not isinstance(item["confidence"], float):
                return False
    return True

with open(recog_path, "r", encoding="utf-8") as f:
    datas = json.load(f)


async def extract_memory_component_async(data, module_type, lang, max_tries=5):
    global TOTAL_INPUT_TOKEN, TOTAL_OUTPUT_TOKEN
    data = data["raw_text"]

    system_prompt, user_prompt = get_module_type_prompt(module_type=module_type, person_name=person_name, data=data, lang=lang)
    messages_for_model = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ]
    # print(messages_for_model)
    for _ in range(max_tries):
        try:
            client = AsyncOpenAI(api_key=azure_api_key, base_url=azure_endpoint)
            response = await client.chat.completions.create(
                                model=model,
                                messages=messages_for_model,
                                temperature=0.2,
                                top_p=0.8,
                                frequency_penalty=0.0,
                                presence_penalty=0.0,
                            )
            output = response.choices[0].message.content
            output = output[7:-3].strip()
            if hasattr(response, 'usage') and response.usage:
                batch_input_tokens = response.usage.prompt_tokens
                batch_output_tokens = response.usage.completion_tokens
                
                TOTAL_INPUT_TOKEN += batch_input_tokens
                TOTAL_OUTPUT_TOKEN += batch_output_tokens
            try:
                output_dic = json.loads(output)
                # print(f"Output dic: {output_dic}")
                if not check_output_format(output_dic, module_type):
                    print(f"Invalid output format: {output}")
                    print(f"raw data: {data}")
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
            print(f"API error: {e}")
            print(f"the response: {response}")
            await asyncio.sleep(60)
            continue
    print(f"Invalid output format after {max_tries} try: {output}")
    return {
                "raw_text": data,
                "recognized_result": {},
                "module_type": module_type,
            }  # Return an empty dic if no valid output is found

async def extract_memory_component_with_limit(data, module_type, lang):
    async with semaphore:
        result = await extract_memory_component_async(data, module_type, lang)
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
            tasks.append(extract_memory_component_with_limit(item, cat, lang))

    if not tasks:
        print("No tasks to run. Check input data.")
        return [], [], []

    print(f"Processing {len(tasks)} extract tasks...")
    results = await tqdm.gather(*tasks, desc="Extracting")


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

output_path = f"./data/{dataset}/character_data/{person_name}/persona_module/{model}"
os.makedirs(output_path, exist_ok=True)

with open(f"{output_path}/memory.json", "w", encoding="utf-8") as f:
    json.dump(mem_bucket, f, ensure_ascii=False, indent=4)
with open(f"{output_path}/speaking_style.json", "w", encoding="utf-8") as f:
    json.dump(speak_bucket, f, ensure_ascii=False, indent=4)
with open(f"{output_path}/SIB.json", "w", encoding="utf-8") as f:
    json.dump(SIB_bucket, f, ensure_ascii=False, indent=4)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
try:
    with open(f"../token_report.json", "r", encoding="utf-8") as f:
        token_report = json.load(f)


    token_report[f"{dataset}_{person_name}_Open_memoryExtraction_{model}_{timestamp}"] = {
        "input_tokens": TOTAL_INPUT_TOKEN,
        "output_tokens": TOTAL_OUTPUT_TOKEN,
    }
except:
    token_report = {f"{dataset}_{person_name}_Open_memoryExtraction_{model}_{timestamp}": {
        "input_tokens": TOTAL_INPUT_TOKEN,
        "output_tokens": TOTAL_OUTPUT_TOKEN,
    }}
with open(f"../token_report.json", "w", encoding="utf-8") as f:
    json.dump(token_report, f, ensure_ascii=False, indent=4)