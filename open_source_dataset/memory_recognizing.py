import argparse
import json
from dotenv import load_dotenv, find_dotenv
from openai import AsyncOpenAI
import os
from prompt import get_recognize_module_prompt
import ast
import asyncio
from tqdm.asyncio import tqdm
from datetime import datetime
load_dotenv(find_dotenv())

parser = argparse.ArgumentParser("Recongnizing memory components from text data")
parser.add_argument("--model", type=str, default="gpt-4o-mini")
parser.add_argument("--batch_size", type=int, default=20)
parser.add_argument("--person_name", type=str, required=True, help="Specify the target person name")
parser.add_argument("--dataset", type=str, required=True, help="Specify the target dataset", choices=["RoleAgentBench", "RoleBench", "CharacterEval"])
args = parser.parse_args()

model = args.model
person_name = args.person_name
dataset = args.dataset
batch_size = args.batch_size

TOTAL_INPUT_TOKEN = 0
TOTAL_OUTPUT_TOKEN = 0

semaphore = asyncio.Semaphore(batch_size)



azure_api_key = os.environ['AZURE_OPENAI_API_KEY']
azure_api_version=os.environ['AZURE_API_VERSION']
azure_endpoint = os.environ['AZURE_ENDPOINT']


character_path = f"./data/{dataset}/character_data/{person_name}"

with open(f"{character_path}/dialogue.json", "r", encoding="utf-8") as f:
    datas = json.load(f)

with open(f"{character_path}/profile.json", "r", encoding="utf-8") as f:
    profiles = json.load(f)
profile = profiles["profile"]
lang = profiles["lang"]


def check_output_format(output):
    for component in output:
        if component not in ["SIB", "Speaking_Style", "Memory"]:
            return False
    return True


async def recognize_memory_component_async(data, lang, max_tries=5):
    global TOTAL_INPUT_TOKEN, TOTAL_OUTPUT_TOKEN
    system_prompt, user_prompt = get_recognize_module_prompt(person_name=person_name, data=data, lang=lang)
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
                                temperature=1.0,
                                top_p=1.0,
                                frequency_penalty=0.0,
                                presence_penalty=0.0,
                            )
            output = response.choices[0].message.content
            output = response.choices[0].message.content
            output = output[7:-3].strip()
            if hasattr(response, 'usage') and response.usage:
                batch_input_tokens = response.usage.prompt_tokens
                batch_output_tokens = response.usage.completion_tokens
                
                TOTAL_INPUT_TOKEN += batch_input_tokens
                TOTAL_OUTPUT_TOKEN += batch_output_tokens
            try:
                output_dic = json.loads(output)
                if not check_output_format(output_dic["categorys"]):
                    print(f"Invalid output format: {output}")
                    continue
                return output_dic
            except:
                print(f"Failed to decode JSON from output: {output}")
                continue
        except Exception as e:
            print(f"API error: {e}")
            print(f"the response: {response}")
            await asyncio.sleep(60)
            continue
    print(f"Invalid output format after {max_tries} try: {output}")
    return {}  # Return an empty dic if no valid output is found

async def recognize_memory_component_with_limit(data, idx, lang):
    async with semaphore:
        result = await recognize_memory_component_async(data, lang)
        return result
    
async def process_reconizing_in_parallel(data_for_memories, lang):
    tasks = [recognize_memory_component_with_limit(conv, idx, lang) for idx, conv in enumerate(data_for_memories)]
    
    # Use tqdm.asyncio.gather for progress tracking
    print(f"Processing {len(data_for_memories)} memory components...")
    all_recognized_component = await tqdm.gather(*tasks, desc="Recognizing memories")
    
    # print(all_recognized_component)
    recognized_components = []
    for idx, recog_result in enumerate(all_recognized_component):
        if recog_result == {}:
            continue
        single_result = {
            "raw_text": data_for_memories[idx],
            "recognized_result": recog_result
        }
        recognized_components.append(single_result)
    
    print(f"Successfully processed {len(recognized_components)} out of {len(data_for_memories)} components")
    return recognized_components

print(f"Starting memory recognition for {len(datas)} items...")
recognized_component = asyncio.run(process_reconizing_in_parallel(datas, lang))

os.makedirs(f"./data/{dataset}/character_data/{person_name}/recognized", exist_ok=True)
output_path = f"./data/{dataset}/character_data/{person_name}/recognized/recognized_dialogue.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(recognized_component, f, ensure_ascii=False, indent=4)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
try:
    with open(f"../token_report.json", "r", encoding="utf-8") as f:
        token_report = json.load(f)


    token_report[f"{dataset}_{person_name}_Open_memoryRecog_{model}_{timestamp}"] = {
        "input_tokens": TOTAL_INPUT_TOKEN,
        "output_tokens": TOTAL_OUTPUT_TOKEN,
    }
except:
    token_report = {f"{dataset}_{person_name}_Open_memoryRecog_{model}_{timestamp}": {
        "input_tokens": TOTAL_INPUT_TOKEN,
        "output_tokens": TOTAL_OUTPUT_TOKEN,
    }}
with open(f"../token_report.json", "w", encoding="utf-8") as f:
    json.dump(token_report, f, ensure_ascii=False, indent=4)