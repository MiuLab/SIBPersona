import argparse
import json
from dotenv import load_dotenv, find_dotenv
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
import os
from prompt import get_recognize_module_prompt
import ast
import asyncio
from tqdm.asyncio import tqdm
import time
import random
from datetime import datetime, timedelta
from google import genai
from google.genai import types
import re
load_dotenv(find_dotenv())

parser = argparse.ArgumentParser("Recongnizing memory components from text data")
parser.add_argument("--model", type=str, default="gpt-4o-mini")
parser.add_argument("--batch_size", type=int, default=20)
parser.add_argument("--file_name", type=str, default="data_example_for_persona_bank_construction")
parser.add_argument("--person_name", type=str, required=True, help="Specify the target person name")
args = parser.parse_args()

model = args.model
person_name = args.person_name
batch_size = args.batch_size
file_name = args.file_name

TOTAL_INPUT_TOKEN = 0
TOTAL_OUTPUT_TOKEN = 0
semaphore = asyncio.Semaphore(batch_size)

azure_api_key = os.environ.get('AZURE_OPENAI_API_KEY', '')
azure_api_version = os.environ.get('AZURE_API_VERSION', '')
azure_endpoint = os.environ.get('AZURE_ENDPOINT', '')
claude_api_key = os.environ.get('ANTHROPIC_API_KEY', '')
gemini_key = os.environ.get('GEMINI_API_KEY', '')

# Initialize Gemini client if needed
gemini_client = None
if model.startswith("gemini"):
    gemini_client = genai.Client(api_key=gemini_key)


path = f"./data/{person_name}/{file_name}.json"

with open(path, "r", encoding="utf-8") as f:
    datas = json.load(f)


def check_output_format(output):
    for component in output:
        if component not in ["SIB", "Speaking_Style", "Memory"]:
            return False
    return True


async def recognize_memory_component_async(data, max_tries=5):
    global TOTAL_INPUT_TOKEN, TOTAL_OUTPUT_TOKEN
    await asyncio.sleep(random.uniform(0.1, 0.3))
    system_prompt = "你是一個專門辨識文字中所含資訊的模型"
    user_prompt = get_recognize_module_prompt(person_name=person_name, data=data)
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
                    temperature=1.0,
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
                                        temperature=1.0,
                                        top_p=1.0,
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
            # print("Raw output:", output)
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
            if "rate" in str(e).lower() or "429" in str(e):
                wait_time = (_ + 1) * 30  # 30s, 60s, 90s, ...
                print(f"⏱️ Rate limit detected, waiting {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                print(f"⏱️ Error detected, waiting a bit before retrying... (attempt {_ + 1}): {e}")
                await asyncio.sleep(random.uniform(1, 3))
            continue
    print(f"Invalid output format after {max_tries} try")
    return {}  # Return an empty dic if no valid output is found

async def recognize_memory_component_with_limit(data, idx):
    async with semaphore:
        result = await recognize_memory_component_async(data)
        return result
    
async def process_reconizing_in_parallel(data_for_memories):
    tasks = [recognize_memory_component_with_limit(conv, idx) for idx, conv in enumerate(data_for_memories)]
    
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

recognized_component = asyncio.run(process_reconizing_in_parallel(datas))

os.makedirs(f"./data/{person_name}/recognized", exist_ok=True)
output_path = f"./data/{person_name}/recognized/recognized_{file_name}_{model}.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(recognized_component, f, ensure_ascii=False, indent=4)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
try:
    with open(f"./token_report.json", "r", encoding="utf-8") as f:
        token_report = json.load(f)


    token_report[f"{person_name}_memoryRecog_{model}_{timestamp}"] = {
        "input_tokens": TOTAL_INPUT_TOKEN,
        "output_tokens": TOTAL_OUTPUT_TOKEN,
    }
except:
    token_report = {f"{person_name}_memoryRecog_{model}_{timestamp}": {
        "input_tokens": TOTAL_INPUT_TOKEN,
        "output_tokens": TOTAL_OUTPUT_TOKEN,
    }}
with open(f"./token_report.json", "w", encoding="utf-8") as f:
    json.dump(token_report, f, ensure_ascii=False, indent=4)
