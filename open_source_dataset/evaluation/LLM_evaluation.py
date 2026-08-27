import argparse
import json
from dotenv import load_dotenv, find_dotenv
from google import genai
from openai import OpenAI, AsyncOpenAI
import os
from prompt import get_batch_penalty_judge_prompt
import asyncio
from typing import Literal, Dict, Any, List
from tqdm.asyncio import tqdm
import numpy as np
from collections import defaultdict
import subprocess
import sys
from aiolimiter import AsyncLimiter
import google.auth
import langchain_google_vertexai
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
load_dotenv(find_dotenv())

parser = argparse.ArgumentParser("LLM Evaluation for Open Source Dataset")
parser.add_argument("--model", type=str, default="gpt-4o-mini")
parser.add_argument("--batch_size", type=int, default=3)
parser.add_argument("--eval_batch_size", type=int, default=1, help="Number of samples to evaluate together")
parser.add_argument("--task", required=True, type=str, choices=["summary", "general_response"])
parser.add_argument("--language", required=True, type=str, choices=["zh", "eng"])
parser.add_argument("--file_name", type=str, required=True)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--dimension", type=str, required=True, choices=["speaking_style", "memory", "SIB"])
parser.add_argument("--run_time", type=int, default=0, help="To distinguish different runs")
args = parser.parse_args()

model = args.model
batch_size = args.batch_size
eval_batch_size = args.eval_batch_size
dimension = args.dimension
file_name = args.file_name
dataset = args.dataset
language = args.language
task = args.task
run_time = args.run_time
semaphore = asyncio.Semaphore(batch_size)
limiter = AsyncLimiter(max_rate=50, time_period=60)

TOTAL_INPUT_TOKEN = 0
TOTAL_OUTPUT_TOKEN = 0

run_time_str = ""
if run_time > 0:
    run_time_str = f"_run{run_time}"


azure_api_key = os.environ['AZURE_OPENAI_API_KEY']
azure_api_version=os.environ['AZURE_API_VERSION']
azure_endpoint = os.environ['AZURE_ENDPOINT']
gemini_key = os.environ.get('GEMINI_API_KEY', '')

# load data
datas_folder = f"../data/{dataset}/results/{task}/{language}"
datas_path = f"{datas_folder}/{file_name}.json"
out_folder = f"./data/{dataset}/{task}/{language}/LLM_eval_results/{file_name}_eval={model}{run_time_str}"

with open(datas_path, "r", encoding="utf-8") as f:
    datas = json.load(f)

profile_cache: Dict[str, str] = {}
persona_cache: Dict[str, Dict[str, Any]] = {}

def load_person_data(person_name: str) -> tuple[str, Dict[str, Any]]:

    if person_name in profile_cache and person_name in persona_cache:
        return profile_cache[person_name], persona_cache[person_name]
    
    try:
        # load profile
        profile_path = f"../data/{dataset}/character_data/{person_name}/profile.json"
        with open(profile_path, "r", encoding="utf-8") as f:
            profile_data = json.load(f)
        profile = profile_data["profile"]
        if not os.path.exists(f"../data/{dataset}/character_data/{person_name}/representative_persona/"):
            cmd = [
                        sys.executable,
                        "filter_representative_persona.py",
                        "--person_name", person_name,
                        "--dataset", dataset,
                    ]
            print(f"Finding representative persona for {person_name}")
            subprocess.run(cmd)
        # loading representative persona
        persona_rep_path = f"../data/{dataset}/character_data/{person_name}/representative_persona/{dimension}_representative.json"
        with open(persona_rep_path, "r", encoding="utf-8") as f:
            persona_rep = json.load(f)
        
        profile_cache[person_name] = profile
        persona_cache[person_name] = persona_rep
        
        return profile, persona_rep
        
    except FileNotFoundError as e:
        print(f"Warning: Could not load data for {person_name}: {e}")
        return "", {}


def check_output_format(output: dict) -> bool:
    if "flaws" not in output:
        return False
    if not isinstance(output["flaws"], list):
        return False
    if len(output["flaws"]) == 0:
        return True  # Allow empty items
    for idx, item in enumerate(output["flaws"]):
        if not isinstance(item, dict):
            return False
        if "instance" not in item or "type" not in item or "severity" not in item or "sample_index" not in item:
            return False
        if not isinstance(item["instance"], str) or not isinstance(item["type"], str) or not isinstance(item["severity"], int) or not isinstance(item["sample_index"], list):
            return False
    return True


async def multiple_evaluation_async(batch_data: List[Dict[str, Any]], batch_idx: int, max_tries=5):
    global TOTAL_INPUT_TOKEN, TOTAL_OUTPUT_TOKEN
    if not batch_data:
        return []
    

    person_name = batch_data[0]["person_name"]
    profile, persona_rep = load_person_data(person_name)
    
    if not profile or not persona_rep:
        print(f"Skipping batch for {person_name} due to missing data")
        return []
    
    eval_data = []
    for data in batch_data:
        eval_item = {
            "question": data["question"],
            "model_reply": data["model_reply"],
            "ground_truth": data["ground_truth"]
        }
        eval_data.append(eval_item)
    
    prompts = get_batch_penalty_judge_prompt(person_name, profile, persona_rep, eval_data, dimension, len(eval_data), language)
    system_prompt = prompts["system"]
    user_prompt = prompts["user"]
    messages_for_model = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt}
    ]
    # print(user_prompt)
    for attempt in range(max_tries):
        try:
            async with limiter:
                if model.startswith("gpt-4") or model.startswith("gpt-5"):
                    client = AsyncOpenAI(api_key=azure_api_key, base_url=azure_endpoint)
                    if "gpt-5" in model:
                        response = await client.chat.completions.create(
                            model=model,
                            messages=messages_for_model,
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
                    output = response.choices[0].message.content
                    if hasattr(response, 'usage') and response.usage:
                        batch_input_tokens = response.usage.prompt_tokens
                        batch_output_tokens = response.usage.completion_tokens
                        
                        TOTAL_INPUT_TOKEN += batch_input_tokens
                        TOTAL_OUTPUT_TOKEN += batch_output_tokens
                elif model.startswith("gemini"):
                    client = genai.Client(api_key=gemini_key)
                    
                    # Combine system and user prompts for Gemini
                    gemini_contents = f"{system_prompt}\n\n{user_prompt}"
                    response = await client.aio.models.generate_content(
                        model=model,
                        contents=gemini_contents,
                    )
                    output = response.text
                    if response.usage_metadata:
                        batch_input_tokens = response.usage_metadata.prompt_token_count or 0
                        batch_output_tokens = response.usage_metadata.candidates_token_count or 0
                        batch_output_tokens += response.usage_metadata.thoughts_token_count or 0

                        TOTAL_INPUT_TOKEN += batch_input_tokens
                        TOTAL_OUTPUT_TOKEN += batch_output_tokens
                else:
                    raise ValueError(f"Unsupported model type: {model}")
                
                # print(output)
                try:
                    output = output.replace("```json", "")
                    output = output.replace("```", "")
                    output_dic = json.loads(output)
                    
                    if not check_output_format(output_dic):
                        print(f"Invalid output format for {person_name}, attempt {attempt + 1}")
                        continue
                    

                    outputs = {
                        "data": batch_data,
                        "dimension": dimension,
                        "flaws": output_dic["flaws"],
                    }
                    
                    return outputs
                
                except Exception as e:
                    print(f"Failed to decode JSON from output for {person_name}: {e}")
                    continue
                
        except Exception as e:
            print(f"Error during API call for {person_name}: {e}")
            await asyncio.sleep(11 * (attempt + 1))
            continue
    
    print(f"Failed to get valid output for {person_name} after {max_tries} tries")

    result = {
        "data": data,
        "person_name": data["person_name"],
        "dimension": dimension,
        "flaws": [],
    }
    return result

async def eval_batch_with_limit(batch_data: List[Dict[str, Any]], batch_idx: int):
    async with semaphore:
        result = await multiple_evaluation_async(batch_data, batch_idx)
        return result

def create_person_batches(data_list: List[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
    person_groups = defaultdict(list)
    for data in data_list:
        person_name = data["person_name"]
        person_groups[person_name].append(data)
    

    all_batches = []
    for person_name, person_data in person_groups.items():
        for i in range(0, len(person_data), batch_size):
            batch = person_data[i:i + batch_size]
            all_batches.append(batch)
    
    return all_batches

async def run_all(datas):
    datas = datas
    batches = create_person_batches(datas, eval_batch_size)
    tasks = []
    
    for batch_idx, batch_data in enumerate(batches):
        tasks.append(eval_batch_with_limit(batch_data, batch_idx))

    if not tasks:
        print("No tasks to run. Check input data.")
        return []
    
    print(f"Processing {len(batches)} batches with up to {eval_batch_size} samples each...")
    batch_results = await tqdm.gather(*tasks, desc="Evaluating batches")
    
    all_results = []
    for batch_result in batch_results:
        if batch_result:
            all_results.append(batch_result)
    
    return all_results


eval_results = asyncio.run(run_all(datas))
# print(eval_results)
def cal_score(flaws):
    """calculate score based on flaws"""
    minus = 0
    for flaw in flaws:
        minus += 5 * flaw["severity"]
    score = 100 - minus
    if score < 0:
        score = 0
    return score

scores = []
for i, result in enumerate(eval_results):
    score = cal_score(result["flaws"])
    eval_results[i]["score"] = score
    scores.append(score)

avg_score = np.mean(scores) if scores else 0


final_results = {
    "overall_avg_score": avg_score,
    "total_samples": len(eval_results),
    "eval_results": eval_results
}


os.makedirs(out_folder, exist_ok=True)
output_path = f"{out_folder}/{dimension}.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(final_results, f, ensure_ascii=False, indent=4)


base_folder = out_folder.rsplit("/", 1)[0]
sub_folder = out_folder.rsplit("/", 1)[1]

try:
    with open(f"{base_folder}/score_report.json", "r", encoding="utf-8") as f:
        score_report = json.load(f)
    
    if sub_folder not in score_report:
        score_report[sub_folder] = {}
    score_report[sub_folder][dimension] = avg_score
    
    if len(score_report[sub_folder]) > 1:
        dimension_scores = [v for k, v in score_report[sub_folder].items() if k != "avg"]
        score_report[sub_folder]["avg"] = np.mean(dimension_scores)
    else:
        score_report[sub_folder]["avg"] = avg_score
        
    with open(f"{base_folder}/score_report.json", "w", encoding="utf-8") as f:
        json.dump(score_report, f, ensure_ascii=False, indent=4)
        
except (FileNotFoundError, json.JSONDecodeError):
    score_report = {sub_folder: {dimension: avg_score, "avg": avg_score}}
    with open(f"{base_folder}/score_report.json", "w", encoding="utf-8") as f:
        json.dump(score_report, f, ensure_ascii=False, indent=4)


timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
try:
    with open(f"../../token_report.json", "r", encoding="utf-8") as f:
        token_report = json.load(f)


    token_report[f"{file_name}_Open_LLMEvaluation_{dimension}{run_time_str}_{model}_{timestamp}"] = {
        "input_tokens": TOTAL_INPUT_TOKEN,
        "output_tokens": TOTAL_OUTPUT_TOKEN,
    }
except:
    token_report = {f"{file_name}_Open_LLMEvaluation_{dimension}{run_time_str}_{model}_{timestamp}": {
        "input_tokens": TOTAL_INPUT_TOKEN,
        "output_tokens": TOTAL_OUTPUT_TOKEN,
    }}
with open(f"../../token_report.json", "w", encoding="utf-8") as f:
    json.dump(token_report, f, ensure_ascii=False, indent=4)
