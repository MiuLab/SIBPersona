import argparse
import json
import pandas as pd
from agent import retrieve_bundle, build_messages, generate_reply, format_persona_for_prompt
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
import transformers
import subprocess
import sys
import gc
import psutil
from datetime import datetime
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, type=str, choices=["RoleAgentBench", "CharacterEval"])
parser.add_argument("--task", required=True, type=str, choices=["summary","general_response", "charactereval"])
parser.add_argument("--language", required=True, type=str, choices=["zh", "eng"])
parser.add_argument("--k_persona", type=int, default=3)
parser.add_argument("--k_diags", type=int, default=5)
parser.add_argument("--emb_model", default="text-embedding-3-small")
parser.add_argument("--extraction_model", default="gpt-4o-mini")
parser.add_argument("--model", default="gpt-4o-mini", 
                   choices=[
                       "allenai/Olmo-3-7B-Instruct",
                       "Neph0s/CoSER-Llama-3.1-8B"
                   ],
                   help="Model to use for generation")
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--max_tokens", type=int, default=180)
parser.add_argument("--add_persona", action="store_true")
parser.add_argument("--add_profile", action="store_true")
parser.add_argument("--anonymous", action="store_true", help="Replace person_name with <anonymous_character> in prompts")
parser.add_argument("--delete_SIB", action="store_true", help="not use SIB_triplet as part of the retrieved bundle, even if it exists")
parser.add_argument("--delete_speaking_style", action="store_true", help="not use Speaking Style as part of the retrieved bundle, even if it exists")
parser.add_argument("--save_batch_size", type=int, default=300, help="Batch size for saving results to prevent OOM")
args = parser.parse_args()

def check_memory_usage(stage_name=""):
    """check current RAM and GPU memory usage and print it out"""
    process = psutil.Process()
    
    ram_gb = process.memory_info().rss / 1024**3
    
    system_ram = psutil.virtual_memory()
    system_ram_gb = system_ram.used / 1024**3
    system_total_gb = system_ram.total / 1024**3
    
    gpu_memory = 0
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3
    
    print(f"📊 {stage_name}")
    print(f"   Process RAM: {ram_gb:.2f}GB")
    print(f"   System RAM: {system_ram_gb:.1f}/{system_total_gb:.1f}GB ({system_ram.percent:.1f}%)")
    if gpu_memory > 0:
        print(f"   GPU Memory: {gpu_memory:.2f}GB")
    print()


TOTAL_INPUT_TOKEN = 0
TOTAL_OUTPUT_TOKEN = 0
extraction_model = args.extraction_model
dataset = args.dataset
language = args.language
task = args.task
if dataset == "RoleAgentBench":
    script_to_lang = {"Friends S1E1": "eng", "Harry Potter": "eng", "Merchant of Venice": "eng", "Sherlock - A Study in Pink": "eng",
                  "The Big Bang Theory S1E1": "eng", "九品芝麻官": "zh", "唐人街探案": "zh", "家有儿女 S1E1": "zh", "狂飙 S1E1": "zh", "西游记-三打白骨精": "zh"}
    data_path = f"./data/RoleAgentBench"
    with open(f"{data_path}/info.json", "r") as f:
        script_info = json.load(f)
    script = [key for key, val in script_to_lang.items() if val == language]
    character_list = []
    for sc in script:
        character_list.extend(script_info["scripts"][sc]["roles"])
    test_data = []
    for sc in script:
        with open(f"{data_path}/{sc}/{task}.json", "r") as f:
            script_test_data = json.load(f)
        test_data.extend(script_test_data)
elif dataset == "CharacterEval":
    data_path = f"./data/CharacterEval"
    with open(f"{data_path}/data/test_data.jsonl", "r") as f:
        test_data = json.load(f)
    
only_dialogue = False


output_folder = f"./data/{dataset}/results/{task}/{language}"

profile_str = ""
if args.add_profile:
    profile_str = "_Profile"
persona_file_str = ""
anonymous_str = ""
if args.add_persona:
    persona_file_str = f"_Persona_k={args.k_persona}"
if args.anonymous:
    anonymous_str = "_Anonymous"
response_outputs = []
batch_size = args.save_batch_size
if args.delete_SIB:
    persona_file_str += "_noSIB"
if args.delete_speaking_style:
    persona_file_str += "_noSpeakingStyle"
current_batch = 0

model_instance = None
tokenizer = None
if "Olmo" in args.model or "CoSER" in args.model:
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model_instance = AutoModelForCausalLM.from_pretrained(
            args.model,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"✓ Set pad_token to eos_token: {tokenizer.eos_token}")
    if model_instance.config.pad_token_id is None:
        model_instance.config.pad_token_id = tokenizer.eos_token_id
        print(f"✓ Set model pad_token_id to: {tokenizer.eos_token_id}")


def optimize_memory_settings():
    if torch.cuda.is_available():

        torch.cuda.empty_cache()

        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        print(f"✓ GPU memory optimized. Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    

    gc.set_threshold(100, 5, 5)
    print(f"✓ Memory optimization settings applied")


optimize_memory_settings()


def save_batch_and_clear_memory(response_outputs, output_folder, file_name, batch_num):
    os.makedirs(output_folder, exist_ok=True)
    batch_file = f"{output_folder}/{file_name}_batch_{batch_num}.json"
    
    with open(batch_file, 'w', encoding='utf-8') as f:
        json.dump(response_outputs, f, ensure_ascii=False, indent=4)
    
    print(f"✓ Batch {batch_num} saved to {batch_file} ({len(response_outputs)} items)")
    

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        current_memory = torch.cuda.memory_allocated() / 1e9
        print(f"  GPU memory after cleanup: {current_memory:.2f}GB")
    gc.collect()
    
    return []

def merge_batch_files(output_folder, file_name, total_batches):
    all_responses = []
    
    for batch_num in range(total_batches):
        batch_file = f"{output_folder}/{file_name}_batch_{batch_num}.json"
        if os.path.exists(batch_file):
            with open(batch_file, 'r', encoding='utf-8') as f:
                batch_data = json.load(f)
                all_responses.extend(batch_data)
            
            os.remove(batch_file)
            print(f"✓ Merged batch {batch_num} ({len(batch_data)} items)")
        else:
            print(f"⚠ Batch file {batch_file} not found")
    
    return all_responses

os.makedirs(output_folder, exist_ok=True)
model_str = args.model.replace("/", "-")
file_name = f"extraction={extraction_model}_generate={model_str}{persona_file_str}{profile_str}{anonymous_str}"


for i, data in enumerate(tqdm(test_data)):
    k_memory = args.k_persona
    k_speaking_style = args.k_persona if not args.delete_speaking_style else 0
    k_SIB = args.k_persona if not args.delete_SIB else 0
    if dataset == "RoleAgentBench":
        if task == "self_knowledge":
            source_role = ""
        else:
            source_role = data["source_role"]
        person_name = data["target_role"]
        if task in ["general_response", "summary"]:
            question = data["question"]

        if task in ["general_response"]:
            ground_truth = data["answer"]
        elif task == "summary":
            ground_truth = data["summary"]
            entities = data["entities"]

        index_fold = f"./data/{dataset}/character_data/{person_name}/persona_module/{extraction_model}"
        diags_index = f"{index_fold}/dialogue_index.json"
        speaking_style_index = f"{index_fold}/speaking_style_index.json"
        memory_index = f"{index_fold}/memory_index.json"
        SIB_index = f"{index_fold}/SIB_index.json"

        profile = None
        if args.add_profile:
            with open(f"./data/{dataset}/character_data/{person_name}/profile.json", "r") as f:
                profile = json.load(f)["profile"]
        bundle = None
        if args.add_persona:
            bundle = retrieve_bundle(
                query=question,
                memory_index=memory_index,
                speaking_style_index=speaking_style_index,
                SIB_index=SIB_index,
                diags_index=diags_index,
                k_memory=k_memory,
                k_speaking_style=k_speaking_style,
                k_SIB=k_SIB,
                k_diags=args.k_diags,
                emb_model=args.emb_model,
                only_dialogue=only_dialogue
            )
        messages = build_messages(
            person_name=person_name,
            question=question,
            bundle=bundle,
            language=language,
            source_role=source_role,
            profile=profile,
            dataset=dataset,
            only_dialogue=only_dialogue,
            anonymous=args.anonymous
        )
        persona_str = ""
        if bundle is not None:
            persona_str = format_persona_for_prompt(bundle, only_dialogue=only_dialogue)
        # print(persona_str)
        reply, input_tokens, output_tokens = generate_reply(messages, model=args.model, temperature=args.temperature, max_tokens=args.max_tokens, profile=profile, person_name=person_name, language=language, model_instance=model_instance, tokenizer=tokenizer, anonymous=args.anonymous)
        TOTAL_INPUT_TOKEN += input_tokens
        TOTAL_OUTPUT_TOKEN += output_tokens
        # print(reply)
        correct = None
        if task == "general_response" or task == "summary":
            reply = reply.strip()
        response_output = {"person_name": person_name, "source_role": source_role, "question": question, "ground_truth": ground_truth, "persona": persona_str, "model_reply": reply, "correct": correct}
        response_outputs.append(response_output)
        
        if (i + 1) % batch_size == 0:
            response_outputs = save_batch_and_clear_memory(response_outputs, output_folder, file_name, current_batch)
            current_batch += 1
            print(f"Memory usage check - Batch {current_batch} completed")
    elif dataset == "CharacterEval":
        person_name = data["role"]
        context_text = data["context"]
        index_fold = f"./data/{dataset}/character_data/{person_name}/persona_module/{extraction_model}"
        diags_index = f"{index_fold}/dialogue_index.json"
        speaking_style_index = f"{index_fold}/speaking_style_index.json"
        memory_index = f"{index_fold}/memory_index.json"
        SIB_index = f"{index_fold}/SIB_index.json"

        profile = None
        if args.add_profile:
            with open(f"./data/{dataset}/character_data/{person_name}/profile.json", "r") as f:
                profile = json.load(f)["profile"]
        bundle = None
        if args.add_persona:
            bundle = retrieve_bundle(
                query=context_text,
                memory_index=memory_index,
                speaking_style_index=speaking_style_index,
                SIB_index=SIB_index,
                diags_index=diags_index,
                k_memory=k_memory,
                k_speaking_style=k_speaking_style,
                k_SIB=k_SIB,
                k_diags=args.k_diags,
                emb_model=args.emb_model,
                only_dialogue=only_dialogue
            )

        messages = build_messages(
            person_name=person_name,
            question=context_text,
            bundle=bundle,
            language=language,
            profile=profile,
            dataset=dataset,
            only_dialogue=only_dialogue,
            anonymous=args.anonymous
        )
        persona_str = ""
        if bundle is not None:
            persona_str = format_persona_for_prompt(bundle, only_dialogue=only_dialogue)
        # print(persona_str)
        # print("messages:", messages)

        reply, input_tokens, output_tokens = generate_reply(messages, model=args.model, temperature=args.temperature, max_tokens=args.max_tokens, profile=profile, person_name=person_name, language=language, model_instance=model_instance, tokenizer=tokenizer, anonymous=args.anonymous)
        TOTAL_INPUT_TOKEN += input_tokens
        TOTAL_OUTPUT_TOKEN += output_tokens
        # print("reply:", reply)
        response_output = {"id": data["id"], "person_name": person_name, "novel_name": data["novel_name"], "context": context_text, "persona": persona_str, "model_reply": reply}
        response_outputs.append(response_output)
        
        if (i + 1) % batch_size == 0:
            response_outputs = save_batch_and_clear_memory(response_outputs, output_folder, file_name, current_batch)
            current_batch += 1
            print(f"Memory usage check - Batch {current_batch} completed")


if response_outputs:
    save_batch_and_clear_memory(response_outputs, output_folder, file_name, current_batch)
    current_batch += 1

print("Merging all batch files...")
response_outputs = merge_batch_files(output_folder, file_name, current_batch)
print(f"✓ Final merge completed. Total items: {len(response_outputs)}")


with open(f"{output_folder}/{file_name}.json", "w", encoding="utf-8") as f:
    json.dump(response_outputs, f, ensure_ascii=False, indent=4)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
try:
    with open(f"../token_report.json", "r", encoding="utf-8") as f:
        token_report = json.load(f)


    token_report[f"{dataset}_{task}_{language}_Open_generateOutput_{args.model}_{timestamp}"] = {
        "input_tokens": TOTAL_INPUT_TOKEN,
        "output_tokens": TOTAL_OUTPUT_TOKEN,
    }
except:
    token_report = {f"{dataset}_{task}_{language}_Open_generateOutput_{args.model}_{timestamp}": {
        "input_tokens": TOTAL_INPUT_TOKEN,
        "output_tokens": TOTAL_OUTPUT_TOKEN,
    }}
with open(f"../token_report.json", "w", encoding="utf-8") as f:
    json.dump(token_report, f, ensure_ascii=False, indent=4)
