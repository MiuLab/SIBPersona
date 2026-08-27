#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_outputs_impersona.py

This script runs IMPersona as a baseline for datasets.
It uses IMPersona's hierarchical memory system to generate persona-aware responses.
"""

import argparse
import json
import os
import sys
import gc
import psutil
import subprocess
from datetime import datetime
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import transformers

# Add IMPersona to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'IMPersona'))
from IMPersona.IMPersona.memory_module import HierarchicalMemoryModule
from IMPersona.IMPersona.agents import RoleBenchAgent

# List of supported open-source models
OPEN_SOURCE_MODELS = [
    "allenai/Olmo-3-7B-Instruct",
]

parser = argparse.ArgumentParser(description="Generate outputs using IMPersona baseline for RoleAgentBench/CharacterEval")
parser.add_argument("--dataset", required=True, type=str, choices=["RoleAgentBench", "CharacterEval"])
parser.add_argument("--task", required=True, type=str, 
                    choices=["summary", "general_response", "charactereval"])
parser.add_argument("--language", required=True, type=str, choices=["zh", "eng"])
parser.add_argument("--model", default="gpt-4o-mini", 
                    help="Model to use for generation. Supports OpenAI/Azure models and open-source models like allenai/Olmo-3-7B-Instruct")
parser.add_argument("--extraction_model", default="gpt-4o-mini", help="Model used for memory extraction")
parser.add_argument("--add_persona", action="store_true", help="Add IMPersona memory retrieval")
parser.add_argument("--add_profile", action="store_true", help="Add character profile")
parser.add_argument("--save_batch_size", type=int, default=300, help="Batch size for saving results")
parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
parser.add_argument("--anonymous", action="store_true", help="Replace person_name with <anonymous_character> in prompts")
parser.add_argument("--resume", action="store_true", help="Resume from existing batch files")
parser.add_argument("--start_from", type=int, default=0, help="Start from specific index (used with resume)")
args = parser.parse_args()


def is_open_source_model(model_name: str) -> bool:
    """Check if the model is an open-source model."""
    return any(open_model in model_name for open_model in OPEN_SOURCE_MODELS)


def check_memory_usage(stage_name=""):
    """Check current memory usage"""
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


def optimize_memory_settings():
    """Set up memory optimization parameters"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        print(f"✓ GPU memory optimized. Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    gc.set_threshold(100, 5, 5)
    print(f"✓ Memory optimization settings applied")


def load_open_source_model(model_name: str):
    """Load open-source model and tokenizer."""
    model_instance = None
    tokenizer = None
    
    if model_name == "allenai/Olmo-3-7B-Instruct":
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model_instance = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            # attn_implementation="flash_attention_2"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"✓ Set pad_token to eos_token: {tokenizer.eos_token}")
        if model_instance.config.pad_token_id is None:
            model_instance.config.pad_token_id = tokenizer.eos_token_id
            print(f"✓ Set model pad_token_id to: {tokenizer.eos_token_id}")
        check_memory_usage("After loading Olmo-3-7B-Instruct")
        
    return model_instance, tokenizer


def load_memory_module(person_name: str, extraction_model: str, data_dir: str, dataset: str , model: str = "gpt-4o", 
                       model_instance=None, tokenizer=None, language: str = "zh") -> HierarchicalMemoryModule:
    """Load IMPersona's memory module for a character."""
    memory_bank_path = f"{data_dir}/{person_name}/IMPersona/{extraction_model}/memory_bank.json"
    
    if not os.path.exists(memory_bank_path):
        print(f"Warning: Memory bank not found at {memory_bank_path}")
        return None
    
    with open(memory_bank_path, 'r', encoding='utf-8') as f:
        memory_bank = json.load(f)
    
    memory_module = HierarchicalMemoryModule(
        attribute_path=memory_bank_path,
        person_name=person_name,
        extraction_model=extraction_model,
        model=model,
        model_instance=model_instance,
        tokenizer=tokenizer,
        language=language,
        dataset=dataset
    )
    
    return memory_module


def save_batch_and_clear_memory(response_outputs, output_folder, file_name, batch_num):
    """Save current batch results and clear memory"""
    os.makedirs(output_folder, exist_ok=True)
    batch_file = f"{output_folder}/{file_name}_batch_{batch_num}.json"
    
    with open(batch_file, 'w', encoding='utf-8') as f:
        json.dump(response_outputs, f, ensure_ascii=False, indent=4)
    
    print(f"✓ Batch {batch_num} saved to {batch_file} ({len(response_outputs)} items)")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return []


def merge_batch_files(output_folder, file_name, total_batches):
    """Merge all batch files into final result"""
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


def load_existing_batches(output_folder, file_name):
    """Load existing batch files to resume processing"""
    existing_data = []
    batch_num = 0
    
    while True:
        batch_file = f"{output_folder}/{file_name}_batch_{batch_num}.json"
        if os.path.exists(batch_file):
            with open(batch_file, 'r', encoding='utf-8') as f:
                batch_data = json.load(f)
                existing_data.extend(batch_data)
            print(f"✓ Loaded existing batch {batch_num} ({len(batch_data)} items)")
            batch_num += 1
        else:
            break
    
    return existing_data, batch_num


def main():
    TOTAL_INPUT_TOKEN = 0
    TOTAL_OUTPUT_TOKEN = 0
    
    extraction_model = args.extraction_model
    dataset = args.dataset
    language = args.language
    task = args.task
    data_dir = args.data_dir
    model_name = args.model
    if args.anonymous:
        anonymous = True
        anonymous_str = "_Anonymous"
    else:
        anonymous = False
        anonymous_str = ""
    # Optimize memory settings
    optimize_memory_settings()
    
    # Load open-source model if needed
    model_instance = None
    tokenizer = None
    if is_open_source_model(model_name):
        print(f"Loading open-source model: {model_name}")
        model_instance, tokenizer = load_open_source_model(model_name)
    
    # Load test data and character info
        
    if dataset == "RoleAgentBench":
        script_to_lang = {
            "Friends S1E1": "eng", "Harry Potter": "eng", "Merchant of Venice": "eng",
            "Sherlock - A Study in Pink": "eng", "The Big Bang Theory S1E1": "eng",
            "九品芝麻官": "zh", "唐人街探案": "zh", "家有儿女 S1E1": "zh",
            "狂飙 S1E1": "zh", "西游记-三打白骨精": "zh"
        }
        data_path = f"{data_dir}/RoleAgentBench"
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
        data_path = f"{data_dir}/CharacterEval"
        with open(f"{data_path}/data/test_data.jsonl", "r") as f:
            test_data = json.load(f)
    
    # Setup output folder and file name
    output_folder = f"{data_dir}/{dataset}/results/{task}/{language}"
    os.makedirs(output_folder, exist_ok=True)
    
    profile_str = "_Profile" if args.add_profile else ""
    persona_str = "_IMPersona" if args.add_persona else ""
    model_str = args.model.replace("/", "-")
    file_name = f"extraction={extraction_model}_generate={model_str}{persona_str}{profile_str}{anonymous_str}"
    
    # Check if final output already exists in resume mode
    final_output_path = f"{output_folder}/{file_name}.json"
    if args.resume and os.path.exists(final_output_path):
        print(f"✅ Final output already exists: {final_output_path}")
        with open(final_output_path, 'r', encoding='utf-8') as f:
            response_outputs = json.load(f)
        print(f"   Total items: {len(response_outputs)}/{len(test_data)}")
        
        if len(response_outputs) >= len(test_data):
            print(f"✅ All data already processed! Skipping execution.")
            print(f"   To rerun, remove the file or don't use --resume flag")
            return
        else:
            print(f"⚠️  File exists but incomplete ({len(response_outputs)}/{len(test_data)} items)")
            print(f"   Continuing from where it left off...")
    
    # Cache for memory modules and agents per character
    memory_cache = {}
    agent_cache = {}
    
    response_outputs = []
    batch_size = args.save_batch_size
    current_batch = 0
    start_index = 0
    
    # Resume from existing batches if requested
    if args.resume:
        existing_responses, current_batch = load_existing_batches(output_folder, file_name)
        start_index = len(existing_responses)
        print(f"🔄 Resume mode: Found {start_index} existing items in {current_batch} batches")
        print(f"   Starting from index {start_index}/{len(test_data)}")
    elif args.start_from > 0:
        start_index = args.start_from
        current_batch = start_index // batch_size
        print(f"⏩ Starting from index {start_index} (batch {current_batch})")
    
    check_memory_usage("Before processing")
    
    for i, data in enumerate(tqdm(test_data[start_index:], initial=start_index, total=len(test_data))):
        if dataset == "RoleAgentBench":
            person_name = data["target_role"]
            source_role = data.get("source_role", "")
            
            if task in ["general_response", "summary"]:
                question = data["question"]
            
            if task in ["general_response"]:
                ground_truth = data["answer"]
            elif task == "summary":
                ground_truth = data["summary"]
        
        elif dataset == "CharacterEval":
            person_name = data["role"]
            question = data["context"]
            ground_truth = None
            source_role = ""
            pure_question = question
        # Load or get cached memory module for this character
        if args.add_persona:
            if person_name not in memory_cache:
                character_data_dir = f"{data_dir}/{dataset}/character_data"
                memory_cache[person_name] = load_memory_module(
                    person_name, extraction_model, character_data_dir,
                    model=model_name,
                    model_instance=model_instance,
                    tokenizer=tokenizer,
                    language=language,
                    dataset=dataset
                )
            memory_module = memory_cache[person_name]
        else:
            memory_module = None
        
        # Load profile
        profile = None
        if args.add_profile:
            profile_path = f"{data_dir}/{dataset}/character_data/{person_name}/profile.json"
            if os.path.exists(profile_path):
                with open(profile_path, "r", encoding="utf-8") as f:
                    profile = json.load(f).get("profile", "")
        
        # Get or create agent for this character
        agent_key = f"{person_name}_{args.model}"
        if agent_key not in agent_cache:
            agent_cache[agent_key] = RoleBenchAgent(
                model_name=args.model,
                impersonation_name=person_name,
                memory_module=memory_module,
                data_dir=f"{data_dir}/{dataset}/character_data",
                lang=language,
                dataset=dataset,
                profile=profile,
                model_instance=model_instance,
                tokenizer=tokenizer,
                anonymous=anonymous
            )
        agent = agent_cache[agent_key]
        
        # Update memory module if needed (in case it was loaded later)
        agent.memory_module = memory_module
        if profile:
            agent.profile = profile
        
        # Generate response
        reply, memorie_summary, top, second, first, persona_text = agent.generate_response_impersona(
            question=question,
            pure_question= question,
            source_role=source_role, anonymous=anonymous
        )
        # print("---------------"*5)
        # print(f"memorie_summary: {memorie_summary}")
        # print("---------------"*5)
        # print(f"Top-level memories: {top}")
        # print("---------------"*5)
        # print(f"First-level memories: {first}")
        # print("---------------"*5)
        # print(f"Second-level memories: {second}")
        # print("---------------"*5)
        # print(f"Reply: {reply}")
        if reply == "":
            if dataset == "CharacterEval":
                reply = ""
            else:
                reply = "无法生成回复" if language == "zh" else "N/A"
        
        # Calculate metrics
        if dataset == "RoleAgentBench":
            correct = None
            
            memory_str = ""
            if second:
                memory_str = str([{
                    'attribute': m.get('attribute', ''),
                    'source_attributes': [{'attribute': a.get('attribute', ''), 'citation': a.get('citation', '')} 
                                          for a in m.get('source_attributes', [])]
                } for m in second])
            
            response_output = {
                "person_name": person_name,
                "source_role": source_role,
                "question": question,
                "ground_truth": ground_truth,
                "persona": memory_str,
                "model_reply": reply,
                "correct": correct
            }
        
        elif dataset == "CharacterEval":
            memory_str = ""
            if second:
                memory_str = str([{
                    'attribute': m.get('attribute', ''),
                    'source_attributes': [{'attribute': a.get('attribute', ''), 'citation': a.get('citation', '')} 
                                          for a in m.get('source_attributes', [])]
                } for m in second])
            
            response_output = {
                "id": data["id"],
                "person_name": person_name,
                "novel_name": data["novel_name"],
                "context": question,
                "persona": memory_str,
                "model_reply": reply
            }
        
        response_outputs.append(response_output)
        
        # Save batch and clear memory (adjust index for batch calculation)
        actual_index = start_index + i + 1
        if actual_index % batch_size == 0:
            response_outputs = save_batch_and_clear_memory(response_outputs, output_folder, file_name, current_batch)
            current_batch += 1
            check_memory_usage(f"After batch {current_batch}")
    
    # Save final batch
    if response_outputs:
        save_batch_and_clear_memory(response_outputs, output_folder, file_name, current_batch)
        current_batch += 1
    
    # Merge all batches
    print("Merging all batch files...")
    response_outputs = merge_batch_files(output_folder, file_name, current_batch)
    print(f"✓ Final merge completed. Total items: {len(response_outputs)}")
    


    
    # Save final results
    final_output_path = f"{output_folder}/{file_name}.json"
    with open(final_output_path, "w", encoding="utf-8") as f:
        json.dump(response_outputs, f, ensure_ascii=False, indent=4)
    
    print(f"✓ Results saved to {final_output_path}")
    
    # Save token usage report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    token_report_path = "../token_report.json"
    try:
        with open(token_report_path, "r", encoding="utf-8") as f:
            token_report = json.load(f)
    except:
        token_report = {}
    
    token_report[f"{dataset}_{task}_{language}_IMPersona_{args.model}_{timestamp}"] = {
        "input_tokens": TOTAL_INPUT_TOKEN,
        "output_tokens": TOTAL_OUTPUT_TOKEN,
    }
    
    with open(token_report_path, "w", encoding="utf-8") as f:
        json.dump(token_report, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
