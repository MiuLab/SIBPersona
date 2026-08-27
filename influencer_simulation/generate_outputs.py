import argparse
import json
import pandas as pd
from agent import retrieve_bundle, build_messages, generate_reply, format_persona_for_prompt
import os
from tqdm import tqdm
from datetime import datetime
import tiktoken
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import subprocess
import transformers
import sys



parser = argparse.ArgumentParser()
parser.add_argument("--person_name", required=True, help="Persona display name")
parser.add_argument("--dataset_file_name", required=True)
parser.add_argument("--k_persona", type=int, default=3)
parser.add_argument("--k_pairs", type=int, default=0)
parser.add_argument("--emb_model", default="text-embedding-3-small")
parser.add_argument("--extraction_model", default="gpt-4o-mini")
parser.add_argument("--model", default="gpt-4o-mini",
                   choices=[
                       "gpt-4o-mini", "gpt-4.1", "gpt-5",
                       "gemini-3-pro-preview",
                       "gemini-3-flash-preview",
                       "gpt-5.2",
                       "DeepSeek-V3.2",
                       "claude-opus-4-6"
                   ],
                   help="Model to use for generation")
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--max_tokens", type=int, default=4096)
parser.add_argument("--add_profile", action="store_true")
args = parser.parse_args()

person_name = args.person_name
extraction_model = args.extraction_model

index_fold = f"./data/{person_name}/persona_module/{extraction_model}"
pairs_index = f"{index_fold}/dialogue_index.json"

speaking_style_index = f"{index_fold}/speaking_style_index.json"
memory_index = f"{index_fold}/memory_index.json"
SIB_index = f"{index_fold}/SIB_index.json"

profile = None
profile_str = ""
if args.add_profile:
    profile_str = "_withProfile"
    with open(f"./data/{person_name}/profile.json", "r") as f:
        profile = json.load(f)["profile"]


data_path = f"./data/{person_name}/for_evaluation/{args.dataset_file_name}.csv"
output_folder = f"./data/{person_name}/response_outputs"

data_df = pd.read_csv(data_path)

TOTAL_INPUT_TOKEN = 0
TOTAL_OUTPUT_TOKEN = 0
TOTAL_ESTIMATED_INPUT_TOKEN = 0
response_outputs = []


model_instance = None
tokenizer = None



for idx, row in tqdm(data_df.iterrows()):
    fan_text = row["query"]
    article = row["article"]
    ground_truth = row["answer"]
    bundle = retrieve_bundle(
        query=fan_text,
        memory_index=memory_index,
        speaking_style_index=speaking_style_index,
        SIB_index=SIB_index,
        pairs_index=pairs_index,
        k_persona=args.k_persona,
        k_pairs=args.k_pairs,
        emb_model=args.emb_model,
    )

    messages = build_messages(
        person_name=person_name,
        current_article=article,
        fan_text=fan_text,
        bundle=bundle,
        profile=profile
    )

    
    persona_str = format_persona_for_prompt(bundle, person_name)
    # print(persona_str)
    reply, input_tokens, output_tokens = generate_reply(messages, model=args.model, temperature=args.temperature, max_tokens=args.max_tokens, model_instance=model_instance, tokenizer=tokenizer)
    # print(f"Input tokens: {input_tokens}, Output tokens: {output_tokens}")
    # print(reply)
    TOTAL_INPUT_TOKEN += input_tokens
    TOTAL_OUTPUT_TOKEN += output_tokens
    response_output = {"article":article, "query":fan_text, "ground_truth": ground_truth, "persona": persona_str, "model_reply": reply}
    response_outputs.append(response_output)



os.makedirs(output_folder, exist_ok=True)
with open(f"{output_folder}/extraction={extraction_model}_generate={args.model}_{args.dataset_file_name}{profile_str}.json", "w", encoding="utf-8") as f:
    json.dump(response_outputs, f, ensure_ascii=False, indent=4)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
try:
    with open(f"./token_report.json", "r", encoding="utf-8") as f:
        token_report = json.load(f)


    token_report[f"{person_name}_{args.dataset_file_name}_generateOutput_{args.model}_{timestamp}"] = {
        "input_tokens": TOTAL_INPUT_TOKEN,
        "output_tokens": TOTAL_OUTPUT_TOKEN,
    }
except:
    token_report = {f"{person_name}_{args.dataset_file_name}_generateOutput_{args.model}_{timestamp}": {
        "input_tokens": TOTAL_INPUT_TOKEN,
        "output_tokens": TOTAL_OUTPUT_TOKEN,
    }}
with open(f"./token_report.json", "w", encoding="utf-8") as f:
    json.dump(token_report, f, ensure_ascii=False, indent=4)