import subprocess
import sys
import os
import json
import argparse

parser = argparse.ArgumentParser("parameter for memory pipeline")
parser.add_argument("--dataset", type=str, required=True, help="Specify the target person", choices=["RoleAgentBench", "CharacterEval"])
parser.add_argument("--language", type=str, required=True, help="Specify the target language", choices=["zh", "eng"])
parser.add_argument("--recognizing_model", type=str, default="gpt-4o-mini")
parser.add_argument("--extraction_model", type=str, default="gpt-4o-mini")


args = parser.parse_args()
dataset = args.dataset
language = args.language
recognizing_model = args.recognizing_model
extraction_model = args.extraction_model

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
elif dataset == "CharacterEval":
    characterEval_dir = "./data/CharacterEval"
    with open(f"{characterEval_dir}/data/character_profiles.json", "r") as f:
        character_dic = json.load(f)
    character_list = list(character_dic.keys())
    language = "zh"  # CharacterEval is only in Chinese
print(f"Character list for dataset {dataset} in language {language}: {character_list}")
start_run = False
for character in character_list:
    if os.path.exists(f"./data/{dataset}/character_data/{character}/recognized/recognized_dialogue.json"):
        print(f"recognized file for {character} already exists, skipping...")
    else:
        cmd = [
            sys.executable,
            "memory_recognizing.py",
            "--person_name", character,
            "--dataset", dataset,
            "--model", recognizing_model,
        ]
        print(f"Running recognizing for {character}...")
        subprocess.run(cmd)
    if os.path.exists(f"./data/{dataset}/character_data/{character}/persona_module/{extraction_model}/memory.json") and os.path.exists(f"./data/{dataset}/character_data/{character}/persona_module/{extraction_model}/SIB.json") and os.path.exists(f"./data/{dataset}/character_data/{character}/persona_module/{extraction_model}/speaking_style.json"):
        print(f"extraction file for {character} already exists, skipping...")
    else:
        cmd = [
            sys.executable,
            "memory_extraction.py",
            "--person_name", character,
            "--dataset", dataset,
            "--model", extraction_model,
        ]
        print(f"Running extraction for {character}...")
        subprocess.run(cmd)

    for module in ["memory", "speaking_style", "SIB"]:
        if os.path.exists(f"./data/{dataset}/character_data/{character}/persona_module/{extraction_model}/{module}_index.json"):
            print(f"Index for {character} - module {module} already exists, skipping...")
            continue
        cmd = [
            sys.executable,
            "memory_module.py",
            "build-index",
            "--input", f"./data/{dataset}/character_data/{character}/persona_module/{extraction_model}/{module}.json",
            "--out", f"./data/{dataset}/character_data/{character}/persona_module/{extraction_model}/{module}_index.json",
            "--type", "persona",
            "--person_name", character,
            "--dataset", dataset,
            "--extraction_model", extraction_model,
        ]
        print(f"Building index for {character} - module {module}...")
        subprocess.run(cmd)