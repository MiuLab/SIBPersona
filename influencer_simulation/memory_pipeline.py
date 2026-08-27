import subprocess
import sys
import os
import json
import argparse

parser = argparse.ArgumentParser("parameter for memory pipeline")
parser.add_argument("--recognizing_model", type=str, default="gpt-4o-mini")
parser.add_argument("--extraction_model", type=str, default="gpt-4o-mini")
parser.add_argument("--file_name_for_construct_memory_bank", type=str, default="data_example_for_persona_bank_construction")

args = parser.parse_args()
recognizing_model = args.recognizing_model
extraction_model = args.extraction_model
file_name_for_construct_memory_bank = args.file_name_for_construct_memory_bank
person_list = ["Influencer_A"]

for person in person_list:
    if os.path.exists(f"./data/{person}/recognized/recognized_{file_name_for_construct_memory_bank}_{recognizing_model}.json"):
        print(f"recognized file for {person} already exists, skipping...")
    else:
        cmd = [
            sys.executable,
            "memory_recognizing.py",
            "--person_name", person,
            "--file_name", file_name_for_construct_memory_bank,
            "--model", recognizing_model,
        ]
        print(f"Running recognizing for {person}...")
        subprocess.run(cmd)
    if os.path.exists(f"./data/{person}/persona_module/{extraction_model}/memory.json"):
        print(f"extraction for {person} already exists, skipping...")
    else:
        cmd = [
            sys.executable,
            "memory_extraction.py",
            "--person_name", person,
            "--model", extraction_model,
            "--recognizing_model", recognizing_model,
            "--file_name", f"recognized_{file_name_for_construct_memory_bank}",
        ]
        print(f"Running extraction for {person}...")
        subprocess.run(cmd)

    for module in ["SIB", "speaking_style", "memory"]:
        if os.path.exists(f"./data/{person}/persona_module/{extraction_model}/{module}_index.json"):
            print(f"Index for {person} - module {module} already exists, skipping...")
            continue
        cmd = [
            sys.executable,
            "memory_module.py",
            "build-index",
            "--input", f"./data/{person}/persona_module/{extraction_model}/{module}.json",
            "--out", f"./data/{person}/persona_module/{extraction_model}/{module}_index.json",
            "--type", "persona",
            "--target_name", person,
            "--extraction_model", extraction_model,
        ]
        print(f"Building index for {person} - module {module}...")
        subprocess.run(cmd)