import subprocess
import sys
import os
import json
import argparse

parser = argparse.ArgumentParser("parameter for inference pipeline")
parser.add_argument("--persona_score_model", type=str, default="gpt-4o-mini")
parser.add_argument("--recognizing_model", type=str, default="gpt-4o-mini")
parser.add_argument("--extraction_model", type=str, default="gpt-4o-mini")
parser.add_argument("--generate_model", default="gpt-4.1")

args = parser.parse_args()
recognizing_model = args.recognizing_model
extraction_model = args.extraction_model
generate_model = args.generate_model
persona_score_model = args.persona_score_model

person_list = ["Influencer_A"] 

file_names = [f"sample_data_{persona_score_model}_score=low", f"sample_data_{persona_score_model}_score=medium", f"sample_data_{persona_score_model}_score=high"]

for file_name in file_names:
    for person in person_list:
        cmd = [
            sys.executable,
            "generate_outputs.py",
            "--person_name", person,
            "--extraction_model", extraction_model,
            "--dataset_file_name", file_name,
            "--model", generate_model,
            "--add_profile",
        ]
        print(f"Running inference for {person}...")
        subprocess.run(cmd)