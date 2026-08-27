import subprocess
import sys
import os
import json
import argparse

parser = argparse.ArgumentParser("parameter for dataset pipeline")
parser.add_argument("--persona_score_model", type=str, default="gpt-4o-mini")
parser.add_argument("--sample_n", type=int, default=50)

args = parser.parse_args()
persona_score_model = args.persona_score_model
person_list = ["Influencer_A"]
sample_n = args.sample_n

for person in person_list:
    if os.path.exists(f"../data/{person}/for_sample_dataset/test_set_comment_reply_pairs_labeled_{persona_score_model}.json"):
        print(f"sample dataset for {person} already exists, skipping...")
    else:
        print(f"filter persona data for {person}...")
        cmd = [
            sys.executable,
            "persona_scoring.py",
            "--person_name", person,
            "--model", persona_score_model,
        ]
        subprocess.run(cmd)
    if os.path.exists(f"../data/{person}/for_evaluation/sample_data_{persona_score_model}_score=low.csv"):
        print(f"sample dataset for {person} already exists, skipping...")
    else:
        print(f"sample persona low data for {person}...")
        cmd = [
            sys.executable,
            "sample_dataset.py",
            "--person_name", person,
            "--n", str(sample_n),
            "--persona_score_range", "low",
        ]
        subprocess.run(cmd)
    if os.path.exists(f"../data/{person}/for_evaluation/sample_data_{persona_score_model}_score=medium.csv"):
        print(f"sample dataset for {person} already exists, skipping...")
    else:
        print(f"sample persona medium data for {person}...")
        cmd = [
            sys.executable,
            "sample_dataset.py",
            "--person_name", person,
            "--n", str(sample_n),
            "--persona_score_range", "medium",
        ]
        subprocess.run(cmd)
    if os.path.exists(f"../data/{person}/for_evaluation/sample_data_{persona_score_model}_score=high.csv"):
        print(f"sample dataset for {person} already exists, skipping...")
    else:
        print(f"sample persona high data for {person}...")
        cmd = [
            sys.executable,
            "sample_dataset.py",
            "--person_name", person,
            "--n", str(sample_n),
            "--persona_score_range", "high",
        ]
        subprocess.run(cmd)