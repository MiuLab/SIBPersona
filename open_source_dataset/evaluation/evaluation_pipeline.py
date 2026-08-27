import subprocess
import sys
import os
import json
import argparse

parser = argparse.ArgumentParser("parameter for memory pipeline")
parser.add_argument("--extraction_model", type=str, default="gpt-4o-mini")
parser.add_argument('--evaluation_model', type=str, default="gemini-2.5-flash")

args = parser.parse_args()
extraction_model = args.extraction_model
evaluation_model = args.evaluation_model
anonymous_str = "_Anonymous"
generate_models = ["allenai-Olmo-3-7B-Instruct"]
conditions = ["Persona_Profile"]
tasks = [["RoleAgentBench", "general_response"] ,["RoleAgentBench", "summary"]]
dimension_list = ["memory", "speaking_style", "SIB"]
languages = ["zh", "eng"]
runtimes = ["0"]
for gen_model in generate_models:
    for condition in conditions:
        for dataset, task in tasks:
            for language in languages:
                for runtime in runtimes:
                    runtime_str = ""
                    if int(runtime) > 0:
                        runtime_str = f"_run{runtime}"
                    file_path = f"extraction={extraction_model}_generate={gen_model}_{condition}{anonymous_str}"
                    for dimension in dimension_list:
                        cmd = [
                            sys.executable,
                            "LLM_evaluation.py",
                            "--model", evaluation_model,
                            "--task", task,
                            "--dataset", dataset,
                            "--language", language,
                            "--file_name", file_path,
                            "--dimension", dimension,
                            "--run_time", runtime,
                            ]
                        if os.path.exists(f"./data/{dataset}/{task}/{language}/LLM_eval_results/{file_path}_eval={evaluation_model}{runtime_str}/{dimension}.json"):
                            print(f"Skipping existing evaluation for task {task}, dataset {dataset}, language {language}, condition {condition}, {file_path} dimension {dimension} runtime {runtime}...")
                        else:
                            print(f"Running LLM eval for task {task}, dataset {dataset}, language {language}, condition {condition}, {file_path} dimension {dimension} runtime {runtime}...")
                            subprocess.run(cmd)