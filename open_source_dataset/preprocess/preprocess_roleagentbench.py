import pandas as pd
import json
import os
from typing import Literal


roleAgentbench_dir = "../data/RoleAgentBench"
script_to_lang = {"Friends S1E1": "eng", "Harry Potter": "eng", "Merchant of Venice": "eng", "Sherlock - A Study in Pink": "eng",
                  "The Big Bang Theory S1E1": "eng", "九品芝麻官": "zh", "唐人街探案": "zh", "家有儿女 S1E1": "zh", "狂飙 S1E1": "zh", "西游记-三打白骨精": "zh"}

def save_roleAgentbench_character_data(script):
    lang = script_to_lang[script]
    with open(f"{roleAgentbench_dir}/{script}/raw/role_summary.json", "r") as f:
        roleAgentbench_character_dic = json.load(f)

    for person_name, profile in roleAgentbench_character_dic.items():
        output_dir = f"{roleAgentbench_dir}/character_data/{person_name}"
        os.makedirs(output_dir, exist_ok=True)
        profile_dic = {"profile":profile, "lang": lang}
        with open(f"{output_dir}/profile.json", "w") as f:
            json.dump(profile_dic, f, ensure_ascii=False, indent=4)
        
        conversations = []
        with open(f"{roleAgentbench_dir}/{script}/profiles/{person_name}.jsonl", "r") as f:
            for line in f:
                convo = json.loads(line)
                conversations.append(convo)
        
        chat_format = []
        diag_id = conversations[0]["scene_id"]
        single_chat = {
            "diag_id": diag_id,
            "dialogue": []
        }
        for convo in conversations:
            if convo["scene_id"] == diag_id:
                single_diag = {
                    "role": convo["role"],
                    "content": convo["content"]
                }
                single_chat["dialogue"].append(single_diag)
            else:
                diag_id = convo["scene_id"]
                chat_format.append(single_chat)
                single_chat = {
                    "diag_id": diag_id,
                    "dialogue": []
                }
                single_diag = {
                    "role": convo["role"],
                    "content": convo["content"]
                }
                single_chat["dialogue"].append(single_diag)
        with open(f"{output_dir}/dialogue.json", "w") as f:
            json.dump(chat_format, f, ensure_ascii=False, indent=4)

for script in script_to_lang.keys():
    save_roleAgentbench_character_data(script)