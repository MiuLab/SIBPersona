import pandas as pd
import json
import os
from typing import Literal

characterEval_dir = "../data/CharacterEval"

with open(f"{characterEval_dir}/data/character_profiles.json", "r") as f:
    character_dic = json.load(f)

with open(f"{characterEval_dir}/data/test_data.jsonl", "r") as f:
    test_data = json.load(f)

def save_characterEval_character_data(character_dic):
    for person_name, profile in character_dic.items():
        output_dir = f"{characterEval_dir}/character_data/{person_name}"
        os.makedirs(output_dir, exist_ok=True)
        profile_dic = {"profile":profile, "lang": "zh"}
        with open(f"{output_dir}/profile.json", "w") as f:
            json.dump(profile_dic, f, ensure_ascii=False, indent=4)
        
        conversations = []
        diag_id = 0
        for data in test_data:
            if data["role"] == person_name:
                conv_text = data["context"]
                conv = conv_text.split("\n")
                single_diag = []
                if len(conv) > 1:
                    for turn in conv:
                        split_turn = turn.split("：", 1)
                        if len(split_turn) == 2:
                            single_diag.append({"role": split_turn[0], "content": split_turn[1]})
                    conversations.append({"diag_id": diag_id, "dialogue": single_diag})
                    diag_id += 1


        with open(f"{output_dir}/dialogue.json", "w") as f:
            json.dump(conversations, f, ensure_ascii=False, indent=4)
if __name__ == "__main__":
    save_characterEval_character_data(character_dic)

