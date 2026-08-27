uv run generate_outputs.py --dataset CharacterEval --task charactereval --language zh --k_persona 3 --k_diags 0  --add_profile --add_persona --model allenai/Olmo-3-7B-Instruct --anonymous
uv run generate_outputs_impersona.py --dataset CharacterEval --task charactereval --language zh --add_profile --add_persona --model allenai/Olmo-3-7B-Instruct --anonymous
uv run generate_outputs.py --dataset CharacterEval --task charactereval --language zh --add_profile --model Neph0s/CoSER-Llama-3.1-8B --anonymous


# ablation study
uv run generate_outputs.py --dataset CharacterEval --task charactereval --language zh --k_persona 3 --k_diags 0  --add_profile --add_persona --model allenai/Olmo-3-7B-Instruct --anonymous --delete_SIB
uv run generate_outputs.py --dataset CharacterEval --task charactereval --language zh --k_persona 3 --k_diags 0  --add_profile --add_persona --model allenai/Olmo-3-7B-Instruct --anonymous --delete_SIB --delete_speaking_style
uv run generate_outputs.py --dataset CharacterEval --task charactereval --language zh --k_persona 3 --k_diags 0  --add_profile --add_persona --model allenai/Olmo-3-7B-Instruct --anonymous --delete_speaking_style
