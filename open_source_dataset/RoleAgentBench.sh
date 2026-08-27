for task in general_response summary
do
   for language in eng zh 
      do
         uv run generate_outputs.py --dataset RoleAgentBench --task "$task" --language "$language" --add_profile --add_persona --k_persona 3 --k_diags 0 --model allenai/Olmo-3-7B-Instruct --anonymous
         uv run generate_outputs_impersona.py --dataset RoleAgentBench --task "$task" --language "$language" --add_profile --add_persona --model allenai/Olmo-3-7B-Instruct --anonymous
   done
done


