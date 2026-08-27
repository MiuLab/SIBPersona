for task in general_response summary; do
    for eval_model in gemini-2.5-flash; do
        for who_first in SIB SOTA; do
            for language in zh eng; do
                for gen_model in allenai-Olmo-3-7B-Instruct; do
                    uv run python pair_wise.py --task $task --language $language --eval_model $eval_model --who_first $who_first --generation_model $gen_model --anonymous
                done               
            done
        done
    done
done