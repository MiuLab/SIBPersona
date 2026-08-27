import json
from tqdm import tqdm
import argparse
import os
import openai
import traceback
import asyncio
from openai import AsyncOpenAI
from google import genai
from dotenv import load_dotenv, find_dotenv
from agent import retrieve_bundle
from datetime import datetime

load_dotenv(find_dotenv())
parser = argparse.ArgumentParser()
parser.add_argument('--language', type=str, choices=['zh', 'eng'], required=True)
parser.add_argument('--task', type=str, choices=['summary', 'general_response'], )
parser.add_argument('--extraction_model', type=str, default='gpt-4o-mini')
parser.add_argument('--generation_model', type=str, default='gpt-4o-mini')
parser.add_argument('--eval_model', type=str, default='gpt-4o-mini')
parser.add_argument('--who_first', type=str, choices=['SIB', 'SOTA'])
parser.add_argument("--anonymous", action="store_true")


args = parser.parse_args()
# condition = args.condition
task = args.task
who_first = args.who_first
extraction_model = args.extraction_model
generation_model = args.generation_model
eval_model = args.eval_model
language = args.language

anonymous_str = "_Anonymous" if args.anonymous else ""
azure_api_key = os.environ['AZURE_OPENAI_API_KEY']
azure_api_version=os.environ['AZURE_API_VERSION']
azure_endpoint = os.environ['AZURE_ENDPOINT']
gemini_key = os.environ['GEMINI_API_KEY']

if who_first == 'SIB':
    who_first_str = 'SIBFirst'
    condition_1 = 'SIB'
    condition_2 = 'SOTA'
else:
    who_first_str = 'SOTAFirst'
    condition_1 = 'SOTA'
    condition_2 = 'SIB'

SIB_file_name = f"extraction={extraction_model}_generate={generation_model}_Persona_Profile{anonymous_str}"
SOTA_file_name = f"extraction={extraction_model}_generate={generation_model}_IMPersona_Profile{anonymous_str}"

folder = f"./data/RoleAgentBench/results/{task}/{language}"
global TOTAL_INPUT_TOKEN
global TOTAL_OUTPUT_TOKEN
TOTAL_INPUT_TOKEN = 0
TOTAL_OUTPUT_TOKEN = 0
print('condition_1:', condition_1, 'condition_2:', condition_2)

async def get_result(SIB_data, SOTA_data, system_prompt, user_prompt, try_time):
    global TOTAL_INPUT_TOKEN, TOTAL_OUTPUT_TOKEN
    if try_time >= 5: return None
    # if data['model_output'] == None:
    #     data['rank'] = []
    #     return data
    try:
        if eval_model.startswith("gpt"):
            client = AsyncOpenAI(api_key=azure_api_key, base_url=azure_endpoint)
            response = await client.chat.completions.create(
                                model=eval_model,
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_prompt}
                                ],
                                temperature=0.2,
                                top_p=0.8,
                                frequency_penalty=0.0,
                                presence_penalty=0.0,
                            )
            # print(response.choices[0].message.content)
            rank_dic = response.choices[0].message.content
        elif eval_model.startswith("gemini"):
            client = genai.Client(api_key=gemini_key)
            response = await client.aio.models.generate_content(
                model=eval_model,
                contents=system_prompt + "\n\n" + user_prompt,
            )
            rank_dic = response.text
        else:
            raise ValueError(f"Unsupported model type: {eval_model}")

        # rank_dic = response.text
        # print("rank_dic:", rank_dic)
        rank_dic = rank_dic.replace('`','')
        if rank_dic[:6] == 'python':
            rank_dic = rank_dic[6:]
        if rank_dic[:4] == "json":
            rank_dic = rank_dic[4:]
        rank_dic = json.loads(rank_dic)
        # print(rank_dic)
        for i in rank_dic:
            if i['condition'] == "condition_1":
                i['condition'] = condition_1
            elif i['condition'] == "condition_2":
                i['condition'] = condition_2
        # print(rank_dic)
        winner = rank_dic[0]['condition']
        evaluation_data = {
            "question": SIB_data["question"],
            "person_name": SIB_data["person_name"],
            "SIB_model_reply": SIB_data["model_reply"],
            "SOTA_model_reply": SOTA_data["model_reply"],
            "ground_truth": SIB_data["ground_truth"],
            "rank" : rank_dic,
            "winner": winner
        }

        # print(rank_dic)
        return evaluation_data
    except Exception as e:
        print(f"[Try {try_time}] Error occurred:", e)
        traceback.print_exc()
        await asyncio.sleep((try_time+1) * 5)
        return await get_result(SIB_data, SOTA_data, system_prompt, user_prompt, try_time+1)


with open(f"{folder}/{SIB_file_name}.json", 'r', encoding='utf-8') as f:
    SIB_data = json.load(f)

with open(f"{folder}/{SOTA_file_name}.json", 'r', encoding='utf-8') as f:
    SOTA_data = json.load(f)



async def create_evaluation_task(SIB_data, SOTA_data, index):
    """Create a single evaluation task"""
    person_name = SIB_data['person_name']
    with open(f"./data/RoleAgentBench/character_data/{person_name}/profile.json", "r") as f:
        role_description = json.load(f)["profile"]
    question = SIB_data['question']
    index_fold = f"./data/RoleAgentBench/character_data/{person_name}/persona_module/{extraction_model}"
    diags_index = f"{index_fold}/dialogue_index.json"
    speaking_style_index = f"{index_fold}/speaking_style_index.json"
    memory_index = f"{index_fold}/memory_index.json"
    SIB_index = f"{index_fold}/SIB_index.json"
    bundle = retrieve_bundle(
                query=question,
                memory_index=memory_index,
                speaking_style_index=speaking_style_index,
                SIB_index=SIB_index,
                diags_index=diags_index,
                k_memory=0,
                k_speaking_style=0,
                k_SIB=0,
                k_diags=10,
                emb_model="text-embedding-3-small",
                only_dialogue=True
            )
    diags = []
    items = bundle.get("Dialogue", [])
    for i, r in enumerate(items, 1):
        diaglogues = getattr(r, 'dialogue', '')
        diags.append(diaglogues)
    # print("Retrieved bundle for evaluation:", diags)
    representative_dialogues = diags
    ground_truth = SIB_data["ground_truth"]
    if who_first == 'SIB':
        list_model_answer_dict = [{"condition": "condition_1", "answer": SIB_data["model_reply"]}, {"condition": "condition_2", "answer": SOTA_data['model_reply']}]
    else:
        list_model_answer_dict = [{"condition": "condition_1", "answer": SOTA_data['model_reply']}, {"condition": "condition_2", "answer": SIB_data["model_reply"]}]

    if language == 'zh':
        system_prompt = "你是一个角色扮演的效果对比助手，你会根据输出的角色特征和质量来对condition进行排名，然后使用python dict list 输出结果。"
        prompt = f"""下列condition要扮演的角色是"{person_name}"。{person_name}的角色描述是"{role_description}"。我需要根据下面两个原则对下列condition进行排名：
1. 哪一个的角色说话风格特征更加明显，说话更加符合角色描述，说话越有特色就越好；
2. 哪一个的结果蕴含了更多与角色相关的知识和记忆，越丰富越好。
以下是{person_name}的一些代表性对话，供你参考：
{representative_dialogues}
输入给各个condition的问题是：
{question}
{person_name}的真實回答是：
{ground_truth}
各个condition针对该问题的回答分别为：
{list_model_answer_dict}
现在请你根据上述两个原则，对各个condition进行排名。避免任何位置偏见，并确保模型回答的呈现顺序不会影响你的决定。不要对condition的名字带有偏见。然后使用一个包含condition与其排名、这样排名的理由的列表返回结果，也就是说，请务必使用如下格式返回结果：
[{{"condition": <condition-name>, "reason": <rank-reason>, "rank": <condition-rank>}}, {{"condition":<condition-name>, "reason": <rank-reason>, "rank": <condition-rank>}}]
你的回答必须是一个有效的python 字典列表以保证我能够直接使用python 解析它，不要有多余的内容！请给出尽可能准确的、符合大多数人直觉的排名。
"""
    else:
        system_prompt = "You are a role−playing performance comparison assistant. You should rank the conditions based on the role characteristics and text quality of their responses. The rankings are then output using Python dictionaries and lists."
        prompt = f"""The conditions below are to play the role of "{person_name}". The role description of "{person_name}" is "{role_description}". I need to rank the following conditions based on the two criteria below:
1. Which one has more pronounced role speaking style, and speaks more in line with the role description. The more distinctive the speaking style, the better.
2. Which one's output contains more knowledge and memories related to the role; the richer, the better. (If the question contains reference answers, then the role−specific knowledge and memories are based on the reference answer.)
Here are some representative dialogues of {person_name} for your reference:
{representative_dialogues}
The question provided to each condition is:
{question}
The respective answers from the conditions to this question are:
{list_model_answer_dict}
The ground truth answer from {person_name} is:
{ground_truth}
Now, based on the above two criteria, please rank the conditions. Avoid any positional biases and ensure that the order in which the responses are presented does not influence your decision. Do not favor certain model names.
Then, use a list containing the condition's name, its rank, and the reason for its ranking to return the results, i.e., please ensure to use the following format to return the results:
[{{"condition": <condition−name>, "reason": <rank−reason>, "rank": <condition−rank>}}, {{"condition": <condition−name>, "reason": <rank−reason>, "rank": <condition−rank>}}]
Your answer must be a valid Python list of dictionaries to ensure I can directly parse it using Python. Do not include any extraneous content! Please provide a ranking that is as accurate as possible and aligns with the intuition of most people.
"""
    # print(prompt)
    ans = await get_result(SIB_data, SOTA_data, system_prompt, prompt, 0)
    ans["diags"] = representative_dialogues
    if ans is None:
        print(f"Error processing item {c}")
    return ans

async def main():
    print('data:', SIB_file_name)
    print(f'Processing {len(SIB_data)} items with batch async...')


    semaphore = asyncio.Semaphore(5)  

    async def limited_task(SIB_data, SOTA_data, original_index):
        async with semaphore:
            result = await create_evaluation_task(SIB_data, SOTA_data, original_index)
            return original_index, result


    indexed_tasks = []
    assert len(SIB_data) == len(SOTA_data), "SIB_data and SOTA_data must have the same length"
    for c, (SIB_d, SOTA_d) in enumerate(zip(SIB_data, SOTA_data)):
        if SOTA_d["model_reply"] is None or SIB_d["model_reply"] is None or SOTA_d["model_reply"].strip() == "" or SIB_d["model_reply"].strip() == "" or SOTA_d["model_reply"].startswith("Error") or SIB_d["model_reply"].startswith("Error"):
            print(f"Skipping item {c} due to None model_reply")
            indexed_tasks.append(asyncio.sleep(0, result=(c, None)))  
        else:
            indexed_tasks.append(limited_task(SIB_d, SOTA_d, c))

    print("Running batch async evaluation...")

    results = []


    with tqdm(total=len(indexed_tasks), desc="Processing evaluations") as pbar:
        for coro in asyncio.as_completed(indexed_tasks):
            try:
                result = await coro
                results.append(result)
                pbar.update(1)
            except Exception as e:
                print(f"\nError in task: {e}")
                results.append((len(results), None))
                pbar.update(1)


    results.sort(key=lambda x: x[0])
    final_result = [result for _, result in results]


    successful_results = [r for r in final_result if r is not None]
    failed_count = len(final_result) - len(successful_results)

    print(f"Successfully processed: {len(successful_results)}/{len(final_result)} items")
    if failed_count > 0:
        print(f"Failed: {failed_count} items")

    out_folder = f"./data/RoleAgentBench/pair_wise_eval/{task}/{language}/{who_first_str}"

    os.makedirs(f'{out_folder}', exist_ok=True)
    with open(f'{out_folder}/{SIB_file_name}_eval={eval_model}.json','w', encoding='utf-8') as f:
        f.write(json.dumps(final_result, ensure_ascii=False, indent=4))

    print(f"Completed! Results saved to {out_folder}/{SIB_file_name}_eval={eval_model}.json")

if __name__ == "__main__":
    asyncio.run(main())
