# random sample comment data from persona distinct comments
import json
import random
import argparse
import os
import pandas as pd

parser = argparse.ArgumentParser(description="random sample comment data from persona distinct comments")
parser.add_argument("--person_name", type=str, default="Influencer_A")
parser.add_argument("--n", type=int, default=50, help="number of positive pairs to sample")
parser.add_argument("--model", type=str, default="gpt-4o-mini")
parser.add_argument("--seed", type=int, default=42, help="random seed (optional)")
parser.add_argument("--persona_score_range", type=str, choices=["high", "medium", "low"], default="high", help="persona score range to sample from")
args = parser.parse_args()

random.seed(args.seed)
model = args.model

person_name = args.person_name
if args.persona_score_range == "high":
    sample_upper_bound = 1.1
    sample_lower_bound = 0.8
elif args.persona_score_range == "medium":
    sample_upper_bound = 0.8
    sample_lower_bound = 0.5
else:  # low
    sample_upper_bound = 0.5
    sample_lower_bound = 0



in_path  = f"../data/{args.person_name}/for_sample_dataset/test_set_comment_reply_pairs_labeled_{model}.json"
articles_path = f"../data/{args.person_name}/for_sample_dataset/test_set_article.json"
for_eval_folder = f"../data/{args.person_name}/for_evaluation"


with open(in_path, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(articles_path, "r", encoding="utf-8") as f:
    articles = json.load(f)

article_map = {a["id"]: a for a in articles}


positives = []
for a_idx, article in enumerate(data):
    article_id = article.get("article_id")
    for p_idx, pair in enumerate(article.get("pairs", [])):
        if sample_lower_bound <= pair.get("persona_score") < sample_upper_bound:

            article_info = article_map.get(article_id, "empty")
            positives.append({
                "article_id": article_id,
                "pair_index": p_idx,  
                "Fans": pair["Fans"],
                "reply": pair[person_name],
                "article": article_info
            })

# sampling, if k > len(positives), just return all positives
k = min(args.n, len(positives))
sampled = random.sample(positives, k) if k > 0 else []


df = {"article":[], "publish_date":[], "query":[], "answer":[]}
for data in sampled:
    title = ""
    if data["article"]["title"] is not None:
        title = data["article"]["title"] + "\n"
    text = data["article"]["text"]
    df["article"].append(text)
    df["publish_date"].append(data["article"]["publish_date"])
    df["query"].append(data["Fans"])
    df["answer"].append(data["reply"])

df = pd.DataFrame(df)
os.makedirs(for_eval_folder, exist_ok=True)
df.to_csv(f"{for_eval_folder}/sample_data_{model}_score={args.persona_score_range}.csv", index=False)
print(f"Total positives: {len(positives)} | Sampled: {k}")
