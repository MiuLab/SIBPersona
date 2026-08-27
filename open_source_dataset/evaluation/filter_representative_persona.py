import json
import argparse
import os
from sklearn.cluster import KMeans
import kmedoids
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

argument_parser = argparse.ArgumentParser("Filter representative persona components")
argument_parser.add_argument("--person_name", type=str, default="")
argument_parser.add_argument("--dataset", type=str, required=True)
argument_parser.add_argument("--extraction_model", type=str, default="gpt-4o-mini")
argument_parser.add_argument("--num_of_points", type=int, default=30)
args = argument_parser.parse_args()

person_name = args.person_name
num_of_points = args.num_of_points
extraction_model = args.extraction_model
dataset = args.dataset
speaking_style_path = f"../data/{dataset}/character_data/{person_name}/persona_module/{extraction_model}/speaking_style_index.json"
memory_path = f"../data/{dataset}/character_data/{person_name}/persona_module/{extraction_model}/memory_index.json"
SIB_path = f"../data/{dataset}/character_data/{person_name}/persona_module/{extraction_model}/SIB_index.json"

with open(speaking_style_path, "r", encoding="utf-8") as f:
    speaking_style = json.load(f)
with open(SIB_path, "r", encoding="utf-8") as f:
    SIB = json.load(f)
with open(memory_path, "r", encoding="utf-8") as f:
    memory = json.load(f)


speaking_style_vec = speaking_style["vecs"]
SIB_situation_vec = SIB["vecs"]
SIB_internal_vec = SIB["internal_vecs"]
SIB_behavior_vec = SIB["behavior_vecs"]
memory_vec = memory["vecs"]

speaking_style_vec_np = np.asarray(speaking_style["vecs"], dtype=np.float32)
SIB_situation_vec_np = np.asarray(SIB["vecs"], dtype=np.float32)
memory_vec_np = np.asarray(memory["vecs"], dtype=np.float32)


speaking_style_text = speaking_style["texts"]
SIB_situation_text = SIB["texts"]
SIB_internal_text = SIB["internal_texts"]
SIB_behavior_text = SIB["behavior_texts"]
memory_text = memory["texts"]



def select_kmedoids_prototypes(X: np.ndarray, n_clusters: int = 10, method: str = "fasterpam1", random_state: int = 42):

    n = X.shape[0]
    k = min(n_clusters, n)

    sim = cosine_similarity(X)  # (n, n)
    diss = (1.0 - sim).astype(np.float32)

    if method == "fasterpam":
        res = kmedoids.fasterpam(diss, k, random_state=random_state)
    elif method == "fastpam1":
        res = kmedoids.fastpam1(diss, k, random_state=random_state)
    else:  # "pam"
        res = kmedoids.pam(diss, k, random_state=random_state)

    centers = np.array(res.medoids, dtype=int)
    print(f"Selected {len(centers)} medoids using {method} from {n} samples.")
    return np.unique(centers)


if len(memory_vec_np) == 0:
    rep_idx_memory = []
else:
    rep_idx_memory = select_kmedoids_prototypes(memory_vec_np, n_clusters=num_of_points+20)
if len(speaking_style_vec_np) == 0:
    rep_idx_speaking_style = []
else:
    rep_idx_speaking_style = select_kmedoids_prototypes(speaking_style_vec_np, n_clusters=num_of_points)
if len(SIB_situation_vec_np) == 0:
    rep_idx_SIB = []
else:
    rep_idx_SIB = select_kmedoids_prototypes(SIB_situation_vec_np, n_clusters=num_of_points)

speaking_after = {}
for key in speaking_style.keys():
    speaking_after[key] = [item for i, item in enumerate(speaking_style[key]) if i in rep_idx_speaking_style]
SIB_after = {}
for key in SIB.keys():
    SIB_after[key] = [item for i, item in enumerate(SIB[key]) if i in rep_idx_SIB]
memory_after = {}
for key in memory.keys():
    memory_after[key] = [item for i, item in enumerate(memory[key]) if i in rep_idx_memory]

speaking_rep = []
for i, meta in enumerate(speaking_after["metas"]):
    single_persona = {
        "atomic_point": meta["claim"]
    }
    if meta["dialogue"] != {}:
        single_persona["dialogue"] = meta["dialogue"]
    speaking_rep.append(single_persona)
SIB_rep = []
for i, meta in enumerate(SIB_after["metas"]):
    single_persona = {
        "situation": meta["situation"],
        "internal_state": meta["internal_state"],
        "behavior": meta["behavior"]
    }
    if meta["dialogue"] != {}:
        single_persona["dialogue"] = meta["dialogue"]
    SIB_rep.append(single_persona)
memory_rep = []
for i, meta in enumerate(memory_after["metas"]):
    single_persona = {
        "atomic_point": meta["claim"]
    }
    if meta["dialogue"] != {}:
        single_persona["dialogue"] = meta["dialogue"]
    memory_rep.append(single_persona)   


out_dir = f"../data/{dataset}/character_data/{person_name}/representative_persona"

os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "speaking_style_representative.json"), "w", encoding="utf-8") as f:
    json.dump(speaking_rep, f, ensure_ascii=False, indent=2)
with open(os.path.join(out_dir, "SIB_representative.json"), "w", encoding="utf-8") as f:
    json.dump(SIB_rep, f, ensure_ascii=False, indent=2)
with open(os.path.join(out_dir, "memory_representative.json"), "w", encoding="utf-8") as f:
    json.dump(memory_rep, f, ensure_ascii=False, indent=2)
