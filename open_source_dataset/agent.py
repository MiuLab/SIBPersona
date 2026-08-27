#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from memory_module import (
    retrieve_topk_per_index,
)
from prompt import get_roleagentbench_response_prompt, get_charactereval_response_prompt
# ====== LLM Client (OpenAI or Azure OpenAI) ======

from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
import google.auth
import langchain_google_vertexai

load_dotenv(find_dotenv())



def _get_llm_client(model):
    """Return appropriate client based on model type."""
    open_source_models = [
        "allenai/Olmo-3-7B-Instruct",
        "Neph0s/CoSER-Llama-3.1-8B"
    ]
    
    # 檢查是否為開源模型
    if any(open_model in model for open_model in open_source_models):
        return "open_source"  
    

    elif model.startswith("gpt-"):
        azure_api_key = os.environ['AZURE_OPENAI_API_KEY']
        azure_api_version=os.environ['AZURE_API_VERSION']
        azure_endpoint = os.environ['AZURE_ENDPOINT']
        if model.startswith("gpt-4o") or model.startswith("gpt-5"):
            azure_api_key = os.environ['AZURE_OPENAI_API_KEY_gpt5']
            azure_endpoint = os.environ['AZURE_ENDPOINT_gpt5']
        """Return (client, is_azure: bool)."""
        try:
            client = OpenAI(api_key=azure_api_key, base_url=azure_endpoint)
            return client
        except:
            return None
        return client

# ====== Utilities ======

def _truncate(s: Optional[str], max_chars: int) -> str:
    if not s:
        return ""
    if max_chars <= 0:
        return s
    return s if len(s) <= max_chars else s[:max_chars] + "…"


def retrieve_bundle(
    query: str,
    memory_index: Optional[str],
    speaking_style_index: Optional[str],
    SIB_index: Optional[str],
    diags_index: Optional[str],
    k_memory: int = 3,
    k_speaking_style: int = 3,
    k_SIB: int = 3,
    k_diags: int = 5,
    emb_model: str = "text-embedding-3-small",
    only_dialogue: bool = False,
) -> Dict[str, List]:
    """Retrieve top-k per index. Missing paths are skipped gracefully."""
    out = {"Memory": [], "Speaking_Style": [], "SIB": [], "Dialogue": []}
    index_fold = memory_index.rsplit("/", 1)[0]
    if not only_dialogue:
        if memory_index and os.path.exists(memory_index) and k_memory > 0:
            out["Memory"] = retrieve_topk_per_index(query, memory_index, k=k_memory, model_name=emb_model, cache_path=f"{index_fold}/emb_cache.json")
        if speaking_style_index and os.path.exists(speaking_style_index) and k_speaking_style > 0:
            out["Speaking_Style"] = retrieve_topk_per_index(query, speaking_style_index, k=k_speaking_style, model_name=emb_model, cache_path=f"{index_fold}/emb_cache.json")
        if SIB_index and os.path.exists(SIB_index) and k_SIB > 0:
            out["SIB"] = retrieve_topk_per_index(query, SIB_index, k=k_SIB, model_name=emb_model, cache_path=f"{index_fold}/emb_cache.json")
    if diags_index and os.path.exists(diags_index) and k_diags > 0:
        out["Dialogue"] = retrieve_topk_per_index(query, diags_index, k=k_diags, model_name=emb_model, cache_path=f"{index_fold}/emb_cache.json")
    # print(out)
    return out

def format_persona_for_prompt(bundle, only_dialogue = False) -> str:
    if only_dialogue:
        diags = []
        items = bundle.get("Dialogue", [])
        for i, r in enumerate(items, 1):
            diaglogues = getattr(r, 'dialogue', '')
            diags.append(diaglogues)
        return diags
    persona = {"Memory":[], "Speaking_Style":[], "SIB":[], "Dialogue":[]}
    for mod in ("Memory", "Speaking_Style"):
        items = bundle.get(mod, [])
        if not items:
            continue
        for i, r in enumerate(items, 1):
            claim = getattr(r, 'claim', '')
            diaglogues = getattr(r, 'dialogue', '')
            if len(diaglogues) > 0:
                persona[mod].append({"atomic_point": claim, "diaglogues": diaglogues})
            else:
                persona[mod].append({"atomic_point": claim})
    SIB_triplet = bundle.get("SIB", [])
    # print("SIB_triplet:", SIB_triplet)
    for i, r in enumerate(SIB_triplet, 1):
        situation = getattr(r, 'situation', '')
        internal_state = getattr(r, 'internal_state', '')
        behavior = getattr(r, 'behavior', '')
        diaglogues = getattr(r, 'dialogue', '')
        if len(diaglogues) > 0:
            persona["SIB"].append({"situation": situation, "internal_state": internal_state, "behavior": behavior, "diaglogues": diaglogues})
        else:
            persona["SIB"].append({"situation": situation, "internal_state": internal_state, "behavior": behavior})
    items = bundle.get("Dialogue", [])
    for i, r in enumerate(items, 1):
        diaglogues = getattr(r, 'dialogue', '')
        persona["Dialogue"].append(diaglogues)
    for key in ["Memory", "Speaking_Style", "SIB", "Dialogue"]:
        if len(persona[key]) == 0:
            del persona[key]
    return persona


def build_messages(
    person_name: str,
    question: str,
    bundle: Dict[str, List] | None,
    language: str,
    dataset: str = "RoleBench",
    source_role: str = "",
    profile: Optional[str] = None,
    only_dialogue: bool = False,
    anonymous: bool = False
) -> List[Dict[str, str]]:
    """Compose chat messages for the LLM."""
    # 根据 anonymous参数决定是否使用匿名化的角色名称
    if bundle is not None:
        persona = format_persona_for_prompt(bundle, only_dialogue=only_dialogue)

    if dataset == "RoleAgentBench":
        if bundle is None:
            system_prompt, user_prompt = get_roleagentbench_response_prompt(profile=profile, source_role=source_role, person_name=person_name, persona=None, question=question, language=language, raw_dialogue=None)
        elif only_dialogue:
            system_prompt, user_prompt = get_roleagentbench_response_prompt(profile=profile, source_role=source_role, person_name=person_name, persona=None, question=question, language=language, raw_dialogue=persona)
        else:
            system_prompt, user_prompt = get_roleagentbench_response_prompt(profile=profile, source_role=source_role, person_name=person_name, persona=persona, question=question, language=language)
    elif dataset == "CharacterEval":
        if bundle is None:
            system_prompt, user_prompt = get_charactereval_response_prompt(profile=profile, person_name=person_name, persona=None, raw_dialogue=None)
        elif only_dialogue:
            system_prompt, user_prompt = get_charactereval_response_prompt(profile=profile, person_name=person_name, persona=None, raw_dialogue=persona)
        else:
            system_prompt, user_prompt = get_charactereval_response_prompt(profile=profile, person_name=person_name, persona=persona)
    if anonymous:
        system_prompt = system_prompt.replace(person_name, "<anonymous_character>")
        user_prompt = user_prompt.replace(person_name, "<anonymous_character>")
    # print("system prompt:", system_prompt)
    # print("user prompt:", user_prompt)
    if dataset == "CharacterEval":
        def make_inputs(context):
            dialogues= context.split('\n') 
            inputs = []  
            for dial in dialogues:
                role = dial.split("：")[0]
                dial = "：".join(dial.split("：")[1:])
                inputs.append({"from":role,"value":dial})
            return inputs
        def concat_messages(conversations, role, system, user):
            history = []
            first_query = user
            if conversations[0]['from'] == role:
                first_response = f"好的！现在我来扮演{role}。" + "我首先发话：" + conversations[0]['value']
            else:
                first_response = f"好的！现在我来扮演{role}。"
            history.append({"role": "system", "content": system})
            history.append({"role": "user", "content": first_query})
            if anonymous:
                first_response = first_response.replace(role, "<anonymous_character>")
            history.append({"role": "assistant", "content": first_response})
            
            for i in range(len(conversations)):
                if conversations[i]['from'] == role:
                    if i ==0:
                        continue
                    else:
                        assert conversations[i-1]['from'] != role
                        query = f"{conversations[i-1]['from']}：" + conversations[i-1]['value']
                        response = f"{conversations[i]['from']}：" + conversations[i]['value']
                    if anonymous:
                        query = query.replace(role, "<anonymous_character>")
                        response = response.replace(role, "<anonymous_character>")
                    history.append({"role": "user", "content": query})
                    history.append({"role": "assistant", "content": response})
            assert conversations[-1]['from'] != role
            
            query = f"{conversations[-1]['from']}：" + conversations[-1]['value']
            if anonymous:
                query = query.replace(role, "<anonymous_character>")
            return history, query
        messages, query = concat_messages(make_inputs(question), person_name, system_prompt, user_prompt)
        out = messages + [
            {"role": "user", "content": query},
        ]
    else:
        out = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    # print("\n=== Prompt Messages ===\n", out)
    return out


def generate_reply(messages: List[Dict[str, str]], model: str = "gpt-4o-mini", temperature: float = 0.6, max_tokens: int = 180, profile: Optional[str] = None, person_name: Optional[str] = None, language: Optional[str] = None, model_instance: Optional[AutoModel] = None, tokenizer: Optional[AutoTokenizer] = None, anonymous: bool = False) -> str:
    input_tokens = 0
    output_tokens = 0
    max_tries = 5
    client = _get_llm_client(model)
    # print("client:", client)

    if client == "open_source":
        # print("open_source")
        return generate_reply_open_source(messages, model, profile=profile, person_name=person_name, language=language, model_instance=model_instance, tokenizer=tokenizer, max_tokens=max_tokens, anonymous=anonymous)

    for attempt in range(max_tries):
        try:
            if model.startswith("gpt-"):
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                # print(f"LLM response: {resp}")
                if hasattr(resp, 'usage') and resp.usage:
                    batch_input_tokens = resp.usage.prompt_tokens
                    batch_output_tokens = resp.usage.completion_tokens

                    input_tokens += batch_input_tokens
                    output_tokens += batch_output_tokens
                if resp.choices and len(resp.choices) > 0:
                    content = resp.choices[0].message.content
                
                input_tokens += batch_input_tokens
                output_tokens += batch_output_tokens
            if content:

                if anonymous and person_name:
                    content = content.replace("<anonymous_character>", person_name)
                # print(content)
                return content.strip(), input_tokens, output_tokens
            # print(resp)
            print(f"Attempt {attempt + 1}: No valid response content")
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
    
    print("All attempts failed")
    return "", input_tokens, output_tokens

# ====== opensource model ======

def generate_reply_open_source(messages: List[Dict[str, str]], model: str, profile: Optional[str] = None, person_name: Optional[str] = None, language: Optional[str] = None, model_instance: Optional[AutoModel | AutoModelForCausalLM] = None, tokenizer: Optional[AutoTokenizer] = None, max_tokens: int = 200, anonymous: bool = False) -> str:
    """Generate reply using open source models via transformers."""
    
    
    if "Olmo" or "CoSER" in model:
        # formatted_string = tokenizer.apply_chat_template(messages, tokenize=False)
        # print(formatted_string)
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",  # 返回 PyTorch Tensor 格式
            add_generation_prompt=True  
        )
        attention_mask = torch.ones_like(input_ids)
        device = next(model_instance.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        output = model_instance.generate(
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.pad_token_id,  # 明確指定 pad_token_id
            eos_token_id=tokenizer.eos_token_id,
            temperature=0.6,
            max_new_tokens=max_tokens,
            top_p=0.95,
            do_sample=True,
            use_cache=True,
        )
        generated_text = tokenizer.decode(output[0, input_ids.shape[-1]:], skip_special_tokens=True)
        if generated_text.startswith("assistant\n"):
            generated_text = generated_text[len("assistant\n"):]

        if anonymous and person_name:
            generated_text = generated_text.replace("<anonymous_character>", person_name)
        return generated_text.strip(), 0, 0
