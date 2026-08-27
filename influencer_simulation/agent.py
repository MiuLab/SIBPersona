#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from memory_module import (
    retrieve_topk_per_index,
    SearchResult_persona,
    SearchResult_pair,
)
from prompt import get_response_system_prompt, get_response_user_prompt
# ====== LLM Client (OpenAI or Azure OpenAI) ======
from anthropic import Anthropic
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import pandas as pd
from google import genai
from google.genai import types
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
load_dotenv(find_dotenv())



def _get_llm_client(model):
    """Return appropriate client based on model type."""

    
    azure_api_key = os.environ['AZURE_OPENAI_API_KEY']
    azure_api_version=os.environ['AZURE_API_VERSION']
    azure_endpoint = os.environ['AZURE_ENDPOINT']
    claude_api_key = os.environ['ANTHROPIC_API_KEY']
    gemini_key = os.environ.get('GEMINI_API_KEY', '')
    
    """Return (client, is_azure: bool)."""
    try:
        if model.startswith("gpt") or model.lower().startswith("deepseek"):
            client = OpenAI(
                api_key=azure_api_key,
                base_url=azure_endpoint,
            )
        elif model.startswith("claude"):
            client = Anthropic(api_key=claude_api_key)
        elif model.startswith("gemini"):
            client = genai.Client(api_key=gemini_key)
        return client
    except:
        return None


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
    pairs_index: Optional[str],
    k_persona: int = 3,
    k_pairs: int = 5,
    emb_model: str = "text-embedding-3-small",
) -> Dict[str, List]:
    """Retrieve top-k per index. Missing paths are skipped gracefully."""
    out = {"Memory": [], "Speaking_Style": [], "SIB": [], "Pair": []}
    index_fold = memory_index.rsplit("/", 1)[0]
    if k_persona > 0 and memory_index and os.path.exists(memory_index):
        out["Memory"] = retrieve_topk_per_index(query, memory_index, k=k_persona, model_name=emb_model, cache_path=f"{index_fold}/emb_cache.json")
    if k_persona > 0 and speaking_style_index and os.path.exists(speaking_style_index):
        out["Speaking_Style"] = retrieve_topk_per_index(query, speaking_style_index, k=k_persona, model_name=emb_model, cache_path=f"{index_fold}/emb_cache.json")
    if k_persona > 0 and SIB_index and os.path.exists(SIB_index):
        out["SIB"] = retrieve_topk_per_index(query, SIB_index, k=k_persona, model_name=emb_model, cache_path=f"{index_fold}/emb_cache.json")
    if k_pairs > 0 and pairs_index and os.path.exists(pairs_index):
        out["Pair"] = retrieve_topk_per_index(query, pairs_index, k=k_pairs, model_name=emb_model, cache_path=f"{index_fold}/emb_cache.json")
    # print(out)
    return out


def format_persona_for_prompt(bundle, person_name) -> str:
    persona = {"Memory":[], "Speaking_Style":[], "SIB":[], "Comment_pair":[]}
    for mod in ("Memory", "Speaking_Style"):
        items = bundle.get(mod, [])
        if not items:
            continue
        for i, r in enumerate(items, 1):
            claim = getattr(r, 'claim', '')
            evidence_pair = getattr(r, 'evidence_pair', '')
            evidence_pair_list = [value for key, value in evidence_pair.items()]
            evidence_pair_adjust = []
            for pair in evidence_pair_list:
                if isinstance(pair, dict) and "Fans" in pair and person_name in pair:
                    evidence_pair_adjust.append({"粉絲": pair["Fans"], person_name: pair[person_name]})
            if len(evidence_pair_adjust) > 0:
                persona[mod].append({
                    "atomic_point": claim,
                    "evidence_pair": evidence_pair_adjust,
                })
            else:
                persona[mod].append({"atomic_point": claim})
    
    SIB_triplet = bundle.get("SIB", [])
    for i, r in enumerate(SIB_triplet, 1):
        situation = getattr(r, 'situation', '')
        internal_state = getattr(r, 'internal_state', '')
        behavior = getattr(r, 'behavior', '')
        evidence_pair = getattr(r, 'evidence_pair', '')
        evidence_pair_list = [value for key, value in evidence_pair.items()]
        evidence_pair_adjust = []
        for pair in evidence_pair_list:
            if isinstance(pair, dict) and "Fans" in pair and person_name in pair:
                evidence_pair_adjust.append({"粉絲": pair["Fans"], person_name: pair[person_name]})
        if len(evidence_pair_adjust) > 0:
            persona["SIB"].append({
                "situation": situation,
                "internal_state": internal_state,
                "behavior": behavior,
                "evidence_pair": evidence_pair_adjust,
            })
        else:
            persona["SIB"].append({
                "situation": situation,
                "internal_state": internal_state,
                "behavior": behavior,
            })
    
    pairs = bundle.get("Pair", [])
    if pairs:
        for i, r in enumerate(pairs, 1):
            fan = getattr(r, 'fan_text', '')
            aut = getattr(r, 'author_text', '')
            persona["Comment_pair"].append({"粉絲": fan, person_name: aut})
    
    keys_to_remove = [key for key in persona.keys() if len(persona[key]) == 0]
    for key in keys_to_remove:
        persona.pop(key)
    return persona


def build_messages(
    person_name: str,
    current_article: str,
    fan_text: str,
    bundle: Dict[str, List],
    profile: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Compose chat messages for the LLM."""
    article_text = _truncate(current_article, max_chars=4000)

    persona = format_persona_for_prompt(bundle, person_name)

    system = get_response_system_prompt(person_name)
    # print(persona)
    instructions = get_response_user_prompt(profile, person_name, persona, article_text, fan_text)

    out = [
        {"role": "system", "content": system},
        {"role": "user", "content": instructions},
    ]
    # print("\n=== Prompt Messages ===\n", instructions)
    return out


def generate_reply(messages: List[Dict[str, str]], model: str = "gpt-4o-mini", temperature: float = 0.6, max_tokens: int = 180, model_instance: Optional[AutoModelForCausalLM] = None, tokenizer: Optional[AutoTokenizer] = None) -> str:
    max_tries = 5
    input_tokens = 0
    output_tokens = 0
    thought_tokens = 0
    client = _get_llm_client(model)

    
    # Azure uses deployment name in `model`
    for attempt in range(max_tries):
        try:
            if "deepseek" in model.lower():
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                )
            elif "gpt-5" in model:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    reasoning_effort = "low"
                )
            elif model.startswith("gpt"):
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            elif model.startswith("claude"):
                system_prompt_for_claude = messages[0]['content']
                messages_for_claude = [msg for msg in messages if msg['role'] != 'system']
                resp = client.messages.create(
                    model=model,
                    system=system_prompt_for_claude,
                    messages=messages_for_claude,
                    thinking={
                        "type": "adaptive"
                    },
                    output_config={
                        "effort": "low"
                    },
                    max_tokens=max_tokens,
                )
                if hasattr(resp, 'usage') and resp.usage:
                    input_tokens = resp.usage.input_tokens
                    output_tokens = resp.usage.output_tokens

                content = "".join(
                    block.text for block in resp.content
                    if block.type == "text" and block.text.strip()
                )
                if content:
                    return content.strip(), input_tokens, output_tokens

                print(f"Attempt {attempt + 1}: No valid Claude response content: {resp}")
                continue
            elif model.startswith("gemini"):
                # Convert messages to Gemini format
                gemini_contents = []
                system_instruction = None
                for msg in messages:
                    if msg["role"] == "system":
                        system_instruction = msg["content"]
                    elif msg["role"] == "user":
                        gemini_contents.append(types.Content(
                            role="user",
                            parts=[types.Part(text=msg["content"])]
                        ))
                
                # Build config
                if model == "gemini-3-pro-preview":
                    thinking_level = "low"
                elif model == "gemini-3-flash-preview":
                    thinking_level = "minimal"
                generate_config = types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    thinking_config=types.ThinkingConfig(thinking_level=thinking_level)
                )
                if system_instruction:
                    generate_config.system_instruction = system_instruction
                
                resp = client.models.generate_content(
                    model=model,
                    contents=gemini_contents,
                    config=generate_config,
                )
                
                # Extract tokens from Gemini response
                if hasattr(resp, 'usage_metadata') and resp.usage_metadata:
                    input_tokens = resp.usage_metadata.prompt_token_count or 0
                    output_tokens = resp.usage_metadata.candidates_token_count or 0
                    # Check for thought tokens in candidates_tokens_details if available
                    if hasattr(resp.usage_metadata, 'thoughts_token_count'):
                        thought_tokens = resp.usage_metadata.thoughts_token_count or 0
                
                # print(f"Gemini input tokens: {input_tokens}, output tokens: {output_tokens}, thought tokens: {thought_tokens}")
                output_tokens += thought_tokens
                # Extract text from Gemini response
                content = resp.text
                if content:
                    return content.strip(), input_tokens, output_tokens
                
                print(f"Attempt {attempt + 1}: No valid Gemini response content: {resp}")
                continue
            
            # Handle GPT response
            if hasattr(resp, 'usage') and resp.usage:
                batch_input_tokens = resp.usage.prompt_tokens
                batch_output_tokens = resp.usage.completion_tokens

                input_tokens += batch_input_tokens
                output_tokens += batch_output_tokens
            if resp.choices and len(resp.choices) > 0:
                content = resp.choices[0].message.content
                if content:
                    return content.strip(), input_tokens, output_tokens
            # print(resp)
            print(f"Attempt {attempt + 1}: No valid response content")
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
    
    print("All attempts failed")
    return "", input_tokens, output_tokens



# ====== CLI ======
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--person_name", required=True, help="Persona display name")
    parser.add_argument("--article", help="article text", default="")
    parser.add_argument("--fan_text", help="The fan comment to reply to", default="")

    parser.add_argument("--k_persona", type=int, default=3)
    parser.add_argument("--k_pairs", type=int, default=5)
    parser.add_argument("--emb_model", default=os.getenv("EMB_MODEL", "text-embedding-3-small"))
    parser.add_argument("--extraction_model", default="gpt-4o-mini")
    parser.add_argument("--model", default="gpt-4.1")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_tokens", type=int, default=180)
    parser.add_argument("--index_fold", type=str, default="")
    parser.add_argument("--add_profile", action="store_true")
    args = parser.parse_args()


    article = args.article.strip()
    fan_text = args.fan_text.strip()

    person_name = args.person_name.strip()
    extraction_model = args.extraction_model
    index_fold = f"./data/{person_name}/persona_module/{extraction_model}"
    pairs_index = f"{index_fold}/pairs_index.json"
    speaking_style_index = f"{index_fold}/speaking_style_index.json"
    memory_index = f"{index_fold}/memory_index.json"
    SIB_index = f"{index_fold}/SIB_index.json"


    profile = None
    if args.add_profile:
        with open(f"./data/{person_name}/profile.json", "r") as f:
            profile = json.load(f)["profile"]
    bundle = retrieve_bundle(
        query=fan_text,
        memory_index=memory_index,
        speaking_style_index=speaking_style_index,
        SIB_index=SIB_index,
        pairs_index=pairs_index,
        k_persona=args.k_persona,
        k_pairs=args.k_pairs,
        emb_model=args.emb_model,
    )
    # print(bundle)
    messages = build_messages(
        person_name=person_name,
        current_article=article,
        fan_text=fan_text,
        bundle=bundle,
        profile=profile
    )
    print(messages)
    reply, input_tokens, output_tokens = generate_reply(messages, model=args.model, temperature=args.temperature, max_tokens=args.max_tokens)
    print("\n=== Generated Reply ===\n" + reply)
