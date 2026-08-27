#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
process_memory_rolebench.py

This script runs IMPersona's memory extraction pipeline on RoleBench/RoleAgentBench data.
It converts the dialogue.json format to IMPersona's conversation format and extracts
hierarchical memories for each character.

"""

import json
import os
import sys
import warnings
import argparse
from typing import List, Dict, Literal
from dotenv import load_dotenv, find_dotenv

# Suppress specific asyncio warnings
warnings.filterwarnings("ignore", message=".*Event loop is closed.*", category=RuntimeWarning)

load_dotenv(find_dotenv())

# Add IMPersona to path
sys.path.insert(0, os.path.dirname(__file__))
from memory_utils import (
    extract_personal_attributes_parallel, 
    combine_and_infer_attributes_parallel, 
    generate_top_order_memories_sync
)

parser = argparse.ArgumentParser(description="Process RoleBench/RoleAgentBench dialogues into IMPersona memory bank")
parser.add_argument('--dataset', type=str, required=True, choices=["RoleAgentBench", "CharacterEval"])
parser.add_argument('--language', type=str, required=True, choices=["zh", "eng"])
parser.add_argument('--person_name', type=str, required=True, help="Character name to process")
parser.add_argument('--model_name_extract', type=str, default="gpt-4o-mini")
parser.add_argument('--model_name_combine', type=str, default="gpt-4o-mini")
parser.add_argument('--batch_size_extract', type=int, default=1)
parser.add_argument('--batch_size_combine', type=int, default=25)
parser.add_argument('--model_name_top', type=str, default="gpt-4o-mini")
parser.add_argument('--target_count_top', type=int, default=25)
parser.add_argument('--max_concurrent', type=int, default=5)
parser.add_argument('--data_dir', type=str, default="../data", help="Base data directory")
parser.add_argument('--force', action='store_true', help="Force reprocessing even if memory bank exists")
args = parser.parse_args()


def convert_dialogue_to_conversation(dialogue_data: List[Dict], person_name: str) -> List[List[Dict]]:
    """
    Convert RoleBench dialogue.json format to IMPersona conversation format.
    
    Args:
        dialogue_data: List of dialogues from dialogue.json
        person_name: Name of the character
        
    Returns:
        List of conversations in IMPersona format
    """
    conversations = []
    
    for diag in dialogue_data:
        dialogue = diag.get("dialogue", [])
        if not dialogue:
            continue
        
        # Convert each dialogue entry to IMPersona format
        conversation = []
        for entry in dialogue:
            role = entry.get("role", "")
            content = entry.get("content", "")
            
            # Determine speaker type
            if role == person_name:
                speaker = person_name
            else:
                speaker = role
            
            conversation.append({
                "speaker": speaker,
                "content": content,
            })
        
        if conversation:
            conversations.append(conversation)
    
    return conversations


def main():
    dataset = args.dataset
    language = args.language
    person_name = args.person_name
    
    # Set paths
    base_data_dir = f"{args.data_dir}/{dataset}/character_data/{person_name}"
    dialogue_path = f"{base_data_dir}/dialogue.json"
    
    # Output paths
    impersona_dir = f"{base_data_dir}/IMPersona/{args.model_name_extract}"
    temp_dir = f"{impersona_dir}/attributes"
    memory_bank_path = f"{impersona_dir}/memory_bank.json"
    gathered_attributes_path = f"{temp_dir}/gathered_attributes.json"
    combined_attributes_path = f"{temp_dir}/combined_attributes.json"
    top_memories_path = f"{temp_dir}/top_memories.json"
    conversation_store_path = f"{impersona_dir}/conversation_store.json"
    
    # Ensure directories exist
    os.makedirs(impersona_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    # Check if memory bank already exists
    if os.path.exists(memory_bank_path) and not args.force:
        print(f"Memory bank already exists at {memory_bank_path}. Use --force to reprocess.")
        sys.exit(0)
    
    # Check if dialogue file exists
    if not os.path.exists(dialogue_path):
        print(f"Error: Dialogue file not found at {dialogue_path}")
        sys.exit(1)
    
    # Load dialogue data
    print(f"Loading dialogue data from {dialogue_path}...")
    with open(dialogue_path, 'r', encoding='utf-8') as f:
        dialogue_data = json.load(f)
    
    # Convert to IMPersona format
    print(f"Converting {len(dialogue_data)} dialogues to IMPersona format...")
    conversations = convert_dialogue_to_conversation(dialogue_data, person_name)
    print(f"Converted to {len(conversations)} conversations")
    
    # Save conversation store
    with open(conversation_store_path, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, ensure_ascii=False, indent=4)
    print(f"Saved conversation store to {conversation_store_path}")
    
    # Check if we can skip processing steps
    if os.path.exists(combined_attributes_path) and not args.force:
        print("Found combined attributes, skipping extraction and combination steps...")
        with open(combined_attributes_path, 'r', encoding='utf-8') as f:
            combined_attributes = json.load(f)
        
        if os.path.exists(top_memories_path) and not args.force:
            print("Found top memories, skipping generation step...")
            with open(top_memories_path, 'r', encoding='utf-8') as f:
                top_memories = json.load(f)
        else:
            print("Generating top-order memories...")
            top_memories = generate_top_order_memories_sync(
                combined_attributes,
                args.model_name_top,
                args.target_count_top,
                person_name,
                args.max_concurrent,
                language
            )
            with open(top_memories_path, 'w', encoding='utf-8') as f:
                json.dump(top_memories, f, ensure_ascii=False, indent=4)
        
        combined_attributes["top_memories"] = top_memories
    else:
        # Extract attributes if needed
        if os.path.exists(gathered_attributes_path) and not args.force:
            print("Found gathered attributes, skipping extraction step...")
            with open(gathered_attributes_path, 'r', encoding='utf-8') as f:
                attributes = json.load(f)
        else:
            print(f"Extracting personal attributes using {args.model_name_extract}...")
            attributes = extract_personal_attributes_parallel(
                conversations,
                args.model_name_extract,
                args.batch_size_extract, 
                person_name,
                args.max_concurrent,
                language
            )
            with open(gathered_attributes_path, 'w', encoding='utf-8') as f:
                json.dump(attributes, f, ensure_ascii=False, indent=4)
            print(f"Extracted {len(attributes)} attributes")
        
        # Combine attributes
        print(f"Combining and inferring attributes using {args.model_name_combine}...")
        combined_attributes = combine_and_infer_attributes_parallel(
            attributes,
            args.model_name_combine,
            args.batch_size_combine,
            person_name,
            args.max_concurrent,
            language
        )
        with open(combined_attributes_path, 'w', encoding='utf-8') as f:
            json.dump(combined_attributes, f, ensure_ascii=False, indent=4)
        print(f"Combined into {len(combined_attributes.get('combined_attributes', []))} second-order memories")
        
        # Generate top-order memories
        print(f"Generating top-order memories using {args.model_name_top}...")
        top_memories = generate_top_order_memories_sync(
            combined_attributes,
            args.model_name_top,
            args.target_count_top,
            person_name,
            args.max_concurrent,
            language
        )
        with open(top_memories_path, 'w', encoding='utf-8') as f:
            json.dump(top_memories, f, ensure_ascii=False, indent=4)
        print(f"Generated {len(top_memories)} top-order memories")
        
        combined_attributes["top_memories"] = top_memories
    
    # Save the final memory bank
    print(f"Saving memory bank for {person_name}...")
    with open(memory_bank_path, 'w', encoding='utf-8') as f:
        json.dump(combined_attributes, f, ensure_ascii=False, indent=4)
    
    print(f"✓ Memory bank created successfully at {memory_bank_path}")
    
    # Print summary
    print("\n=== Memory Bank Summary ===")
    print(f"Character: {person_name}")
    print(f"Dataset: {dataset}")
    print(f"Language: {language}")
    print(f"First-order memories: {len(combined_attributes.get('attributes', []))}")
    print(f"Second-order memories: {len(combined_attributes.get('combined_attributes', []))}")
    print(f"Top-order memories: {len(combined_attributes.get('top_memories', []))}")


if __name__ == "__main__":
    main()
