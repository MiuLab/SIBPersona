import os
import json
from openai import AsyncOpenAI
import asyncio
import re
from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm
from typing import Literal
load_dotenv(find_dotenv())

async def extract_personal_attributes_async(conversations, model_name="gpt-4o-mini", person_name="Ben", batch_size=20, lang: Literal["zh", "eng"] = "zh"):
    """
    Process conversations in batches to extract personal attributes/facts about a specific person using async API calls.
    
    Args:
        conversations: List of conversation dictionaries
        batch_size: Number of conversations to process in each batch
        person_name: Name of the person to extract attributes about
        lang: Language for prompts ("zh" or "eng")
        
    Returns:
        List of dictionaries containing attributes, their citations, and timestamps
    """
    azure_api_version=os.environ['AZURE_API_VERSION']
    azure_endpoint = os.environ['AZURE_ENDPOINT']
    azure_api_key = os.environ['AZURE_OPENAI_API_KEY']
    client = AsyncOpenAI(api_key=azure_api_key, base_url=azure_endpoint)
    
    # Format conversations for the prompt with clear conversation numbering
    formatted_convos = []
    for idx, convo in enumerate(conversations):
        formatted_convo = f"CONVERSATION {idx+1} START:\n"
        for message in convo:
            speaker = message.get('speaker', message.get('role', 'unknown'))
            content = message.get('content', '')
            formatted_convo += f"{speaker}: {content}\n"
        formatted_convo += f"CONVERSATION {idx+1} END\n"
        formatted_convos.append(formatted_convo)

    if lang == "zh":
        system_prompt = "从对话中提取个人属性，并搭配引用。"
        user_prompt = f"""
你是一位擅长分析对话并从中提取个人属性和事实的专家。

我将提供你 {len(conversations)} 段有关 {person_name} 的对话。你的任务是：

1. 从这些对话中提取有关 {person_name} 的具体属性和事实。不要提取个性特质或对其性格的评论等内容。
2. 专注于持久性属性（例如兴趣、背景、关系等），而不是一次性事件。

例如：如果 Ben 目前正在参与 AI 的研究项目，你应该提取的是「Ben 进行 AI 方面的研究。」而不是「Ben 目前正在参与一个 AI 的研究项目。」

3. 对于每个属性，都请引用具体的对话内容以证明这项资讯的来源。
4. 每项属性请用清晰、简洁的语句来陈述与 {person_name} 有关的资讯。
5. 重要：一定要包含出处的对话编号（例如："Conversation 2"）。

以下是好的属性提取范例：
- "{person_name} 在西维吉尼亚州摩根镇的摩根镇高中上过高中。" (Citation: Conversation 2, {person_name}: "我是在摩根镇长大的，高中也在那里的摩根镇高中就读。")
- "{person_name} 有一只名叫 Ellie 的狗。" (Citation: Conversation 1, {person_name}: "我等等要带 Ellie，我的狗，去散步。")
- "{person_name} 喜欢创作音乐。" (Citation: Conversation 3, 朋友："你的音乐制作怎么样了？我等不及要再听你那些超棒的歌曲了。")
- "{person_name} 曾在高中时是网球队的一员。" (Citation: Conversation 1, {person_name}: "我很怀念像高中时那样打网球比赛的日子。")

请按照上述格式准确输出，每个属性都包含三行资讯：Attribute、Citation。

- Attribute: {person_name} 在西维吉尼亚州摩根镇的摩根镇高中上过高中.
- Citation: Conversation 2, {person_name}: "我是在摩根镇长大的，高中也在那里的摩根镇高中就读。"

- Attribute: {person_name} 有一只名叫 Ellie 的狗.
- Citation: Conversation 1, {person_name}: "我等等要带 Ellie，我的狗，去散步。"

以下是对话内容：
{'''
'''.join(formatted_convos)}

请依照要求的格式准确输出
"""
    else:  # eng
        system_prompt = "You extract personal attributes from conversations with citations"
        user_prompt = f"""
You are an expert at analyzing conversations and extracting personal attributes and facts about people.

I'll provide you with {len(conversations)} conversations involving {person_name}. Your task is to:

1. Extract specific attributes and facts about {person_name} from these conversations. Do not extract traits, opinions on personality, etc.
2. Focus on persistent attributes (like hobbies, background, relationships) rather than one-time events. 

For example: if Ben is currently in a research project in AI, you should extract that "Ben does research in AI.", not "Ben is currently in a research project in AI".

3. For each attribute, cite a specific message from the conversations where this information appears.
4. Format each attribute as a clear, concise statement about {person_name}.
5. IMPORTANT: Always include the exact conversation number in your citation (e.g., "Conversation 2").

Examples of good attributes:
- "{person_name} went to high school in Morgantown, West Virginia at Morgantown High School." (Citation: Conversation 2, {person_name}: "I grew up in Morgantown and went to high school there at Morgantown High School.")
- "{person_name} has a dog named Ellie." (Citation: Conversation 1, {person_name}: "I need to take Ellie, my dog, for a walk later.")
- "{person_name} likes to create music." (Citation: Conversation 3, Friend: "How's your music production going? I need to hear some banger songs again.")
- "{person_name} played tennis on his high school team." (Citation: Conversation 1, {person_name}: "I miss playing tennis competitively like I did in high school.")

Please format your response EXACTLY as follows, with each attribute having these three lines:

- Attribute: {person_name} went to high school in Morgantown, West Virginia.
- Citation: Conversation 2, {person_name}: "I grew up in Morgantown and went to high school there."

- Attribute: {person_name} has a dog named Ellie.
- Citation: Conversation 1, {person_name}: "I need to take Ellie, my dog, for a walk later."

Here are the conversations:

{'''
'''.join(formatted_convos)}
"""
    
    # Call the model
    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=4096,
            temperature=1,
            top_p=1,
        )
        # print("Raw response content:", response.choices[0].message.content)
        # Parse the response using a custom parser instead of JSON
        if response.choices[0].message.content == None:
            return []
        return parse_attributes_from_text(response.choices[0].message.content)
    finally:
        await client.close()

def parse_attributes_from_text(text):
    """
    Parse attributes from the model's text response.
    
    Args:
        text: The text response from the model
    
    Returns:
        List of dictionaries containing attributes, their citations, and timestamps
    """
    attributes = []
    current_attribute = {}
    
    # Split the text into lines
    lines = text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
            
        # Remove leading dash or bullet point if present
        if line.startswith('- '):
            line = line[2:]
        elif line.startswith('• '):
            line = line[2:]
            
        # Check for attribute, citation, or timestamp
        if line.lower().startswith('attribute:'):
            # If we already have an attribute in progress, save it
            if current_attribute and 'attribute' in current_attribute:
                attributes.append(current_attribute)
                current_attribute = {}
            
            current_attribute['attribute'] = line[len('attribute:'):].strip()
        
        elif line.lower().startswith('citation:'):
            if 'attribute' in current_attribute:  # Only add if we have an attribute
                citation_text = line[len('citation:'):].strip()
                current_attribute['citation'] = citation_text
                
                # Extract conversation number from citation
                conversation_match = re.search(r'conversation\s+(\d+)', citation_text.lower())
                if conversation_match:
                    current_attribute['conversation_number'] = int(conversation_match.group(1))
                else:
                    current_attribute['conversation_number'] = None
        
        elif line.lower().startswith('timestamp:'):
            if 'attribute' in current_attribute:  # Only add if we have an attribute
                current_attribute['timestamp'] = line[len('timestamp:'):].strip()
                
                # If we have all three fields, add to attributes and reset
                if all(k in current_attribute for k in ['attribute', 'citation', 'timestamp']):
                    attributes.append(current_attribute)
                    current_attribute = {}
    
    # Add the last attribute if it wasn't added
    if current_attribute and 'attribute' in current_attribute:
        # Ensure all fields exist
        if 'citation' not in current_attribute:
            current_attribute['citation'] = ''
            current_attribute['conversation_number'] = None
        elif 'conversation_number' not in current_attribute:
            # Try to extract conversation number if not already done
            conversation_match = re.search(r'conversation\s+(\d+)', current_attribute['citation'].lower())
            if conversation_match:
                current_attribute['conversation_number'] = int(conversation_match.group(1))
            else:
                current_attribute['conversation_number'] = None
                
        if 'timestamp' not in current_attribute:
            current_attribute['timestamp'] = ''
        attributes.append(current_attribute)
    
    return attributes

async def process_conversations_in_parallel(conversations, model_name="gpt-4o-mini", batch_size=20, person_name="Ben", max_concurrent=5, lang: Literal["zh", "eng"] = "zh"):
    """
    Process multiple batches of conversations concurrently to extract personal attributes.
    
    Args:
        conversations: List of conversation dictionaries
        batch_size: Number of conversations to process in each batch
        person_name: Name of the person to extract attributes about
        max_concurrent: Maximum number of concurrent API calls
        lang: Language for prompts ("zh" or "eng")
        
    Returns:
        List of dictionaries containing attributes, their citations, and timestamps
    """
    all_attributes = []
    
    # Create batches of conversations
    # print(len(conversations))
    batches = [conversations[i:i+batch_size] for i in range(0, len(conversations), batch_size)]
    total_batches = len(batches)
    # print(f"Total batches to process: {total_batches}")
    # print(f"Batch size: {batch_size}")
    # print(f"Max concurrent requests: {max_concurrent}")
    # Process batches in chunks to control concurrency
    for i in range(0, len(batches), max_concurrent):
        # print(0)
        current_chunk = batches[i:i+max_concurrent]
        current_tasks = []
        
        # Print progress manually instead of using tqdm
        print(f"Processing batches {i+1}-{min(i+max_concurrent, total_batches)} of {total_batches} ({(i+1)/total_batches*100:.1f}% - {min(i+max_concurrent, total_batches)/total_batches*100:.1f}%)")
        
        # Create tasks for all batches in the current chunk
        for batch_idx, batch in enumerate(current_chunk):
            task = extract_personal_attributes_async(batch, model_name, person_name, batch_size, lang)
            current_tasks.append(task)
        
        # Execute all tasks concurrently and wait for them to complete
        batch_results = await asyncio.gather(*current_tasks)
        
        # Process the results
        for batch_idx, result in enumerate(batch_results):
            # Get the corresponding batch
            batch = current_chunk[batch_idx]
            
            # Add the actual conversation to each attribute based on conversation_number
            for attr in result:
                if 'conversation_number' in attr and attr['conversation_number'] is not None:
                    # Adjust the conversation number to be 1-indexed within the batch
                    conv_idx = attr['conversation_number'] - 1
                    if 0 <= conv_idx < len(batch):
                        attr['conversation_content'] = batch[conv_idx]
                    else:
                        attr['conversation_content'] = None
                else:
                    attr['conversation_content'] = None
            
            # Add results to all_attributes
            all_attributes.extend(result)
        
        # Optional: add a small delay to avoid rate limiting
        await asyncio.sleep(0.1)
    
    print(f"Processing complete! Extracted {len(all_attributes)} attributes.")
    return all_attributes

# Function to run the async code from a synchronous context
def extract_personal_attributes_parallel(conversations, model_name="gpt-4o-mini", batch_size=20, person_name="Ben", max_concurrent=5, lang: Literal["zh", "eng"] = "zh"):
    """
    Wrapper function to run the async extraction in a synchronous context.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(process_conversations_in_parallel(
            conversations, model_name, batch_size, person_name, max_concurrent, lang
        ))
        return result
    finally:
        # Ensure all pending tasks are completed before closing
        pending = asyncio.all_tasks(loop)
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()

########################################################
# Combine and infer attributes
########################################################

async def combine_and_infer_attributes_async(attributes_batch, person_name="Ben", model_name="gpt-4o", lang: Literal["zh", "eng"] = "zh"):
    """
    Asynchronously combines related attributes and infers new attributes based on existing ones.
    
    Args:
        attributes_batch: A batch of attribute dictionaries
        person_name: Name of the person the attributes are about
        lang: Language for prompts ("zh" or "eng")
        
    Returns:
        Dictionary containing combined and inferred attributes for this batch
    """
    azure_api_version=os.environ['AZURE_API_VERSION']
    azure_endpoint = os.environ['AZURE_ENDPOINT']
    azure_api_key = os.environ['AZURE_OPENAI_API_KEY']
    client = AsyncOpenAI(api_key=azure_api_key, base_url=azure_endpoint)
    
    # Format attributes for the prompt
    formatted_attributes = []
    for idx, attr in enumerate(attributes_batch):
        counter = 0
        timestamp = attr.get('timestamp', '')
        if not timestamp:
            timestamp = attr.get('Timestamp', '')
        while not timestamp and idx+counter < len(attributes_batch) and idx-counter >= 0:
            timestamp = attributes_batch[idx+counter].get('timestamp', '')
            if not timestamp:
                timestamp = attributes_batch[idx+counter].get('Timestamp', '')
            if not timestamp:
                timestamp = attributes_batch[idx-counter].get('timestamp', '')
            if not timestamp:
                timestamp = attributes_batch[idx-counter].get('Timestamp', '')
            counter += 1
        try:
            formatted_attr = f"Attribute {idx+1}: {attr['attribute']} (Citation: {attr['citation']}, Timestamp: {timestamp})"
        except KeyError:
            print("KeyError in attribute formatting:", attr)
            continue
        formatted_attributes.append(formatted_attr)
    
    if lang == "zh":
        system_prompt = "分析个人属性，把相关的合并并且推论新的属性"
        user_prompt = f"""
你是一位擅长分析个人资讯并能将相关事实联系起来的专家。

我将提供一份从对话中提取的关于 {person_name} 的属性清单。你的任务是：

1.找出并合并那些描述相同事情但语句不同或内容互补的属性。
2.推理出新属性：根据多个现有属性合理推断出 {person_name} 的其他资讯。

例如：
- 如果一条属性说「{person_name} 就读于斯坦福」，另一条说「{person_name} 主修计算机科学」，你可以合并成：「{person_name} 就读于斯坦福，主修计算机科学。」
- 如果有属性提到「{person_name} 在西雅图长大」与「{person_name} 为了大学搬到波士顿」，你可以推断：「{person_name} 为了高等教育从美国西岸搬到了东岸。」

请严格按照以下格式回复，分为两个明确的区块：

COMBINED ATTRIBUTES:
- Attribute: {person_name} 就读于斯坦福，主修计算机科学。
- Based on: attribute 3, attribute 7
    
- Attribute: {person_name} 已经弹钢琴超过十年，并会在当地表演。
- Based on: attribute 12, attribute 15

INFERRED ATTRIBUTES:
- Attribute: {person_name} 为了高等教育从美国西岸搬到了东岸。
- Based on: attribute 2, attribute 9
    
- Attribute: {person_name} 很可能对科技和音乐都有兴趣。
- Based on: attribute 3, attribute 12

以下是有关{person_name}的属性列表：
{'''
'''.join(formatted_attributes)}
"""
    else:  # eng
        system_prompt = "You analyze personal attributes, combine related ones, and infer new information."
        user_prompt = f"""
You are an expert at analyzing personal information and making connections between related facts.

I'll provide you with a list of attributes about {person_name} that were extracted from conversations.
Your task is to:

1. Identify and combine attributes that are talking about the same thing but might be phrased differently or contain complementary information.
2. Infer new attributes that can be reasonably deduced by combining existing attributes.

For example:
- If one attribute says "{person_name} studies at Stanford" and another says "{person_name} is majoring in Computer Science", 
  you might combine them as "{person_name} studies Computer Science at Stanford".
- If attributes mention "{person_name} grew up in Seattle" and "{person_name} moved to Boston for college", 
  you might infer "{person_name} relocated from the West Coast to the East Coast for higher education".

Please format your response EXACTLY as follows, with two clearly labeled sections:

COMBINED ATTRIBUTES:
- Attribute: {person_name} studies Computer Science at Stanford.
- Based on: Attribute 3, Attribute 7

- Attribute: {person_name} has been playing piano for over 10 years and performs in local venues.
- Based on: Attribute 12, Attribute 15

INFERRED ATTRIBUTES:
- Attribute: {person_name} relocated from the West Coast to the East Coast for higher education.
- Based on: Attribute 2, Attribute 9

- Attribute: {person_name} likely has an interest in both technology and music.
- Based on: Attribute 3, Attribute 12

Here are the attributes about {person_name}:

{'''
'''.join(formatted_attributes)}
"""
    
    # Call the model
    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=4096,
            temperature=1,
            top_p=1,
        )
        
        # Parse the response using a custom parser
        result = parse_combined_attributes_from_text(response.choices[0].message.content)
    finally:
        await client.close()
    
    # Add the actual attribute objects to the combined and inferred attributes
    for combined_attr in result["combined_attributes"]:
        source_attributes = []
        if "based_on" in combined_attr:
            # Extract attribute numbers from the "based_on" field
            attr_nums_upper = re.findall(r'Attribute\s+(\d+)', combined_attr["based_on"])
            attr_nums = re.findall(r'attribute\s+(\d+)', combined_attr["based_on"])
            attr_nums = list(set(attr_nums + attr_nums_upper))
            # print("----------------------------------------")
            # print("based_on:", combined_attr["based_on"])
            # print("attr_nums:", attr_nums)
            for num in attr_nums:
                try:
                    idx = int(num) - 1  # Convert to 0-indexed
                    if 0 <= idx < len(attributes_batch):
                        source_attributes.append(attributes_batch[idx])
                except ValueError:
                    continue
        # print("source_attributes:", source_attributes)
        combined_attr["source_attributes"] = source_attributes
    
    for inferred_attr in result["inferred_attributes"]:
        source_attributes = []
        if "based_on" in inferred_attr:
            # Extract attribute numbers from the "based_on" field
            attr_nums = re.findall(r'Attribute\s+(\d+)', inferred_attr["based_on"])
            for num in attr_nums:
                try:
                    idx = int(num) - 1  # Convert to 0-indexed
                    if 0 <= idx < len(attributes_batch):
                        source_attributes.append(attributes_batch[idx])
                except ValueError:
                    continue
        inferred_attr["source_attributes"] = source_attributes
    
    return result

def parse_combined_attributes_from_text(text):
    """
    Parse combined and inferred attributes from the model's text response.
    
    Args:
        text: The text response from the model
    
    Returns:
        Dictionary containing combined and inferred attributes
    """
    combined_attributes = []
    inferred_attributes = []
    
    # Determine which section we're in
    current_section = None
    current_attribute = {}
    
    # Split the text into lines
    lines = text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
        
        # Check for section headers
        if "COMBINED ATTRIBUTES" in line.upper():
            current_section = "combined"
            continue
        elif "INFERRED ATTRIBUTES" in line.upper():
            current_section = "inferred"
            continue
        
        # Remove leading dash or bullet point if present
        if line.startswith('- '):
            line = line[2:]
        elif line.startswith('• '):
            line = line[2:]
            
        # Check for attribute or based_on
        if line.lower().startswith('attribute:'):
            # If we already have an attribute in progress, save it
            if current_attribute and 'attribute' in current_attribute:
                if current_section == "combined":
                    combined_attributes.append(current_attribute)
                elif current_section == "inferred":
                    inferred_attributes.append(current_attribute)
                current_attribute = {}
            
            current_attribute['attribute'] = line[len('attribute:'):].strip()
        
        elif line.lower().startswith('based on:'):
            if 'attribute' in current_attribute:  # Only add if we have an attribute
                current_attribute['based_on'] = line[len('based on:'):].strip()
                
                # If we have both fields, add to appropriate list and reset
                if all(k in current_attribute for k in ['attribute', 'based_on']):
                    if current_section == "combined":
                        combined_attributes.append(current_attribute)
                    elif current_section == "inferred":
                        inferred_attributes.append(current_attribute)
                    current_attribute = {}
    
    # Add the last attribute if it wasn't added
    if current_attribute and 'attribute' in current_attribute:
        # Ensure all fields exist
        if 'based_on' not in current_attribute:
            current_attribute['based_on'] = ''
            
        if current_section == "combined":
            combined_attributes.append(current_attribute)
        elif current_section == "inferred":
            inferred_attributes.append(current_attribute)
    
    return {
        "combined_attributes": combined_attributes,
        "inferred_attributes": inferred_attributes
    }

async def process_attributes_in_parallel(attributes, model_name="gpt-4o", batch_size=50, person_name="Ben", max_concurrent=5, lang: Literal["zh", "eng"] = "zh"):
    """
    Process multiple batches of attributes concurrently to combine and infer new attributes.
    
    Args:
        attributes: List of attribute dictionaries
        batch_size: Number of attributes to process in each batch
        person_name: Name of the person the attributes are about
        max_concurrent: Maximum number of concurrent API calls
        lang: Language for prompts ("zh" or "eng")
        
    Returns:
        Dictionary containing original, combined, and inferred attributes
    """
    import asyncio
    
    # Create batches of attributes
    batches = [attributes[i:i+batch_size] for i in range(0, len(attributes), batch_size)]
    total_batches = len(batches)
    
    all_combined_attributes = []
    all_inferred_attributes = []
    
    # Process batches in chunks to control concurrency
    for i in range(0, len(batches), max_concurrent):
        # print(-1)
        current_chunk = batches[i:i+max_concurrent]
        current_tasks = []
        
        # Print progress
        print(f"Processing batches {i+1}-{min(i+max_concurrent, total_batches)} of {total_batches} ({(i+1)/total_batches*100:.1f}% - {min(i+max_concurrent, total_batches)/total_batches*100:.1f}%)")
        
        for batch in current_chunk:
            task = combine_and_infer_attributes_async(batch, person_name, model_name, lang)
            current_tasks.append(task)
        
        # Wait for all tasks in the current chunk to complete
        results = await asyncio.gather(*current_tasks)
        
        # Collect results
        for result in results:
            all_combined_attributes.extend(result["combined_attributes"])
            all_inferred_attributes.extend(result["inferred_attributes"])
        
        # Optional: add a small delay to avoid rate limiting
        await asyncio.sleep(0.1)
    
    # Return the final results
    return {
        "original_attributes": attributes,
        "combined_attributes": all_combined_attributes,
        "inferred_attributes": all_inferred_attributes
    }

# Function to run the async code from a synchronous context
def combine_and_infer_attributes_parallel(attributes, model_name="gpt-4o", batch_size=50, person_name="Ben", max_concurrent=5, lang: Literal["zh", "eng"] = "zh"):
    """
    Wrapper function to run the async attribute processing in a synchronous context.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(process_attributes_in_parallel(
            attributes, model_name, batch_size, person_name, max_concurrent, lang
        ))
        return result
    finally:
        # Ensure all pending tasks are completed before closing
        pending = asyncio.all_tasks(loop)
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()

async def generate_top_order_memories(attributes_data, model_name="gpt-4o-mini", target_count=20, person_name="Ben", max_concurrent=5, lang: Literal["zh", "eng"] = "zh"):
    """
    Generate top-order memories focused on specific events and life periods rather than general attributes.
    Handles large attribute sets by processing in chunks concurrently.
    """
    # Extract all attributes from the data
    all_attributes = []
    azure_api_version=os.environ['AZURE_API_VERSION']
    azure_endpoint = os.environ['AZURE_ENDPOINT']
    azure_api_key = os.environ['AZURE_OPENAI_API_KEY']
    # Add original attributes
    if "original_attributes" in attributes_data:
        for attr in attributes_data["original_attributes"]:
            all_attributes.append({
                "type": "Original Attribute",
                "id": attr.get("id", len(all_attributes) + 1),
                "content": attr["attribute"],
                "citation": attr.get("citation", "")
            })
    
    # Add combined attributes
    if "combined_attributes" in attributes_data:
        for attr in attributes_data["combined_attributes"]:
            all_attributes.append({
                "type": "Combined Attribute",
                "id": attr.get("id", len(all_attributes) + 1),
                "content": attr["attribute"],
                "citation": attr.get("based_on", "")
            })
    
    # Add inferred attributes
    if "inferred_attributes" in attributes_data:
        for attr in attributes_data["inferred_attributes"]:
            all_attributes.append({
                "type": "Inferred Attribute",
                "id": attr.get("id", len(all_attributes) + 1),
                "content": attr["attribute"],
                "citation": attr.get("based_on", "")
            })
    
    # Handle large attribute sets by chunking
    chunk_size = 100  # Process 100 attributes at a time
    all_memories = []
    
    # Create chunks of attributes
    chunks = [all_attributes[i:i+chunk_size] for i in range(0, len(all_attributes), chunk_size)]
    total_chunks = len(chunks)
    
    client = AsyncOpenAI(api_key=azure_api_key, base_url=azure_endpoint)
    
    # Process chunks in batches to control concurrency
    for i in range(0, len(chunks), max_concurrent):
        # print("1")
        current_batch = chunks[i:i+max_concurrent]
        tasks = []
        
        # Print progress
        print(f"Processing memory chunks {i+1}-{min(i+max_concurrent, total_chunks)} of {total_chunks} ({(i+1)/total_chunks*100:.1f}% - {min(i+max_concurrent, total_chunks)/total_chunks*100:.1f}%)")
        
        # Create tasks for each chunk in the current batch
        for chunk_idx, chunk in enumerate(current_batch):
            # Create a prompt for generating event-based memories from this chunk
            if lang == "zh":
                system_prompt = "你是一个根据属性资料创造有意义、事件关联记忆的助手"
                user_prompt = f"""
你负责根据提供的 {person_name} 属性建立与事件有关的记忆。

重要：请专注于 {person_name} 生命中具体的事件、经历与时间段，而非一般性的特质或性格描述。

好的范例（这些是虚构的，请勿直接使用）：
-"{person_name} 在 2016 年至 2020 年间就读于摩根镇高中，他是问答比赛队的一员，并因为他最喜欢的老师 Johnson 先生而对化学产生了浓厚兴趣。"
-"在 2023 年夏天，{person_name} 与朋友 Alex 和 Jordan 一起前往拉斯维加斯旅行，他们参加了一场音乐节，{person_name} 还在 21 点中赢了 200 美元。"
-"2022 年秋季，{person_name} 开始与张教授合作撰写第一篇重要的 AI 研究论文，主题聚焦于个人化 AI 模型，并于 2023 年 3 月发表。"

不佳范例（过于一般化、非事件性）：
-"{person_name} 喜欢玩像《Stardew Valley》这样的电子游戏。"
-"{person_name} 对 AI 研究有兴趣，并会与朋友合作。"
-"{person_name} 重视财务管理与预算控管。"

对每一段记忆请遵守以下要求：
1.聚焦于具体事件、时间段或经历
2.包含发生的时间、涉及的人物、具体活动内容等细节
3.尽量按照时间顺序整理记忆
4.引用该记忆所根据的属性

属性资料如下(第 {i+chunk_idx+1} 片段，共 {total_chunks} 片段中的一部分)

{json.dumps(chunk, indent=2, ensure_ascii=False)}

请用以下 JSON 数组格式回应，每个对象包含：
- "attribute"：具体的事件型记忆描述
- "citation"：这段记忆依据哪些属性（如："Original Attribute 5, Combined Attribute 2"）
- "timestamp"：已知的时间范围，例如 "2023 年夏天"、"2016-2020"、"2023 年 3 月" 等

请根据此片段的资料撰写最多 5 条高品质的事件型记忆。
"""
            else:  # eng
                system_prompt = "You are a helpful assistant that creates meaningful, event-based memories from attribute data."
                user_prompt = f"""
You are tasked with creating event-based memories about {person_name} based on the attributes provided.

IMPORTANT: Focus on specific events, experiences, and time periods in {person_name}'s life, NOT general traits or characteristics.

Good examples (These are made up, do not use them as examples):
- "{person_name} attended Morgantown High School from 2016 to 2020, where he was part of the quiz bowl team and developed a strong interest in chemistry thanks to his favorite teacher, Mr. Johnson."
- "During summer 2023, {person_name} traveled to Las Vegas with friends Alex and Jordan, where they attended a music festival and {person_name} won $200 at blackjack."
- "In Fall 2022, {person_name} began working on his first major AI research paper with Professor Zhang, focusing on personalized AI models, which was later published in March 2023."

Bad examples (too general, not event-based):
- "{person_name} enjoys playing video games like Stardew Valley."
- "{person_name} is interested in AI research and collaborates with friends."
- "{person_name} values financial management and budgeting."

For each memory:
1. Focus on specific events, time periods, or experiences
2. Include relevant details like when it happened, who was involved, and what specifically occurred
3. Organize memories chronologically when possible
4. Cite which attributes this memory is based on

Here are the attributes to work with (chunk {i+chunk_idx+1} of {total_chunks}):

{json.dumps(chunk, indent=2)}

Return your response as a JSON array of objects, each with:
- "attribute": The detailed event-based memory
- "citation": References to the attributes this is based on (e.g., "Original Attribute 5, Combined Attribute 2")
- "timestamp": Approximate time period if known (e.g., "Summer 2023", "2016-2020", "March 2023")

Create up to 5 high-quality, event-based memories from this chunk of attributes.
"""
            
            # Create a task for this chunk
            task = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=4096,
                top_p=1,
            )
            tasks.append(task)
        
        # Process all tasks in the current batch concurrently
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process the responses
        for response in responses:
            if isinstance(response, Exception):
                print(f"Error processing chunk: {response}")
                continue
                
            try:
                content = response.choices[0].message.content
                # Extract JSON from the response
                json_start = content.find('[')
                json_end = content.rfind(']') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    chunk_memories = json.loads(json_str)
                    all_memories.extend(chunk_memories)
                else:
                    # Fallback if JSON parsing fails
                    print(f"Warning: Could not parse JSON from a chunk. Skipping.")
                    print(f"Response content: {content}")
            except Exception as e:
                print(f"Error processing response: {e}")
        
        # Optional: add a small delay to avoid rate limiting
        await asyncio.sleep(0.1)
    
    # Close the client
    await client.close()
    
    # Sort memories by time_period if available
    all_memories.sort(key=lambda x: x.get("time_period", "Unknown"))
    
    # Return the top memories up to the target count
    return all_memories[:target_count]

def generate_top_order_memories_sync(attributes_data, model_name="gpt-4o-mini", target_count=20, person_name="Ben", max_concurrent=5, lang: Literal["zh", "eng"] = "zh"):
    """
    Wrapper function to run the async memory generation in a synchronous context.
    """
    import asyncio
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(generate_top_order_memories(
            attributes_data, model_name, target_count, person_name, max_concurrent, lang
        ))
        return result
    finally:
        # Ensure all pending tasks are completed before closing
        pending = asyncio.all_tasks(loop)
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()