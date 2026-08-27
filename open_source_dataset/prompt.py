from typing import Literal
def get_recognize_module_prompt(person_name:str, data: dict, lang: Literal["eng", "zh"]) -> str:
    if lang == "zh":
        system_prompt = "你是一个专门辨识文字中所含信息的模型"
        recognize_module_prompt = f"""
你是一位擅长分析对话并分类的专家。

我将提供你{person_name}所参与的一段对话。你的任务是：

1. 分析这段对话，并且判断其中的信息，跟 {person_name} 的哪方面相关，这些方面请从以下选择：
- Memory：有关{person_name}的事实信息，有如长期记忆里面的语意记忆，包含{person_name}的背景，他应该掌握的知识等等(ex. {person_name}出生自上海、{person_name}专长变身术、唐三藏是{person_name}的师父等等)
- Speaking_Style：有关{person_name}的说话风格、口头禅(ex. {person_name}经常用「哈！」来当作说话开头)
- SIB (Situation-Internal state-Behavior)：有关{person_name}在对话时的行为模式(ex. {person_name}面对敌人会用嘲讽的语气、{person_name}面对称赞会谦虚响应谢谢等等)
2. 你可以选择多个方面，也可以都不选择，请根据对话内容做出最合适的判断。
3. 输出请按照以下格式，填入你的分析以及该对话内容包含的信息类别：
```json
{{"analysis": "...", "categorys": ["category1", "category1"]}}
```
补充：若对话内容没有明显的相关方面，categorys则输出空list即可，但是还是需要有analysis。

以下为一个例子：
对话内容：
{{
    "diag_id": 0,
    "dialogue": [
        {{
            "role": "祖师",
            "content": "你姓什么?叫什么名字?原来你是天生地长。我看你相貌举止像个猢狲， 你就姓孙，法名孙悟空，好么?",
        }},
        {{
            "role": "孙悟空",
            "content": "好，好，好，弟子今日才有了名姓，就叫 孙 — — 悟 — — 空!我叫孙悟空了!我叫孙悟空了!"
        }}
    ]
}}

输出：
```json
{{"analysis": "这段对话展示了{person_name}的经历和说话风格。{person_name}的法名叫做孙悟空，本来没有名姓，属于Memory的范畴。此外，{person_name}面对师傅会自称弟子，是Speaking_Style的范畴", "categorys": ["Memory", "Speaking_Style"]}}
```

以下是对话内容：
{data}

请依照要求的格式准确输出
"""
    elif lang == "eng":
        system_prompt = "You are a model specialized in identifying information contained in text."
        recognize_module_prompt = f"""
You are an expert in dialogue analysis and categorization.
I will provide you with a dialogue that {person_name} participated in. Your task is:
1. Analyze the dialogue and determine which aspects of {person_name} the information relates to. Choose from the following:
- Memory: Factual information about {person_name}, similar to long-term semantic memory, including {person_name}'s background, knowledge they should possess, etc. (e.g., {person_name} was born in Shanghai; {person_name}'s specialty is transformation magic; Bob is {person_name}'s master.)
- Speaking_Style: Information related to {person_name}'s way of speaking or verbal habits (e.g., {person_name} often begins sentences with “Ha!”)
- SIB (Situation–Internal state–Behavior): Information about {person_name}'s behavioral patterns in conversations (e.g., {person_name} uses a mocking tone when facing enemies; {person_name} responds humbly when praised.)
2. You may select multiple categories, or none if appropriate. Please make the most suitable judgment based on the dialogue.
3. Output your response in the following format, filling in your analysis and the categories contained in the dialogue:

```json
{{"analysis": "...", "categorys": ["category1", "category2"]}}
```
Note: If the dialogue do not clearly correspond to any category, output an empty list for "categorys", but you must still provide an analysis.
Below is an example:
Dialogue and comment thread:
{{
    "diag_id": 0,
    "dialogue": [
        {{
            "role": "Master",
            "content": "What is your surname? What is your name? So you were born of heaven and earth. From your appearance and behavior, you look like a monkey. You shall take the surname Sun, and your Dharma name shall be Sun Wukong. How about that?"
        }},
        {{
            "role": "Sun Wukong",
            "content": "Good, good, good! Today I finally have a surname and a name. I will be called Sun—Wu—Kong! My name is Sun Wukong! My name is Sun Wukong!"
        }}
    ]
}}
Output:
{{"analysis": "This dialogue shows {person_name}'s personal history and speaking style. {person_name} receives the Dharma name 'Sun Wukong' and previously had no surname, which falls under Memory. In addition, {person_name} refers to himself as a disciple when speaking to his master, which falls under Speaking_Style.", "categorys": ["Memory", "Speaking_Style"]}}
Below is the dialogue:
{data}

Please output strictly following the required format.
"""
    return system_prompt, recognize_module_prompt

def get_extract_memory_prompt(person_name:str, data: list | str, lang: str) -> str:
    if lang == "zh":
        system_prompt = "你是一个专门萃取文字中包含特定对象信息的模型"
        extract_memory_prompt = f"""
你是一位擅长从对话中提取语意记忆的专家。

我将提供你{person_name}所参与的一段对话。你的任务是：

1. 从这段对话中，提取文字中含有{person_name}的语意记忆。
2. 语意记忆是包含{person_name}相关的可验证事实，例如{person_name}的出生地在上海；以及{person_name}所掌握的领域知识，例如蜀国跟魏国是敌人。
3. 在撰写记忆时，请透过文字内容收敛并推理出零至多个claim，每个claim说明单一的概念，并且用简短的文字表达该claim，输出的claim个数没有限制。
4. 输出请按照以下格式，填入对话包含的记忆点于claim字段，以及你对于此分析的信心分数0~1分：

```json
{{"items": [{{"claim": "claim 1", "confidence": <your evidence score>}}, {{"claim": "claim 2", "confidence": <your evidence score>}}]}}
```
补充：若对话内容没有明显可抽取的claim，items字段直接输出空list即可。

以下为一个例子：
对话内容：
{{
    "diag_id": 0,
    "dialogue": [
        {{
            "role": "祖师",
            "content": "你姓什么?叫什么名字?原来你是天生地长。我看你相貌举止像个猢狲， 你就姓孙，法名孙悟空，好么?",
        }},
        {{
            "role": "孙悟空",
            "content": "好，好，好，弟子今日才有了名姓，就叫 孙 — — 悟 — — 空!我叫孙悟空了!我叫孙悟空了!"
        }}
    ]
}}

输出：
```json
{{"items": [{{"claim": "{person_name}长相像是猴子，原本没有名字", "confidence": 0.8}}, {{"claim": "{person_name}法名被祖师取名叫孙悟空", "confidence": 0.9}}]}}
```

以下是对话内容：
{data}

请依照要求的格式准确输出
"""
    elif lang == "eng":
        system_prompt = "You are a model specialized in identifying information contained in text."
        extract_memory_prompt = f"""You are an expert skilled at extracting semantic memory from a dialogue.

I will provide you with a segment of dialogue involving {person_name}. Your task is:

1. Extract the semantic memory related to {person_name} from the text of this dialogue.
2. Semantic memory includes verifiable facts related to {person_name}, such as that {person_name}'s birthplace is Shanghai; and the domain knowledge possessed by {person_name}, such as that the Shu Kingdom and the Wei Kingdom were enemies.
3. When formulating the memory, please use the text content to converge and infer zero to multiple claims. Each claim should describe a single concept and express that claim in concise language. There is no limit to the number of claims to be outputted.
4. Output must strictly follow the format below, filling the `claim` field with the memory points contained in the dialogue and providing a confidence score of 0 to 1 for your analysis:

```json
{{"items": [{{"claim": "claim 1", "confidence": <your evidence score>}}, {{"claim": "claim 2", "confidence": <your evidence score>}}]}}
```
Supplement: If there are no obvious claims to extract from the dialogue content, the items field should output an empty list.

The following is an example:
Dialogue Content:
{{
    "diag_id": 0,
    "dialogue": [
        {{
            "role": "Master",
            "content": "What is your surname? What is your name? You were born from Heaven and Earth. I see your appearance and demeanor resemble a macaque, so your surname shall be Sun, and your religious name shall be Sun Wukong, does that sound good?",
        }},
        {{
            "role": "Sun Wukong",
            "content": "Good, good, good! Disciple finally has a name and surname today. I'll be called Sun—Wu—Kong! My name is Sun Wukong! My name is Sun Wukong!"
        }}
    ]
}}

Output:
```json
{{"items": [{{"claim": "{person_name} looks like a monkey and originally did not have a name", "confidence": 0.8}}, {{"claim": "{person_name}'s religious name was given by the Patriarch as Sun Wukong", "confidence": 0.9}}]}}
```
Dialogue Content:
{data}
Please strictly output according to the required format."""
    return system_prompt, extract_memory_prompt

def get_extract_speaking_style_prompt(person_name:str, data: list | str, lang: str) -> str:
    if lang == "zh":
        system_prompt = "你是一个专门萃取文字中包含特定对象信息的模型"
        extract_speaking_style_prompt = f"""
你是一位擅长从对话中提取一个人说话风格的专家。
我将提供你{person_name}所参与的一段对话。你的任务是：

1. 从这段对话中，提取文字中含有{person_name}在的说话风格。
2. 说话风格是包含{person_name}的用词选择、口头禅等，例如{person_name}经常用「哈哈」作为说话的结尾。
3. 在撰写说话风格时，请透过文字内容收敛并推理出零至多个claim，每个claim说明单一的概念，并且用简短的文字表达该知识点，输出的知识点个数没有限制。
4. 输出请按照以下格式，填入该对话包含的说话风格于claim字段，以及你对于此分析的信心分数0~1分：
```json
{{"items": [{{"claim": "claim 1", "confidence": <your evidence score>}}, {{"claim": "claim 2", "confidence": <your evidence score>}}]}}
```
补充：若对话内容没有明显可抽取的claim，items字段直接输出空list即可。

以下为一个例子：
对话内容：
{{
    "diag_id": 0,
    "dialogue": [
        {{
            "role": "祖师",
            "content": "你姓什么?叫什么名字?原来你是天生地长。我看你相貌举止像个猢狲， 你就姓孙，法名孙悟空，好么?",
        }},
        {{
            "role": "孙悟空",
            "content": "好，好，好，弟子今日才有了名姓，就叫 孙 — — 悟 — — 空!我叫孙悟空了!我叫孙悟空了!"
        }}
    ]
}}

输出：
```json
{{"items": [{{"claim": "{person_name}面对师傅会称自己为弟子", "confidence": 0.8}}, {{"claim": "{person_name}说话会重复段落", "confidence": 0.7}}]}}
```

以下是对话内容：
{data}

请依照要求的格式准确输出"""
    elif lang == "eng":
        system_prompt = "You are a model specialized in identifying information contained in text."
        extract_speaking_style_prompt = f"""You are an expert skilled at extracting the speaking style of an individual from a dialogue.
I will provide you with a segment of dialogue involving {person_name}. Your task is:

1. Extract the speaking style of {person_name} contained within the text of this dialogue.
2. Speaking style includes {person_name}'s word choices, verbal tics (catchphrases), etc. For example, {person_name} frequently uses "Haha" at the end of a sentence.
3. When formulating the speaking style, please use the text content to converge and infer zero to multiple claims. Each claim should describe a single concept and express that knowledge point in concise language. There is no limit to the number of knowledge points (claims) to be outputted.
4. Output must strictly follow the format below, filling the `claim` field with the speaking styles contained in the dialogue and providing a confidence score of 0 to 1 for your analysis:
```json
{{"items": [{{"claim": "claim 1", "confidence": <your evidence score>}}, {{"claim": "claim 2", "confidence": <your evidence score>}}]}}
```
Supplement: If there are no obvious claims to extract from the dialogue content, the items field should output an empty list.

The following is an example: 
Dialogue Content:
{{
    "diag_id": 0,
    "dialogue": [
        {{
            "role": "Master",
            "content": "What is your surname? What is your name? You were born from Heaven and Earth. I see your appearance and demeanor resemble a macaque, so your surname shall be Sun, and your religious name shall be Sun Wukong, does that sound good?",
        }},
        {{
            "role": "Sun Wukong",
            "content": "Good, good, good! Disciple finally has a name and surname today. I'll be called Sun—Wu—Kong! My name is Sun Wukong! My name is Sun Wukong!"
        }}
    ]
}}

Output:
```json
{{"items": [{{"claim": "{person_name} refers to himself as 'disciple' when addressing the Master", "confidence": 0.8}}, {{"claim": "{person_name} repeats phrases or sentences when speaking", "confidence": 0.7}}]}}
```

Dialogue Content:
{data}

Please strictly output according to the required format."""
    return system_prompt, extract_speaking_style_prompt

def get_extract_SIB_triplet_prompt(person_name:str, data: list | str, lang: str) -> str:
    if lang == "zh":
        system_prompt = "你是一个专门萃取文字中包含特定对象信息的模型"
        extract_SIB_triplet_prompt = f"""
你是一位熟悉「Cognitive-Affective Processing System（CAPS）」并擅长从对话中推测{person_name}内在想法及其行为模式的研究助理。

【任务目标】
我将提供你{person_name}所参与的一段对话。你的任务是：
请你依 CAPS 理论，从对话中抽取出{person_name}的 S-I-B 三元组（Situation-Internal state-Behavior），以协助后续让 LLM 模仿{person_name}在对话时的决策与语气。

【CAPS 理论简述】
人在面对特定情境（Situation）时，会启动内在处理历程（Internal state），包含但不限于：
1) 认知：对情境的诠释、注意焦点、语义/社会线索的编码
2) 情感：情绪、心情
3) 目标/价值：当下欲达成之沟通目标与长期价值观
4) 期待/信念：对互动结果、他人反应或社群规范的预期
内在处理历程会导致可观察的响应（Behavior），例如{person_name}实际采用的回复策略和语气。

【在本任务中的 S-I-B 定义】
- Situation（S）：对话的情境（必要时可附上极简的对话摘要，仅限有助判读的最小必要信息）。
- Internal state（I）：依据 {person_name}与其他人的对话，对 {person_name} 在当下可能的内在处理历程进行「可检证、可对齐证据」的简洁推论。
- Behavior（B）：{person_name} 回复时采取的回复策略。

【重要规则】
1. 纵观所有对话后可以产生 0~多个 SIB。
2. 信心程度：对每条 SIB 给出 0~1 的 confidence，反映你对该 SIB 的整体把握度。
3. 输出请按照以下格式：
```json
{{"items": [{{"situation": "situation 1", "internal_state": "internal state 1", "behavior": "behavior 1", "confidence": <your evidence score>}}, {{"situation": "situation 2", "internal_state": "internal state 2", "behavior": "behavior 2", "confidence": <your evidence score>}}]}}
```
补充：若对话内容没有明显可抽取的SIB triplet，items字段直接输出空list即可。

以下为一个例子：
对话内容：
{{
    "diag_id": 31,
    "dialogue": [
        {{
            "role": "祖师",
            "content": "你这猢狲，夜深不去睡觉，到我这里来干什么?",
        }},
        {{
            "role": "孙悟空",
            "content": "师父白天打我三下，关闭中门，明明是要我三更时候，走后门前来学道。师父，师父，您就教我些真本事吧。"
        }},
        {{
            "role": "祖师",
            "content": "念你诚心好学，就传些道术给你。我这里有隐身潜形的变化之术，分为两种。 一种是三十六般变化， 一种是七十二般变化，你愿学哪一种?",
        }},
        {{
            "role": "孙悟空",
            "content": "弟子愿意学多的，师父教我七十二般变化吧。"
        }},
    ]
}}

输出：
```json
{{"items": [{{"situation": "祖师说{person_name}半夜不睡觉在做甚么", "internal_state": "{person_name}觉得无辜，自己只是按照祖师的指示", "behavior": "{person_name}提及祖师白天打了自己三下，要求晚上要来学道", "confidence": 0.8}}, {{"situation": "祖师说变化之术，分为两种，三十六般变化和七十二般变化", "internal_state": "{person_name}好学，觉得自己要多学些", "behavior": "说出自己的想法表达想学七十二般变化", "confidence": 0.7}}]}}
```

以下是对话内容：
{data}

请依照要求的格式准确输出"""
    elif lang == "eng":
        system_prompt = "You are a model specialized in identifying information contained in text."
        extract_SIB_triplet_prompt = f"""
You are a Research Assistant familiar with the "Cognitive-Affective Processing System (CAPS)" and skilled at inferring the internal thoughts and behavioral patterns of {person_name} from a dialogue.

【Task Objective】
I will provide you with a segment of dialogue involving {person_name}. Your task is:
Based on the CAPS theory, extract the S-I-B triplets (Situation-Internal state-Behavior) of {person_name} from the dialogue to assist a subsequent LLM in imitating {person_name}'s decision-making and tone during the conversation.

【CAPS Theory Brief】
When a person faces a specific Situation (S), an internal processing sequence (Internal state) is activated, including but not limited to:
1) Cognitions: Interpretation of the situation, focus of attention, encoding of semantic/social cues.
2) Affect: Emotions, mood.
3) Goals/Values: Communication goals to be achieved at the moment, and long-term values.
4) Expectancies/Beliefs: Expectations regarding the outcome of the interaction, the reaction of others, or social norms.
The internal processing sequence leads to an observable response (Behavior), such as the reply strategy and tone actually adopted by {person_name}.

【Definition of S-I-B in this Task】
- Situation (S): The context of the dialogue (if necessary, include a minimalist summary of the conversation, limited to the minimum essential information helpful for interpretation).
- Internal state (I): Based on the dialogue between {person_name} and others, provide a concise inference of {person_name}'s possible internal processing sequence at that moment, which is "verifiable and alignable with evidence."
- Behavior (B): The reply strategy adopted by {person_name} when responding.

【Crucial Rules】
1. Observing the entire dialogue can yield 0 to multiple SIBs.
2. Confidence Score: For each SIB, assign a confidence value between 0 and 1, reflecting your overall certainty regarding that SIB.
3. Output must strictly follow the format below:
```json
{{"items": [{{"situation": "situation 1", "internal_state": "internal state 1", "behavior": "behavior 1", "confidence": <your evidence score>}}, {{"situation": "situation 2", "internal_state": "internal state 2", "behavior": "behavior 2", "confidence": <your evidence score>}}]}}
```
Supplement: If there are no obvious SIB triplets to extract from the dialogue, the items field should output an empty list.

The following is an example:
Dialogue Content:
{{
    "diag_id": 31,
    "dialogue": [
        {{
            "role": "Master",
            "content": "You wretched monkey, why have you come to me instead of sleeping late at night?",
        }},
        {{
            "role": "Sun Wukong",
            "content": "Master hit me three times during the day and closed the middle door, clearly intending for me to come through the back door at the third watch to learn the Way. Master, Master, please teach me some true skills."
        }},
        {{
            "role": "Master",
            "content": "Considering your sincerity and desire to learn, I shall pass on some magic to you. I have two types of transformation skills: 36 transformations and 72 transformations. Which one do you wish to learn?",
        }},
        {{
            "role": "Sun Wukong",
            "content": "Disciple wishes to learn the greater one. Master, please teach me the 72 transformations."
        }},
    ]
}}

Output:
```json
{{"items": [{{"situation": "Patriarch asks what {person_name} is doing up late instead of sleeping", "internal_state": "{person_name} feels wronged, believing he is merely following the Patriarch's instructions", "behavior": "{person_name} references the Patriarch hitting him three times during the day and asking him to come at night to learn the Way", "confidence": 0.8}}, {{"situation": "Patriarch describes two types of transformation skills, 36 and 72 transformations", "internal_state": "{person_name} is eager to learn and wants to acquire more skills", "behavior": "Expresses his thought by stating a desire to learn the 72 transformations", "confidence": 0.7}}]}}
```

Dialogue Content:
{data}

Please strictly output according to the required format."""
    return system_prompt, extract_SIB_triplet_prompt



def get_roleagentbench_response_prompt(person_name:str, source_role: str, question: str, language: str, persona: dict = None, profile: str = None, raw_dialogue: str = None) -> str:
    if source_role == "":
        source_role_part = ""
    else:
        source_role_part = f"{source_role}: "
    if language == "zh":
        response_system_prompt = f"""你是一位擅长模仿特定人物说话风格的专家，你需要扮演{person_name}，以该人物的口吻与行为习惯回复问题。""" 
        profile_part = ""
        persona_part_up = ""
        persona_part_down = ""
        raw_dialogue_part = ""
        if profile:
            profile_part = f"""# {person_name}的profile
{profile}"""
        if persona:
            persona_part_up = f"""- 以下提供给你三个面向的persona，以及{person_name}的过往对话纪录，可以参考后回复问题。
- 三个面向的persona分别为：Memory（记忆）、Speaking Style（说话风格）、SIB_triplet（情境-内在状态-行为）。Memory指的是{person_name}的个人信息与领域知识，Speaking Style指的是{person_name}的用词选择及口头禅，SIB_triplet指的是{person_name}在面对特定Situation时的Internal state，以及最后的Behavior策略。"""
            persona_part_down = f"""# persona及过往回复纪录
{persona}
"""
        if raw_dialogue:
            raw_dialogue_part = f"""# {person_name}的历史互动纪录
{raw_dialogue}"""
        response_user_prompt = f"""你将扮演{person_name}，根据下方的資訊回复问题。请注意以下事项：
# 目标
- 若历史记忆有明确用语习惯，请沿用。
- 只输出你要回复的文字，不要加其他说明或标题。
{raw_dialogue_part}{persona_part_up}
{profile_part}
{persona_part_down}
# 要回复的问题
{source_role_part}{question}"""
    elif language == "eng":
        response_system_prompt = f"""You are an expert in mimicking the speaking style of specific characters. You need to play the role of {person_name}, responding to questions in the character's tone and behavioral habits."""
        profile_part = ""
        raw_dialogue_part = ""
        persona_part_up = ""
        persona_part_down = ""
        if profile:
            profile_part = f"""# Profile of {person_name}
{profile}"""
        if persona:
            persona_part_up = f"""# Persona and Past Interaction Records
{persona}"""
            persona_part_down = f"""- The three aspects of persona are: Memory, Speaking Style, and SIB_triplet. Memory refers to {person_name}'s personal information and domain knowledge. Speaking Style refers to {person_name}'s word choices and catchphrases. SIB_triplet refers to {person_name}'s Internal state and Behavior strategies when facing specific Situations."""
        if raw_dialogue:
            raw_dialogue_part = f"""# Historical Interaction Records of {person_name}
{raw_dialogue}"""
        response_user_prompt = f"""You will play the role of {person_name}. Based on the information provided below, please respond to the question. Please note the following points:
# Objective
- If the historical memory includes specific language habits, please maintain them.
- Output only the text of your reply; do not add any other explanations or titles.
{raw_dialogue_part}{persona_part_up}
{profile_part}
{persona_part_down}

# Question
{source_role_part}{question}"""

    return response_system_prompt, response_user_prompt


def get_charactereval_response_prompt(person_name:str, persona: dict = None, profile: str = None, raw_dialogue: str = None) -> str:
    response_system_prompt = f"""""" 
    profile_part = ""
    persona_part_up = ""
    persona_part_down = ""
    raw_dialogue_part = ""
    persona_count = 0
    memory_str = ""
    speaking_style_str = ""
    SIB_str = ""
    memory_str_detail = ""
    speaking_style_str_detail = ""
    SIB_str_detail = ""
    persona_count_to_zh_map = {1: "一", 2: "兩", 3: "三"}
    if "Memory" in persona:
        persona_count += 1
        memory_str = "Memory（记忆）"
        memory_str_detail = f"Memory指的是{person_name}的个人信息与领域知识，"
    if "Speaking_Style" in persona:
        persona_count += 1
        speaking_style_str = "Speaking Style（说话风格）"
        speaking_style_str_detail = f"Speaking Style指的是{person_name}的用词选择及口头禅"
    if "SIB" in persona:
        persona_count += 1
        SIB_str = "、SIB_triplet（情境-内在状态-行为）"
        SIB_str_detail = f"，SIB_triplet指的是{person_name}在面对特定Situation时的Internal state，以及最后的Behavior策略"
    if profile:
        profile_part = f"""# {person_name}的profile
{profile}"""
    if persona:
        persona_part_up = f"""- 以下提供给你{persona_count_to_zh_map[persona_count]}个面向的persona，以及{person_name}的过往对话纪录，可以参考后回复對話。
- {persona_count_to_zh_map[persona_count]}个面向的persona分别为：{memory_str}、{speaking_style_str}{SIB_str}。{memory_str_detail}{speaking_style_str_detail}{SIB_str_detail}。"""
        persona_part_down = f"""# persona及过往回复纪录
{persona}
"""
    if raw_dialogue:
        raw_dialogue_part = f"""# {person_name}的历史互动纪录
{raw_dialogue}"""
    response_user_prompt = f"""你将扮演{person_name}進行對話。请注意以下事项：
# 目标
- 若历史记忆有明确用语习惯，请沿用。
- 只输出你要回复的文字，不要加其他说明或标题。
{raw_dialogue_part}{persona_part_up}
{profile_part}
{persona_part_down}"""

    return response_system_prompt, response_user_prompt