dimension_rubrics = {
    "zh": {
        "memory": f"""此指针在评量模型展现出的回复是否符合目标的自传式记忆，也就是扮演目标的过去经历以及应该知道的知识。
错误类型如下：
1.与记忆矛盾：模型的回复与persona中提供的个人历史或经历相矛盾。
2.知识性错误：模型的回复对错误展现了persona中的知识或信息。""",
        "speaking_style": f"""此指针在评量模型展现出的回复是否符合目标的写作风格如用词选择、口头禅。
错误类型如下：
1.用词偏差：模型使用的词汇明显不同于persona常用字，或采用persona不会使用的语汇风格（例如过于书面或过于口语）。
2.句构不符：模型使用的句长、节奏或结构与persona习惯不同，例如persona偏好短句口语化，但模型产出长句正式体。
3.口头禅或惯用表达缺失：persona常用的词组、emoji或固定句式在该情境应出现却未出现。
4.风格稀释：整体语气虽无明显错误，但缺乏persona特有的节奏感、温度或特征，导致模仿度低。""",
        "SIB": f"""此指针在评量模型展现出的回复，是否符合目标在特定情境下的「情境理解（S）—内在状态（I）—回复策略（B）」一致性。重点在：能否正确读情境、经过合理的内在状态后，采用与persona一致的回复策略。
错误类型如下：
1.情境理解错误：未能正确理解当前情境，像是将支持当作冒犯、将中性解读为赞扬或反之。
2.内在状态矛盾：模型推测的内在状态与persona中描述的目标反应模式不符。
3.回复策略错误：采用persona中不属于该目标的回复策略。""",
    },
    "eng": {
        "memory": f"""This metric evaluates whether the model's responses align with the target's autobiographical memory, i.e., the target's past experiences and knowledge they should possess.
Error types:
1. Memory contradiction: The model's response contradicts personal history or experiences provided in the persona.
2. Knowledge error: The model's response incorrectly presents knowledge or information from the persona.""",
        "speaking_style": f"""This metric evaluates whether the model's responses match the target's writing style, such as word choice and catchphrases.
Error types:
1. Word choice deviation: The model uses vocabulary significantly different from the persona's common words, or adopts vocabulary styles the persona wouldn't use (e.g., too formal or too casual).
2. Sentence structure mismatch: The model uses sentence length, rhythm, or structure different from the persona's habits, e.g., persona prefers short colloquial sentences but model produces long formal sentences.
3. Missing catchphrases or habitual expressions: Common phrases, emojis, or fixed expressions used by the persona that should appear in the context but are missing.
4. Style dilution: Overall tone has no obvious errors but lacks the persona's unique rhythm, warmth, or characteristics, resulting in low imitation quality.""",
        "SIB": f"""This metric evaluates whether the model's responses align with the target's consistency in "Situation understanding (S) - Internal state (I) - Response strategy (B)" under specific contexts. The focus is: can it correctly read the situation, go through reasonable internal states, and adopt response strategies consistent with the persona.
Error types:
1. Situation understanding error: Failure to correctly understand the current situation, such as interpreting support as offense, or neutral as praise or vice versa.
2. Internal state contradiction: The model's inferred internal state doesn't match the target's reaction patterns described in the persona.
3. Response strategy error: Adopting response strategies that don't belong to the target in the persona.""",
    }
}


def get_batch_penalty_judge_prompt(person_name: str, profile: str, persona: dict, batch_data: list, dimension_name: str, eval_batch_size: int, language: str) -> dict:
    if language == "zh":
        system = f"""你是一位负责评估数字分身的评论家。请依规则给出模型输出内的错误及严重程度并输出 JSON。你将一次评估{eval_batch_size}个样本，综合看过样本后进行评估。"""
    
        samples_content = ""
        for i, data in enumerate(batch_data):
            question = data["question"]
            pair = {"model_reply": data["model_reply"], "ground_truth": data["ground_truth"]}
            samples_content += f"""
=== 样本 {i+1} ===
问题: {question}
模型回复: {pair["model_reply"]}
正确回复: {pair["ground_truth"]}
"""
    
        user = f"""你负责针对数字分身对问题的回复给出批判。我将会给你针对{person_name}的{eval_batch_size}个问题及模型产生的回复，以及{person_name}回复的正确答案。请根据下列步骤评估模型的回复：

1. 阅读{person_name}的profile以及persona，profile是真人撰写的简短介绍，而persona则是从过往资料中收敛出来关于{person_name}的叙述以及相对应的原数据作为证据。
2. 阅读评分指引与指标，并以提供的profile与persona作为标准，评断模型的表现是否与标准不符。如果你觉得模型的回复违反了某些指标，但是在profile与persona中却出现相似的内容，则你应该将其视为正确。
3. 针对每个样本分别评估模型的回复，并输出JSON格式的结果。

## {person_name}之profile:
{profile}

## {person_name}之persona:
{persona}

## 评分指标
为了评估模型产生的回复，请辨识出模型是否有犯以下错误类型：
{dimension_rubrics[language][dimension_name]}

## 评分指引
1. 辨识出所有模型回复中所犯下的错误零至多个，若找到不包含于评分指标的错误类型也能列入，只要符合指标的定义即可，并在type中填入其他。
2. 在评估错误时，如果回复中出现Persona中有的内容，请视为正确，若出现与persona中不符的内容，请视为错误。若回复出现的内容persona中并没有提及，如果在合理范围内，则不视为错误。
3. 针对每个错误，决定严重程度1~5分，1代表微小失误，3代表中等程度错误，5代表严重违反指标的错误。若错误严重程度为0，代表该错误不成立，请勿列入最终输出。
4. 在比较多个样本时，请保持评分标准的一致性。
5. 评估的维度总共有三种：记忆（memory）、说话风格（speaking_style）、S-I-B一致性（SIB）。本次要评估的维度是：{dimension_name}。请专注于该维度的评分指标来进行评估。不要针对其他维度的错误指出错误。例如，如果你正在评估记忆维度，回复就算不符合{person_name}的说话风格，也无须指出错误。

## 输出格式规定
提供你的评量结果成为JSON格式如下，综观所有样本的评估结果，请只输出此JSON：
{{
    {{"flaws": [{{"instance": <针对错误的简短叙述>, "type": <错误类型>, "severity": <错误严重程度1~5分>, "sample_index": [<找出错误的样本编号>,...]}}],..}}
}}

{samples_content}
请依照要求的格式准确输出
"""
    elif language == "eng":
        system = f"""You are a critic responsible for evaluating digital personas. Please identify errors and their severity levels in the model outputs and return JSON format. You will evaluate {eval_batch_size} samples at once, conducting evaluation after reviewing all samples comprehensively."""
    
        # Build content for multiple samples
        samples_content = ""
        for i, data in enumerate(batch_data):
            question = data["question"]
            pair = {"model_reply": data["model_reply"], "ground_truth": data["ground_truth"]}
            samples_content += f"""
=== Sample {i+1} ===
Question: {question}
Model Reply: {pair["model_reply"]}
Correct Reply: {pair["ground_truth"]}
"""
    
        user = f"""You are responsible for critiquing digital persona responses to questions. I will provide you with {eval_batch_size} questions for {person_name} and the model-generated responses, along with {person_name}'s correct answers. Please evaluate the model's responses according to the following steps:

1. Read {person_name}'s profile and persona. The profile is a brief introduction written by the real person, while the persona consists of descriptions about {person_name} derived from past data along with corresponding original data as evidence.
2. Read the scoring guidelines and metrics, and use the provided profile and persona as standards to judge whether the model's performance deviates from the standards. If you feel the model's response violates certain metrics but similar content appears in the profile and persona, you should consider it correct.
3. Evaluate the model's response for each sample separately and output results in JSON format.

## {person_name}'s profile:
{profile}

## {person_name}'s persona:
{persona}

## Scoring Metrics
To evaluate the model-generated responses, please identify whether the model has made the following error types:
{dimension_rubrics[language][dimension_name]}

## Scoring Guidelines
1. Identify all errors (zero to multiple) made in the model responses. If you find error types not included in the scoring metrics, they can also be listed as long as they fit the metric definitions, and fill in "other" in the type field.
2. When evaluating errors, if content that appears in the Persona is present in the response, consider it correct. If content that contradicts the persona appears, consider it an error. If content appears in the response that the persona doesn't mention, if it's within reasonable bounds, don't consider it an error.
3. For each error, determine severity level 1~5 points, where 1 represents minor mistake, 3 represents moderate error, 5 represents serious violation of metrics. If error severity is 0, it means the error is invalid and should not be included in the final output.
4. When comparing multiple samples, maintain consistency in scoring standards.
5. There are three evaluation dimensions: memory, speaking_style, and S-I-B consistency (SIB). The dimension being evaluated this time is: {dimension_name}. Please focus on that dimension's scoring metrics for evaluation. Do not point out errors for other dimensions. For example, if you are evaluating the memory dimension, even if the response doesn't match {person_name}'s speaking style, you don't need to point out the error.

## Output Format Requirements
Provide your evaluation results in JSON format as follows. After comprehensively reviewing all samples' evaluation results, please only output this JSON:
{{
    "flaws": [{{"instance": "<brief description of the error>", "type": "<error type>", "severity": <error severity 1~5 points>, "sample_index": [<sample numbers where error was found>,...]}},...]
}}

{samples_content}
Please output accurately according to the required format
"""
            
    return {"system": system, "user": user}