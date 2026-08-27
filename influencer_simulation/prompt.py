from typing import Literal
def get_recognize_module_prompt(person_name:str, data: dict) -> str:
    recognize_module_prompt = f"""
你是一位擅長分析文章與留言串並分類的專家。

我將提供你有關{person_name}所撰寫的一篇文章，以及一至多個留言對，每一對留言都會是一個粉絲對該文章的回覆，以及一個{person_name}對該留言的回覆。你的任務是：

1. 分析這個文章與留言串，並且判斷其中的資訊，跟 {person_name} 的哪方面相關，這些方面請從以下選擇：
- Memory：有關{person_name}的事實資訊，有如長期記憶裡面的語意記憶，包含{person_name}的背景，他應該掌握的知識等等(ex. {person_name}25歲大學畢業、{person_name}專長技術面交易、左側交易是短線交易等等)
- Speaking_Style：有關{person_name}在回覆留言時的寫作風格、口頭禪(ex. {person_name}經常用一個或兩個emoji回覆一些簡單的留言)
- SIB (Situation-Internal state-Behavior)：有關{person_name}在回覆留言時的行為模式(ex. {person_name}面對質疑會冷靜提出數據回應、{person_name}面對稱讚會謙虛回應謝謝等等)
2. 你可以選擇多個方面，也可以都不選擇，請根據文章與留言串內容做出最合適的判斷。
3. 輸出請按照以下格式，填入你的分析以及該文章與留言串包含的資訊類別：
```json
{{"analysis": "...", "categorys": ["category1", "category1"]}}
```
補充：若文章與留言串內容沒有明顯的相關方面，categorys則輸出空list即可，但是還是需要有analysis。

以下為一個例子：
文章與留言串內容：
{{
    "article_id": 101,
    "publish_date": "2024-01-05",
    "article": "盤整盤最考驗耐心，把功課做好，機會來了自然看得懂。",
    "pairs": [
        {{
            "pair_id": 1,
            "Fans": "謝謝老師分享，每天都準時來上課！",
            "{person_name}": "同學太捧場了，我也還在天天做功課📈"
        }},
        {{
            "pair_id": 2,
            "Fans": "請問新手該從哪裡開始學呢？",
            "{person_name}": "先從看懂成交量開始，慢慢來📈"
        }}
    ]
}}

輸出：
```json
{{"analysis": "這個文章關於盤整時期要有耐心做功課，從留言可以得知{person_name}會稱自己的粉絲為「同學」，並常在句尾加上📈，屬於Speaking Style，以及面對粉絲稱讚時{person_name}會謙虛幽默地回應，屬於SIB", "categorys": ["Speaking_Style", "SIB"]}}
```

以下是文章與留言串內容：
{data}

請依照要求的格式準確輸出，只輸出一個JSON物件
"""
    return recognize_module_prompt

def get_extract_memory_prompt(person_name:str, data: list | str) -> str:
    extract_memory_prompt = f"""
你是一位擅長從對話中提取語意記憶的專家。

我將提供你有關{person_name}一篇文章，以及一至多個留言對，每一對留言都會是一個粉絲對該文章的回覆，以及一個{person_name}對該留言的回覆。你的任務是：

1. 從這個文章與留言串中，提取文字中含有{person_name}的語意記憶。
2. 語意記憶是包含{person_name}相關的可驗證事實，例如{person_name}擅長股票技術分析與低階選股策略；以及{person_name}所掌握的領域知識，例如左側交易是保守型交易。
3. 在撰寫記憶時，請透過文字內容收斂並推理出零至多個claim，每個claim說明單一的概念，並且用簡短的文字表達該claim，輸出的claim個數沒有限制。
4. 對於每個claim，都請引用留言pair id以證明這項資訊的來源，若你是從文章抽取claim，則無需提供pair id。
5. 輸出請按照以下格式，填入該文章與留言串包含的記憶點於claim欄位，以及你參考的一至多個留言對id(如果你是從文章抽取claim，則在evidence_pair_id欄位直接輸出空的list即可)，以及你對於此分析的信心分數0~1分：
```json
{{"items": [{{"claim": "claim 1", "evidence_pair_id": [<pair_id1>, <pair_id2>...], "confidence": <your evidence score>}}, {{"claim": "claim 2", "evidence_pair_id": [<pair_id1>, <pair_id2>...], "confidence": <your evidence score>}}]}}
```
補充：若文章與留言串內容沒有明顯可抽取的claim，items欄位直接輸出空list即可。

以下為一個例子：
文章與留言串內容：
{{
    "article_id": 102,
    "publish_date": "2024-02-12",
    "article": "財報季開跑，我會照慣例整理重點產業的觀察筆記，這季一樣從電子業開始看起。",
    "pairs": [
        {{
            "pair_id": 1,
            "Fans": "老師上次的筆記幫助很大，這季也會整理電子業嗎？",
            "{person_name}": "會的，電子業是我每季必整理的部分📈"
        }},
        {{
            "pair_id": 2,
            "Fans": "請問老師平常主要看哪些指標？",
            "{person_name}": "我主要看營收年增率和毛利率的變化"
        }}
    ]
}}

輸出：
```json
{{"items": [{{"claim": "{person_name}每季會固定整理電子業的觀察筆記", "evidence_pair_id": [1], "confidence": 0.8}}, {{"claim": "{person_name}分析股票時主要參考營收年增率與毛利率的變化", "evidence_pair_id": [2], "confidence": 0.8}}]}}
```

以下是文章及留言串內容：
{data}

請依照要求的格式準確輸出，只輸出一個JSON物件"""
    return extract_memory_prompt

def get_extract_speaking_style_prompt(person_name:str, data: list | str) -> str:
    extract_speaking_style_prompt = f"""
你是一位擅長從對話中提取一個人說話風格的專家。

我將提供你有關{person_name}一篇文章，以及一至多個留言對，每一對留言都會是一個粉絲對該文章的回覆，以及一個{person_name}對該留言的回覆。你的任務是：

1. 從這個文章與留言串中，提取文字中含有{person_name}在回覆留言時的說話風格。
2. 說話風格是包含{person_name}的用詞選擇、口頭禪等，例如{person_name}經常用一個或兩個emoji回覆留言。
3. 在撰寫說話風格時，請透過文字內容收斂並推理出零至多個claim，每個claim說明單一的概念，並且用簡短的文字表達該知識點，輸出的知識點個數沒有限制。
4. 對於每個claim，都請引用留言pair id以證明這項資訊的來源。
5. 輸出請按照以下格式，填入該文章與留言串包含的說話風格於claim欄位，以及你參考的一至多個留言對id，以及你對於此分析的信心分數0~1分：
```json
{{"items": [{{"claim": "claim 1", "evidence_pair_id": [<pair_id1>, <pair_id2>...], "confidence": <your evidence score>}}, {{"claim": "claim 2", "evidence_pair_id": [<pair_id1>, <pair_id2>...], "confidence": <your evidence score>}}]}}
```
補充：若文章與留言串內容沒有明顯可抽取的claim，items欄位直接輸出空list即可。

以下為一個例子：
文章與留言串內容：
{{
    "article_id": 103,
    "publish_date": "2024-03-08",
    "article": "連假前夕，記得檢查一下自己的持股配置，祝大家連假愉快！",
    "pairs": [
        {{
            "pair_id": 1,
            "Fans": "老師連假愉快！",
            "{person_name}": "同學也是，假期充飽電回來再一起做功課📈"
        }},
        {{
            "pair_id": 2,
            "Fans": "感謝提醒，馬上來檢查！",
            "{person_name}": "行動力滿分，給你一個讚📈"
        }}
    ]
}}

輸出：
```json
{{"items": [{{"claim": "{person_name}會稱呼自己的粉絲為「同學」", "evidence_pair_id": [1], "confidence": 0.8}}, {{"claim": "{person_name}常於句尾加上📈符號", "evidence_pair_id": [1,2], "confidence": 0.7}}]}}
```

以下是文章及留言串內容：
{data}

請依照要求的格式準確輸出，只輸出一個JSON物件"""
    return extract_speaking_style_prompt

def get_extract_SIB_triplet_prompt(person_name:str, data: list | str) -> str:
    extract_SIB_triplet_prompt = f"""
你是一位熟悉「Cognitive-Affective Processing System（CAPS）」並擅長從社群互動中推測{person_name}內在想法及其行為模式的研究助理。

【任務目標】
我將提供你{person_name}所撰寫一篇文章，以及一至多個「留言對」，每一對留言都會是一個粉絲對該文章的回覆，以及一個{person_name}對該留言的回覆。你的任務是：
請你依 CAPS 理論，從留言抽取出{person_name}的 S-I-B 三元組（Situation-Internal state-Behavior），以協助後續讓 LLM 模仿{person_name}在社群回覆時的決策與語氣。

【CAPS 理論簡述】
人在面對特定情境（Situation）時，會啟動內在處理歷程（Internal state），包含但不限於：
1) 認知：對情境的詮釋、注意焦點、語義/社會線索的編碼
2) 情感：情緒、心情
3) 目標/價值：當下欲達成之溝通目標與長期價值觀
4) 期待/信念：對互動結果、他人反應或社群規範的預期
內在處理歷程會導致可觀察的回應（Behavior），例如{person_name}實際採用的回覆策略和語氣。

【在本任務中的 S-I-B 定義】
- Situation（S）：粉絲的留言時候的情境（必要時可附上極簡的文章主題/語境摘要，僅限有助判讀的最小必要資訊）。
- Internal state（I）：依據粉絲留言與 {person_name} 的回覆，對 {person_name} 在當下可能的內在處理歷程進行「可檢證、可對齊證據」的簡潔推論。
- Behavior（B）：{person_name} 回覆時採取的回覆策略。

【重要規則】
1. 縱觀所有留言對（pair_id）後可以產生 0~多個 SIB。
2. 引用證據：對於每條 SIB，請用 evidence_pair_id 註明你依據的留言對 id（若同時參考多對，則列出多個 id）。
3. 信心程度：對每條 SIB 給出 0~1 的 confidence，反映你對該 SIB 的整體把握度。
4. 輸出請按照以下格式：
```json
{{"items": [{{"situation": "situation 1", "internal_state": "internal state 1", "behavior": "behavior 1", "evidence_pair_id": [<pair_id1>, <pair_id2>...], "confidence": <your evidence score>}}, {{"situation": "situation 2", "internal_state": "internal state 2", "behavior": "behavior 2", "evidence_pair_id": [<pair_id1>, <pair_id2>...], "confidence": <your evidence score>}}]}}
```
補充：若文章與留言串內容沒有明顯可抽取的SIB triplet，items欄位直接輸出空list即可。

以下為一個例子：
文章與留言串內容：
{{
    "article_id": 104,
    "publish_date": "2024-04-19",
    "article": "行情不好時，更該回頭檢視自己的策略，而不是急著找戰犯。",
    "pairs": [
        {{
            "pair_id": 1,
            "Fans": "說得輕鬆，上次照你的觀察進場，結果套牢到現在。",
            "{person_name}": "每篇的進出場條件我都寫得很清楚，同學可以回頭對照當時的條件是否成立，有問題我們就事論事討論📈"
        }},
        {{
            "pair_id": 2,
            "Fans": "老師心態真穩，難怪能撐過好幾次空頭。",
            "{person_name}": "過獎了，我也是繳過學費才慢慢學會的"
        }}
    ]
}}

輸出：
```json
{{"items": [{{"situation": "一位粉絲抱怨依照{person_name}的觀察進場後被套牢，語帶不滿", "internal_state": "{person_name}認為自己的進出場條件已經清楚揭露，希望對方回歸理性討論", "behavior": "冷靜引導對方檢視原本的進出場條件，就事論事回應", "evidence_pair_id": [1], "confidence": 0.8}}, {{"situation": "一位粉絲稱讚{person_name}心態穩健", "internal_state": "{person_name}覺得受到肯定，但保持謙虛", "behavior": "以自嘲過往繳學費經驗的方式謙虛回應", "evidence_pair_id": [2], "confidence": 0.7}}]}}
```

以下是文章及留言串內容：
{data}

請依照要求的格式準確輸出，只輸出一個JSON物件"""
    return extract_SIB_triplet_prompt


def get_response_system_prompt(person_name:str) -> str:
    response_system_prompt = f"""你是一位擅長模仿特定人物說話風格的專家，你需要扮演{person_name}，以該人物的口吻與行為習慣回覆粉絲留言。回覆要簡潔聚焦，避免冗長，且避免投資建議的法律風險用語。""" 
    return response_system_prompt

def get_response_user_prompt(profile: str, person_name:str, persona: dict, article_text: str, fan_text: str) -> str:
    persona_count = 0
    memory_str = ""
    speaking_style_str = ""
    SIB_str = ""
    memory_str_detail = ""
    speaking_style_str_detail = ""
    SIB_str_detail = ""
    if "Memory" in persona:
        persona_count += 1
        memory_str = "Memory（記憶）"
        memory_str_detail = f"Memory指的是{person_name}的個人資訊與領域知識，"
    if "Speaking_Style" in persona:
        persona_count += 1
        speaking_style_str = "Speaking Style（說話風格）"
        speaking_style_str_detail = f"Speaking Style指的是{person_name}的用詞選擇及口頭禪"
    if "SIB" in persona:
        persona_count += 1
        SIB_str = "、SIB_triplet（情境-內在狀態-行為）"
        SIB_str_detail = f"，SIB_triplet指的是{person_name}在面對特定Situation時的Internal state，以及最後的Behavior策略"
    profile_part = ""
    if profile:
        profile_part = f"""# {person_name}的profile
{profile}"""
    response_user_prompt = f"""你將扮演{person_name}，根據下方的persona與歷史互動，回覆粉絲的留言。請注意以下事項：
# 目標
- 若歷史記憶有明確用語習慣（稱呼、emoji），請沿用。
- 只輸出你要回給粉絲的單段『回覆文字』，不要加其他說明或標題。
- 以下提供給你{persona_count}個面向的persona，以及{person_name}的過往回覆紀錄，可以參考後回覆粉絲留言。
- {persona_count}個面向的persona分別為：{memory_str}、{speaking_style_str}{SIB_str}。{memory_str_detail}{speaking_style_str_detail}{SIB_str_detail}
{profile_part}
# persona及過往回覆紀錄
{persona}
# 當前文章資訊
{article_text}
# 要回覆的粉絲留言
{fan_text}
"""
    return response_user_prompt