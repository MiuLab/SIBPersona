from dotenv import load_dotenv
load_dotenv()

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Literal
import anthropic
from openai import OpenAI
from together import Together
import os
import sys
import json
import numpy as np
import torch
from .memory_module import MemoryModule
from datetime import datetime
from openai import OpenAI
import copy
import google.auth
import langchain_google_vertexai
# For open-source model type hints
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

openai_api_key = os.environ.get('OPENAI_API_KEY', '')

anthropic_api_key = os.environ.get('ANTHROPIC_API_KEY', '')
deepseek_api_key = os.environ.get('DEEPSEEK_API_KEY', '')
together_api_key = os.environ.get('TOGETHER_API_KEY', '')

today_date = datetime.now().strftime("%Y-%m-%d")

# English prompts
STANDARD_SYSTEM_MESSAGE = "Today's date is {today_date}. You are a human being named {persona_name}. You are not an AI. Respond as yourself. You will first be given the topic of conversation, then any existing conversation history if there is any. Be sure to reply in the style of {persona_name}. Use the '<|msg|>' token to send mulitple messages at once if you wish. "
EXAMPLE_SYSTEM_MESSAGE = "Here is an example conversation between {persona_name} and another individual. This conversation is not relevant to the current conversation. Use this conversation to help aid you to emulate stylistically on how to communicate in your conversations. 'Me' is the user that you are imitating.\n\n [BEGIN EXAMPLE CONVERSATION]\n{examples}\n[END EXAMPLE CONVERSATION]"
MEMORY_SYSTEM_MESSAGE = "Here are some of your relevant memories + facts about yourself that may help you respond authentically. Pay careful attention to the date of the memories, as events have occurred in the past, and you should make reference to them in the appropriate time manner. \n\n[BEGIN MEMORIES]\n{memory_text}\n[END MEMORIES]"
FINAL_SYSTEM_MESSAGE = "Now you may begin the conversation."

# Chinese prompts (Simplified Chinese)
EXAMPLE_SYSTEM_MESSAGE_ZH = "以下是 {persona_name} 和另一个粉丝之间的留言回复实例。这段对话与当前对话无关。请使用这段对话来帮助你模仿 {persona_name} 的交流风格。\n\n [BEGIN EXAMPLE CONVERSATION]\n{examples}\n[END EXAMPLE CONVERSATION]"
MEMORY_SYSTEM_MESSAGE_ZH = "以下是一些与你相关的记忆与事实，可能有助于你做出更真实的回应。请特别留意这些记忆的日期，因为事件是发生在过去，你应该在合适的情境中提及它们。 \n\n[BEGIN MEMORIES]\n{memory_text}\n[END MEMORIES]"
STANDARD_SYSTEM_MESSAGE_ZH = "今天的日期是 {today_date}。你是一位名叫 {persona_name} 的人类。你不是人工智能。请以你自己的身份回应。你将先收到这次对话的主題，若有任何過往對話紀錄，也會一併提供。請務必以 {persona_name} 的風格來回應。\n這是你的基本資訊：{profile}。"

# RoleBench/RoleAgentBench specific prompts
ROLEBENCH_SYSTEM_PROMPT_ZH = "你是一位擅长模仿特定人物说话风格的专家，你需要扮演{person_name}，以该人物的口吻与行为习惯回复问题。"
ROLEBENCH_SYSTEM_PROMPT_ENG = "You are an expert in mimicking the speaking style of specific characters. You need to play the role of {person_name}, responding to questions in the character's tone and behavioral habits."

CHARACTEREVAL_SYSTEM_PROMPT_ZH = ""

ROLEBENCH_USER_PROMPT_ZH = """你将扮演{person_name}，根据下方的资讯回复问题。请注意以下事项：
# 目标
- 若历史记忆有明确用语习惯，请沿用。
- 只输出你要回复的文字，不要加其他说明或标题。
{profile_part}
{memory_part}
# 要回复的问题
{question}
"""

CHARACTEREVAL_USER_PROMPT_ZH = """你将扮演{person_name}進行對話。请注意以下事项：
# 目标
- 若历史记忆有明确用语习惯，请沿用。
- 只输出你要回复的文字，不要加其他说明或标题。
{profile_part}
{memory_part}
"""

ROLEBENCH_USER_PROMPT_ENG = """You will play the role of {person_name}. Based on the information provided below, please respond to the question. Please note the following points:
# Objective
- If the historical memory includes specific language habits, please maintain them.
- Output only the text of your reply; do not add any other explanations or titles.
{profile_part}
{memory_part}
# Question
{question}
"""


# List of supported open-source models
OPEN_SOURCE_MODELS = [
    "allenai/Olmo-3-7B-Instruct",
]

service_account = os.environ['google_type']
project_id = os.environ['google_project_id']
private_key_id = os.environ['google_private_key_id']
private_key = os.environ['google_private_key']
client_email = os.environ['google_client_email']
client_id = os.environ['google_client_id']
auth_uri = os.environ['google_auth_uri']
token_uri = os.environ['google_token_uri']
auth_provider_x509_cert_url = os.environ['google_auth_provider_x509_cert_url']
client_x509_cert_url = os.environ['google_client_x509_cert_url']
universe_domain = os.environ['google_universe_domain']
google_credentials, _ = google.auth.load_credentials_from_dict(
{
    'type': service_account,
    'project_id': project_id,
    'private_key_id': private_key_id,
    'private_key': private_key,
    'client_email': client_email,
    'client_id': client_id,
    'auth_uri': auth_uri,
    'token_uri': token_uri,
    'auth_provider_x509_cert_url': auth_provider_x509_cert_url,
    'client_x509_cert_url': client_x509_cert_url,
    'universe_domain': universe_domain
},
)

def is_open_source_model(model_name: str) -> bool:
    """Check if the model is an open-source model."""
    return any(open_model in model_name for open_model in OPEN_SOURCE_MODELS)


class BaseAgent(ABC):
    """
    Abstract base class for agents that imitate human behavior and content generation.
    This class defines the interface that all agent implementations must follow.
    """
    
    def __init__(self, model_name: str, impersonation_name: str, custom_model_path: str, adapter_path: Optional[str] = None, 
                 data_dir: str = "../data", lang: Literal["zh", "eng"] = "zh",
                 model_instance=None, tokenizer=None):
        """
        Initialize the base agent.
        
        Args:
            model_name (str): Name of the model to use
            impersonation_name (str): Name of the persona to impersonate
            custom_model_path (str): Path to custom model
            adapter_path (Optional[str]): Path to adapter
            data_dir (str): Path to data directory
            lang (Literal["zh", "eng"]): Language for prompts
            model_instance: Pre-loaded open-source model instance (optional)
            tokenizer: Pre-loaded tokenizer for open-source model (optional)
        """
        self.model_name = model_name
        self.lang = lang
        
        # Store open-source model instance and tokenizer
        self.model_instance = model_instance
        self.tokenizer = tokenizer
        
        self.custom_model = None
        self.custom_tokenizer = None
        self.impersonation_name = impersonation_name
        self.history = []
        self.verbose = False
        self.data_dir = data_dir
        
        # Load profile - support both old format and RoleBench format
        profile_path = f"{data_dir}/{self.impersonation_name}/profile.json"
        if os.path.exists(profile_path):
            with open(profile_path, 'r', encoding='utf-8') as f:
                basic_info = json.load(f)
            self.profile = basic_info.get("profile", "")
        else:
            self.profile = ""

    @abstractmethod
    def generate_response(self, input_text: str) -> str:
        """
        Generate a response based on the input text.
        
        Args:
            input_text (str): The input text to respond to
            
        Returns:
            str: The generated response
        """
        pass
    
    @abstractmethod
    def generate_response_impersona(self, history: List[Dict]) -> str:
        """
        Generate a response based on a provided conversation history without maintaining state.
        
        Args:
            history (List[Dict]): List of conversation messages in the format [{'role': 'user|assistant', 'content': str}]
            
        Returns:
            str: The generated response
        """
        pass
    
    def reset(self) -> None:
        """
        Reset the agent's state to initial conditions.
        """
        self.history = []

    def _run_inference(self, max_tokens: int = 180) -> str:
        """
        Runs inference over a specific model on the current history state
        """
        if self.verbose:  # Temporarily force logging
            print("\n=== SYSTEM PROMPT LOGGING ===")
            if self.history:
                print(f"System prompt: {self.history[0]['content']}")
            else:
                print("No system prompt found in history")
            print("=== END SYSTEM PROMPT LOGGING ===\n")
        
        try:
            
            # Check if using open-source model
            if is_open_source_model(self.model_name) and self.model_instance is not None and self.tokenizer is not None:
                return self._run_open_source_inference(max_tokens)
            
            elif 'o1' in self.model_name:
                messages = self.history.copy()
                if messages[0]['role'] == 'system':
                    messages[0]['role'] = 'user'
                client = OpenAI(api_key=openai_api_key)
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                )
                return response.choices[0].message.content
            elif 'gpt' in self.model_name or 'o3' in self.model_name:
                azure_api_key = os.environ.get('AZURE_OPENAI_API_KEY', '')
                azure_api_version = os.environ.get('AZURE_API_VERSION', '')
                azure_endpoint = os.environ.get('AZURE_ENDPOINT', '')
                client = OpenAI(api_key=azure_api_key, base_url=azure_endpoint)
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=self.history,
                )
                # print(response)
                return response.choices[0].message.content
            elif 'gemini' in self.model_name:
                client = langchain_google_vertexai.ChatVertexAI(
                    credentials=google_credentials,
                    location='global',
                    model=self.model_name,
                    project=project_id,
                )
                full_message = ""
                for msg in self.history:
                    full_message += f"{msg['content']}\n"
                response = client.invoke(full_message)
                return response.content
            elif 'deepseek' in self.model_name:
                client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=self.history,
                )
                return response.choices[0].message.content
            elif 'claude' in self.model_name:
                client = anthropic.Anthropic(api_key=anthropic_api_key)
                # Convert messages to Anthropic format
                anthropic_messages = [
                    {
                        "role": msg["role"],
                        "content": [{"type": "text", "text": msg["content"]}]
                    }
                    for msg in self.history
                ]
                if self.history[0]['role'] == 'system':
                    system_message = self.history[0]['content']
                else:
                    system_message = ''
                response = client.messages.create(
                    model=self.model_name,
                    messages=anthropic_messages,
                    max_tokens=4096,
                    system=system_message
                )
                return response.content[0].text
            else:
                client = Together(api_key=together_api_key)
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=self.history,
                    temperature=0.8,
                    top_p=0.9,
                )
                return response.choices[0].message.content
        except Exception as e:
            error_message = str(e)
            if "Incorrect API key provided" in error_message:
                return "Error: Invalid API key"
            elif "Rate limit reached" in error_message:
                return "Error: Rate limit exceeded"
            else:
                return f"Error: {error_message}"

    def _run_open_source_inference(self, max_tokens: int = 180) -> str:
        """
        Run inference using open-source models via transformers.
        
        Args:
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            str: Generated response text
        """
        try:
            if "Olmo" in self.model_name:
                input_ids = self.tokenizer.apply_chat_template(
                    self.history,
                    tokenize=True,
                    return_tensors="pt",
                    add_generation_prompt=True
                )
                attention_mask = torch.ones_like(input_ids)
                if torch.cuda.is_available():
                    input_ids = input_ids.to("cuda")
                    attention_mask = attention_mask.to("cuda")
                
                output = self.model_instance.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    temperature=0.6,
                    max_new_tokens=max_tokens,
                    top_p=0.95,
                    do_sample=True,
                    use_cache=False,
                )
                generated_text = self.tokenizer.decode(output[0, input_ids.shape[-1]:], skip_special_tokens=True)
                if generated_text.startswith("assistant\n"):
                    generated_text = generated_text[len("assistant\n"):]
                return generated_text.strip()
            
            
            else:
                return f"Error: Unsupported open-source model: {self.model_name}"
                
        except Exception as e:
            return f"Error in open-source inference: {str(e)}"

    def parse_conversation(self, conversation_text: str, admin_name: str = "admin") -> List[Dict]:
        """
        Parse a Discord-like conversation into the format needed for the agent's history.
        
        Args:
            conversation_text (str): Raw conversation text with <|> delimiters
            admin_name (str): Name of the admin/assistant in the conversation
            
        Returns:
            List[Dict]: Formatted conversation history
        """
        # Split the text into parts using the delimiter
        parts = conversation_text.split('<|>')
        
        # Extract metadata (parts before the first timestamp)
        metadata = []
        conversation_start = 0
        for i, part in enumerate(parts):
            if '[' in part and ']' in part:  # Found first timestamp
                conversation_start = i
                break
            # Extract the metadata value after the colon
            if ':' in part:
                metadata.append(part.strip())
        
        # Combine metadata into system message
        metadata_text = "\n".join(metadata)
        
        # Format the conversation similar to the training examples
        conversation_lines = []
        
        # Process the actual conversation messages
        for part in parts[conversation_start:]:
            # Skip empty parts
            if not part.strip():
                continue
                
            # Parse the part
            try:
                # Extract username from the format [datetime] username: message
                message_parts = part.split(': ', 1)
                if len(message_parts) < 2:
                    continue  # Skip parts that don't have the expected format
                
                header = message_parts[0]
                message = message_parts[1].strip()
                
                # Replace admin_name with "Me" in the header
                if admin_name in header:
                    header = header.replace(admin_name, "Me")
                
                conversation_lines.append(f"{header}: {message}")
                
            except Exception as e:
                print(f"Error parsing part: {part}")
                continue
        
        # Format the conversation as a single string
        conversation_text = "\n".join(conversation_lines)
        
        # Create a system message with metadata
        system_message = f"Here is a conversation topic that you can begin conversation about with people. Make sure to respond to current conversation, and only use this as a guideline if there hasn't been any conversation yets. [BEGIN TOPIC METADATA]\n{metadata_text}\n[END TOPIC METADATA]."
        
        # Create a user message with the conversation and ending with "Me: " to prompt the model to respond
        user_message = f"Current conversation:\n{conversation_text}\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Me: "
        
        # Return the formatted history
        return [
            {'role': 'system', 'content': system_message},
            {'role': 'user', 'content': user_message}
        ]


class BasicAgent(BaseAgent):
    def __init__(self, model_name: str, impersonation_name: str, user_name: str, example_module = None, 
                 memory_module: Optional[MemoryModule] = None, custom_model_path: Optional[str] = None, 
                 adapter_path: Optional[str] = None, data_dir: str = "../data", lang: Literal["zh", "eng"] = "zh",
                 model_instance=None, tokenizer=None):
        super().__init__(model_name, impersonation_name, custom_model_path, adapter_path=adapter_path, 
                         data_dir=data_dir, lang=lang, model_instance=model_instance, tokenizer=tokenizer)
        self.example_module = example_module
        self.memory_module = memory_module
        self.impersonation_name = impersonation_name
        self.user_name = user_name
        self.query = ""
        
        if self.lang == "zh":
            self.system_message = STANDARD_SYSTEM_MESSAGE_ZH.format(persona_name=self.impersonation_name, today_date=today_date, profile=self.profile)
        else:
            self.system_message = STANDARD_SYSTEM_MESSAGE.format(persona_name=self.impersonation_name, today_date=today_date)
        
        self.history = [{'role': 'system', 'content': self.system_message}]
    
    def _get_system_message(self) -> str:
        curr_system_message = self.system_message
        if self.example_module:
            example_conversation = self.example_module.search_conversation_store(self.user_name)
            if self.lang == "zh":
                curr_system_message = curr_system_message + "\n\n" + EXAMPLE_SYSTEM_MESSAGE_ZH.format(persona_name=self.impersonation_name, examples=example_conversation)
            else:
                curr_system_message = curr_system_message + "\n\n" + EXAMPLE_SYSTEM_MESSAGE.format(persona_name=self.impersonation_name, examples=example_conversation)
        if self.memory_module:
            history_text = ""
            memorie_summary, top, second, first = self.memory_module.search_memory(self.query)
        
        curr_system_message = curr_system_message
        
        return curr_system_message, memorie_summary, top, second, first
        
    def generate_response(self, input_text: str) -> str:
        # Dynamically changing the system message to include the example conversation
        self.history.append({'role': 'user', 'content': input_text + "\n Me: "})
        
        if self.history and self.history[0]['role'] == 'system':
            self.history[0]['content'] = self._get_system_message()
        print(self.history)
        response = self._run_inference()
        self.history.append({'role': 'assistant', 'content': response})
        return response
    
    def generate_response_impersona(self, article, query) -> str:
        self.query = query
        original_history = copy.deepcopy(self.history)
        original_system_message = copy.deepcopy(self.system_message)
        system_prompt, memorie_summary, top, second, first = self._get_system_message()
        persona = []
        for second_memory in second:
            single_memory = {}
            single_memory['attribute'] = second_memory['attribute']
            single_memory['source_attributes'] = []
            for attr in second_memory['source_attributes']:
                single_source_attr = {}
                single_source_attr['attribute'] = attr['attribute']
                single_source_attr['citation'] = attr['citation']
                single_memory['source_attributes'].append(single_source_attr)
            persona.append(single_memory)
        
        if self.lang == "zh":
            input_text = f"""你将扮演{self.impersonation_name}，根据记忆与历史互动，回复粉丝的留言。请注意以下事项：
# 目标
- 若历史记忆有明确用语习惯（称呼、emoji），请沿用。
- 只输出你要回给粉丝的单段『作者回复文字』，不要加其他说明或标题。
- 可以参考以下提供给你的记忆后回覆粉丝留言。
# memory
{persona}
# 当前文章资讯
{article}
# 要回覆的粉丝留言
{query}
"""
        else:
            input_text = f"""You will play the role of {self.impersonation_name}, responding to fan comments based on memories and historical interactions. Please note the following:
# Objective
- If historical memory includes specific language habits (forms of address, emojis), please maintain them.
- Output only the text of your reply to the fan; do not add any other explanations or titles.
- You may refer to the provided memories when responding to fan comments.
# Memory
{persona}
# Current Article Information
{article}
# Fan Comment to Reply To
{query}
"""

        self.history.append({'role': 'user', 'content': input_text})
        if self.history and self.history[0]['role'] == 'system':
            self.history[0]['content'] = system_prompt
        response = self._run_inference()
        self.history = original_history
        self.system_message = original_system_message

        return response, memorie_summary, top, second, first


class RoleBenchAgent(BaseAgent):
    """
    Agent specialized for RoleBench/RoleAgentBench datasets.
    Uses IMPersona's hierarchical memory system for persona-aware generation.
    """
    
    def __init__(self, model_name: str, impersonation_name: str, memory_module: Optional[MemoryModule] = None, 
                 custom_model_path: Optional[str] = None, adapter_path: Optional[str] = None, 
                 data_dir: str = "../data", lang: Literal["zh", "eng"] = "zh", 
                 dataset: str = "RoleBench", profile: Optional[str] = None,
                 model_instance=None, tokenizer=None, anonymous: bool = False):
        super().__init__(model_name, impersonation_name, custom_model_path, adapter_path=adapter_path, 
                         data_dir=data_dir, lang=lang, model_instance=model_instance, tokenizer=tokenizer)
        self.memory_module = memory_module
        self.dataset = dataset
        
        # Allow external profile override (for RoleBench compatibility)
        if profile is not None:
            self.profile = profile
        
        # Set system prompt based on language
        if self.lang == "zh":
            if dataset == "CharacterEval":
                self.system_message = CHARACTEREVAL_SYSTEM_PROMPT_ZH
            else:
                self.system_message = ROLEBENCH_SYSTEM_PROMPT_ZH.format(person_name=self.impersonation_name)
        else:
            self.system_message = ROLEBENCH_SYSTEM_PROMPT_ENG.format(person_name=self.impersonation_name)
        if anonymous:
            self.system_message = self.system_message.replace(self.impersonation_name, "<anonymous_character>")
        self.history = [{'role': 'system', 'content': self.system_message}]
    
    def _format_memory_for_prompt(self, memorie_summary: str, top_order_memories: List[Dict], second_order_memories: List[Dict]) -> str:
        """Format retrieved memories for the prompt."""
        if not second_order_memories:
            return ""
        
        persona = {}
        persona["summary"] = memorie_summary
        persona["top_level_memories"] = []
        persona["second_level_memories"] = []
        for top_memory in top_order_memories:
            persona["top_level_memories"].append(top_memory.get('attribute', ''))
        for second_memory in second_order_memories:
            single_memory = {}
            single_memory['attribute'] = second_memory.get('attribute', '')
            single_memory['source_attributes'] = []
            for attr in second_memory.get('source_attributes', []):
                single_source_attr = {}
                single_source_attr['attribute'] = attr.get('attribute', '')
                single_source_attr['citation'] = attr.get('citation', '')
                single_memory['source_attributes'].append(single_source_attr)
            persona["second_level_memories"].append(single_memory)
        
        return str(persona)
    
    def generate_response(self, input_text: str) -> str:
        """Basic response generation."""
        self.history.append({'role': 'user', 'content': input_text})
        response = self._run_inference()
        self.history.append({'role': 'assistant', 'content': response})
        return response
    
    def generate_response_impersona(self, question: str, source_role: str = "", pure_question: str = "", anonymous: bool = False) -> tuple:
        """
        Generate a response for RoleBench/RoleAgentBench/CharacterEval using IMPersona's memory system.
        
        Args:
            question (str): The question to respond to
            source_role (str): The role asking the question (for RoleAgentBench)
            pure_question (str): The pure question without instruction (for RoleAgentBench)
            anonymous (bool): Whether to use anonymous character names
            
        Returns:
            tuple: (response, memory_summary, top_memories, second_order_memories, first_order_memories, memory_text)
        """
        original_history = copy.deepcopy(self.history)
        
        # Retrieve memories using IMPersona's hierarchical memory system
        memorie_summary, top, second, first = "", [], [], []
        if self.memory_module:
            memorie_summary, top, second, first = self.memory_module.search_memory(pure_question)
        
        # Format memories for prompt
        memory_text = self._format_memory_for_prompt(memorie_summary, top, second)
        
        # Build profile part
        profile_part = ""
        if self.profile:
            if self.lang == "zh":
                profile_part = f"# {self.impersonation_name}的profile\n{self.profile}"
            else:
                profile_part = f"# Profile of {self.impersonation_name}\n{self.profile}"
        
        # Build memory part
        memory_part = ""
        if memory_text:
            if self.lang == "zh":
                memory_part = f"""# 检索到的相关记忆
{memory_text}
"""
            else:
                memory_part = f"""# relevant memories retrieved through memory system
{memory_text}
"""
        
        # Build the full question with source role if applicable
        full_question = question
        if source_role and self.dataset == "RoleAgentBench":
            full_question = f"{source_role}: {question}"
        
        # CharacterEval uses special message format
        if self.dataset == "CharacterEval":
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
        
            user_prompt = CHARACTEREVAL_USER_PROMPT_ZH.format(
                    person_name=self.impersonation_name,
                    profile_part=profile_part,
                    memory_part=memory_part
                )
            if anonymous:
                user_prompt = user_prompt.replace(self.impersonation_name, "<anonymous_character>")
            messages, query = concat_messages(make_inputs(question), self.impersonation_name, self.system_message, user_prompt)
            self.history = messages + [{"role": "user", "content": query}]
        else:
            # Build user prompt for RoleBench/RoleAgentBench
            if self.lang == "zh":
                user_prompt = ROLEBENCH_USER_PROMPT_ZH.format(
                    person_name=self.impersonation_name,
                    profile_part=profile_part,
                    memory_part=memory_part,
                    question=full_question
                )
            else:
                user_prompt = ROLEBENCH_USER_PROMPT_ENG.format(
                    person_name=self.impersonation_name,
                    profile_part=profile_part,
                    memory_part=memory_part,
                    question=full_question
                )
            if anonymous:
                user_prompt = user_prompt.replace(self.impersonation_name, "<anonymous_character>")
            # Set up history for this request
            self.history = [
                {'role': 'system', 'content': self.system_message},
                {'role': 'user', 'content': user_prompt}
            ]
        
        # print(self.history)
        for tries in range(5):
            response = self._run_inference()
            if response is None or (response is not None and response.startswith("Error")):
                print(f"Retrying {tries+1}/5...")
            else:
                break
        if response.startswith("Error"):
            response = ""
        # print(f"system prompt: {self.system_message}")
        # print(f"history: {self.history}")
        # print(f"response: {response}")
        # Restore original history
        self.history = original_history
        if anonymous and self.impersonation_name and response is not None:
            response = response.replace("<anonymous_character>", self.impersonation_name)
        return response, memorie_summary, top, second, first, memory_text