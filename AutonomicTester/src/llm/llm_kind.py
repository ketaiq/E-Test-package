from enum import Enum


class LLMKind(Enum):
    GPT3turbo = "OpenAI GPT-3.5 Turbo"
    GPT3FT = "OpenAI GPT-3.5 Fine-tuned"
    GPT4 = "OpenAI GPT-4"
    GPT4turbo = "OpenAI GPT-4 Turbo"
    GPT4o = "OpenAI GPT-4o"
    Gemini = "Google Gemini"
    LLama2_7B = "Meta LLama2 7B"
    LLama3_8B = "Meta LLama3 8B"
    LLama3_70B = "Meta LLama3 70B"
    LLama3_1_8B = "Meta LLama3.1 8B"
    LLama3_3_70B = "Meta LLama3.3 70B"
    Gemma2_27B = "Google Gemma2 27B"
    Deepseek_R1_70B = "Deepseek R1 70B"

    @classmethod
    def generate_help_msg(cls) -> str:
        description_map = {
            "GPT3turbo": "GPT3: require an API key on the environment path",
            "GPT3FT": "GPT3FT: fine-tuned model with Defects4J dataset",
            "GPT4": "GPT4: require an API key on the environment path",
            "GPT4turbo": "GPT4turbo: require an API key on the environment path",
            "GPT4o": "GPT4o: require an API key on the environment path",
            "Gemini": "Gemini: require an API key on the environment path and a VPN connected to the USA",
            "LLama2_7B": "LLama2_7B: run locally but require 8 GB of RAM",
            "LLama3_8B": "LLama3_8B: run locally but require 8 GB of RAM",
            "LLama3_70B": "LLama3_70B: run locally but require a GPU with 40 GB of RAM",
            "LLama3_1_8B": "LLama3_1_8B: run locally but require 8 GB of RAM",
            "LLama3_3_70B": "LLama3_3_70B: run locally but require 40 GB of RAM",
            "Gemma2_27B": "Gemma2_27B: run locally",
            "Deepseek_R1_70B": "Deepseek_R1_70B: run locally but require 40 GB of RAM"
        }
        return "; ".join(f"{description_map[llm.name]}" for llm in cls)

    def get_intenal_model_name(self) -> str:
        # ft:gpt-3.5-turbo-0125:personal:defects4j-atest:9gx7pSLJ train with Q3 different answer between buggy and fixed 5% size
        # ft:gpt-3.5-turbo-0125:personal:defects4j-atest:9gYItPd8 train with Q3 same 10%
        # ft:gpt-3.5-turbo-0125:personal:defects4j-20:9kSLJleM latest fine-tuned model with 20 samples from Defects4J per scenario using prompt version 4 (Jul 13)
        model_name_map = {
            "GPT3turbo": "gpt-3.5-turbo",
            # "GPT3FT": "ft:gpt-3.5-turbo-0125:personal:defects4j-atest:9gx7pSLJ", # raw prompting
            "GPT3FT": "ft:gpt-3.5-turbo-0125:personal:rag-20250207-032318:Ay9Qrpr3", # with RAG
            "GPT4": "gpt-4",
            "GPT4turbo": "gpt-4-turbo",
            "GPT4o": "gpt-4o",
            "LLama2_7B": "llama2",
            "LLama3_8B": "llama3",
            "LLama3_70B": "llama3:70b",
            "LLama3_1_8B": "llama3.1",
            "LLama3_3_70B": "llama3.3",
            "Gemma2_27B": "gemma2:27b",
            "Deepseek_R1_70B": "deepseek-r1:70b"
        }
        if self.name in model_name_map:
            return model_name_map[self.name]
        else:
            raise ValueError(f"Enum member {self} not found in Ollama models.")

    def get_context_limit(self) -> int:
        model_context_limit_map = {
            "LLama3_8B": 8192,
            "LLama3_1_8B": 8192,
            "LLama3_70B": 8192,
            "GPT3turbo": 16385,
            "GPT3FT": 16385,
            "GPT4turbo": 128000,
            "GPT4o": 128000,
            "Gemma2_27B": 8192,
        }
        if self.name in model_context_limit_map:
            return model_context_limit_map[self.name]
        else:
            raise ValueError(f"LLM {self.name} not supported!")

    def get_hf_model_name(self):
        model_name_map = {
            "LLama3_8B": "meta-llama/Meta-Llama-3-8B",
            "LLama3_70B": "meta-llama/Meta-Llama-3-70B",
            "LLama3_1_8B": "meta-llama/Llama-3.1-8B",
            "LLama3_3_70B": "meta-llama/Llama-3.3-70B-Instruct",
            "Deepseek_R1_70B": "deepseek-ai/DeepSeek-R1"
        }
        if self.name in model_name_map:
            return model_name_map[self.name]
        else:
            raise ValueError(f"Enum member {self} not found in HuggingFace models.")

    def is_llama_model(self):
        return (
            self is LLMKind.LLama3_8B
            or self is LLMKind.LLama3_70B
            or self is LLMKind.LLama3_1_8B
            or self is LLMKind.LLama3_3_70B
            or self is LLMKind.Deepseek_R1_70B
        )

    def is_gpt_model(self):
        return (
            self is LLMKind.GPT3FT
            or self is LLMKind.GPT3turbo
            or self is LLMKind.GPT4
            or self is LLMKind.GPT4o
            or self is LLMKind.GPT4turbo
        )