import os
import re
from functools import lru_cache
from typing import Dict, List

from dotenv import load_dotenv

load_dotenv()

cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES")
if cuda_visible:
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_ID = os.getenv("LLM_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", "900"))
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.25"))
TOP_P = float(os.getenv("LLM_TOP_P", "0.90"))


def strip_thinking(text: str) -> str:
    """
    Some reasoning models may emit hidden-looking thinking tags.
    We remove them from user-visible output.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


class LocalLLM:
    def __init__(self, model_id: str = DEFAULT_MODEL_ID):
        self.model_id = model_id
        print(f"Loading local LLM: {self.model_id}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )

        self.model.eval()

    def build_prompt(self, messages: List[Dict[str, str]]) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            rendered = ""
            for message in messages:
                rendered += f"{message['role'].upper()}:\n{message['content']}\n\n"
            rendered += "ASSISTANT:\n"
            return rendered

    @torch.inference_mode()
    def generate(self, messages: List[Dict[str, str]], max_new_tokens: int = MAX_NEW_TOKENS) -> str:
        prompt = self.build_prompt(messages)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=24000,
        )

        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True if TEMPERATURE > 0 else False,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=1.05,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return strip_thinking(output_text)


@lru_cache(maxsize=1)
def get_local_llm() -> LocalLLM:
    return LocalLLM()
