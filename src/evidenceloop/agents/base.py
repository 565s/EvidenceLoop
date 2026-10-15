import re
from abc import ABC, abstractmethod
from typing import Any

from evidenceloop.logging import logger

BaseChatModel = Any


class BaseAgent(ABC):
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.system_prompt = self._get_system_prompt()
        self.llm = self._init_llm()

    def extract_json(self, text):
        match = re.search(r"\{.*\}", text, re.S)
        return match.group(0) if match else ""

    def fix_unescaped_quotes(self, s):
        s = re.sub(r'""([^"]+)":', r'"\\"\1\\":', s)
        s = re.sub(r'\[""([^"]+)":', r'["\\"\1\\":', s)
        s = re.sub(r"[\x00-\x1F\x7F]", "", s)
        return s

    @abstractmethod
    def _init_llm(self) -> BaseChatModel:
        pass

    def _get_system_prompt(self) -> str:
        raise NotImplementedError("The subclass must implement this method")

    def _invoke_llm(self, llm: BaseChatModel, prompt: str) -> str:
        try:
            messages = [("system", self.system_prompt), ("user", prompt)]
            response = llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            logger.error(f"LLM璋冪敤澶辫触: {str(e)}")
            return ""

    def call_llm(self, prompt: str) -> str:
        return self._invoke_llm(self.llm, prompt)
