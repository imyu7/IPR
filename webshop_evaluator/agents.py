"""
エージェント管理モジュール

異なるタイプのエージェント（OpenAI、HuggingFace）を統一的に作成・管理するための
Factory patternを実装したモジュールです。
"""

import os
from typing import Any, Dict, Union
from abc import ABC, abstractmethod
from dotenv import load_dotenv

from eval_agent.agents.openai_lm_agent import OpenAILMAgent
from eval_agent.agents.huggingface_agent import HuggingFaceAgent
from .config import EvaluationConfig

# .envファイルの読み込み
load_dotenv()


class BaseAgentProvider(ABC):
    """エージェントプロバイダーの基底クラス"""
    
    @abstractmethod
    def create_agent(self, config: EvaluationConfig) -> Any:
        """エージェントを作成する抽象メソッド"""
        pass
    
    @abstractmethod
    def validate_config(self, config: EvaluationConfig) -> None:
        """設定の妥当性を検証する抽象メソッド"""
        pass


class OpenAIAgentProvider(BaseAgentProvider):
    """OpenAIエージェントプロバイダー"""
    
    def validate_config(self, config: EvaluationConfig) -> None:
        """OpenAI設定の妥当性を検証"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEYが設定されていません。"
                ".envファイルを確認してください。"
            )
    
    def create_agent(self, config: EvaluationConfig) -> OpenAILMAgent:
        """OpenAIエージェントを作成"""
        self.validate_config(config)
        
        agent_config = {
            "model_name": config.model.name,
            "temperature": config.model.temperature,
            "max_tokens": config.model.max_tokens,
            "api_key": os.getenv("OPENAI_API_KEY")
        }
        
        return OpenAILMAgent(agent_config)


class HuggingFaceAgentProvider(BaseAgentProvider):
    """HuggingFaceエージェントプロバイダー"""
    
    def validate_config(self, config: EvaluationConfig) -> None:
        """HuggingFace設定の妥当性を検証"""
        # HuggingFaceは特別な環境変数は不要
        # ただし、モデル名の妥当性は実際のロード時に検証される
        pass
    
    def create_agent(self, config: EvaluationConfig) -> HuggingFaceAgent:
        """HuggingFaceエージェントを作成"""
        self.validate_config(config)
        
        agent_config = {
            "model_name": config.model.name,
            "temperature": config.model.temperature,
            "max_new_tokens": config.model.max_new_tokens,
            "device": config.model.device,
            "load_in_8bit": config.model.load_in_8bit,
            "load_in_4bit": config.model.load_in_4bit,
            "trust_remote_code": config.model.trust_remote_code,
            "torch_dtype": config.model.torch_dtype,
            "top_p": config.model.top_p,
            "top_k": config.model.top_k,
            "do_sample": config.agent.do_sample
        }
        
        return HuggingFaceAgent(agent_config)


class AgentFactory:
    """エージェント作成のファクトリークラス"""
    
    _providers = {
        "openai": OpenAIAgentProvider(),
        "huggingface": HuggingFaceAgentProvider()
    }
    
    @classmethod
    def create_agent(cls, config: EvaluationConfig) -> Union[OpenAILMAgent, HuggingFaceAgent]:
        """設定に基づいてエージェントを作成"""
        agent_type = config.agent.type.lower()
        
        if agent_type not in cls._providers:
            raise ValueError(
                f"サポートされていないエージェントタイプ: {agent_type}. "
                f"利用可能なタイプ: {list(cls._providers.keys())}"
            )
        
        provider = cls._providers[agent_type]
        return provider.create_agent(config)
    
    @classmethod
    def validate_agent_config(cls, config: EvaluationConfig) -> None:
        """エージェント設定の妥当性を検証"""
        agent_type = config.agent.type.lower()
        
        if agent_type not in cls._providers:
            raise ValueError(
                f"サポートされていないエージェントタイプ: {agent_type}. "
                f"利用可能なタイプ: {list(cls._providers.keys())}"
            )
        
        provider = cls._providers[agent_type]
        provider.validate_config(config)
    
    @classmethod
    def get_supported_agent_types(cls) -> list:
        """サポートされているエージェントタイプの一覧を取得"""
        return list(cls._providers.keys())
    
    @classmethod
    def register_provider(cls, agent_type: str, provider: BaseAgentProvider) -> None:
        """新しいエージェントプロバイダーを登録"""
        cls._providers[agent_type] = provider 