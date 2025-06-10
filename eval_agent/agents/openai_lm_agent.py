import openai
import logging
import backoff
from openai import OpenAIError
from dotenv import load_dotenv
import os

from .base import LMAgent

logger = logging.getLogger("agent_frame")


class OpenAILMAgent(LMAgent):
    def __init__(self, config):
        super().__init__(config)
        assert "model_name" in config.keys()
        # .envファイルから環境変数を読み込む
        load_dotenv()
        # タイムアウト設定を追加
        self.client = openai.OpenAI(
            api_key=config.get('api_key') or os.getenv('OPENAI_API_KEY'),
            timeout=60.0  # 60秒でタイムアウト
        )

    @backoff.on_exception(
        backoff.expo,  # フィボナッチではなく指数バックオフを使用
        OpenAIError,
        max_tries=3,  # 最大3回リトライ
        max_time=300  # 最大5分で諦める
    )
    def __call__(self, messages) -> str:
        # Prepend the prompt with the system message
        # print('[DEBUG] messages: ', messages)
        try:
            response = self.client.chat.completions.create(
                model=self.config["model_name"],
                messages=messages,
                max_tokens=self.config.get("max_tokens", 512),
                temperature=self.config.get("temperature", 0),
                stop=self.stop_words,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API呼び出しエラー: {e}")
            raise
