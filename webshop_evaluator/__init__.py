"""
WebShop評価パッケージ

このパッケージはWebShopタスクの評価を効率的に実行するための
モジュール化されたツールセットを提供します。

主要コンポーネント:
- Config: 設定管理
- AgentFactory: エージェント作成
- WebShopEvaluator: 評価実行
- ResultProcessor: 結果処理
"""

from .config import EvaluationConfig, ConfigManager
from .agents import AgentFactory
from .evaluator import WebShopEvaluator
from .result_processor import ResultProcessor
from .utils import setup_logging
from .gpu_utils import print_gpu_info, print_model_device_info

__version__ = "1.0.0"
__all__ = [
    "EvaluationConfig",
    "ConfigManager", 
    "AgentFactory",
    "WebShopEvaluator",
    "ResultProcessor",
    "setup_logging",
    "print_gpu_info",
    "print_model_device_info"
] 