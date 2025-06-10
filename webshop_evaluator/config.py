"""
設定管理モジュール

WebShop評価の設定を管理するためのクラスとユーティリティを提供します。
YAML設定ファイルとコマンドライン引数のマージを効率的に処理します。
"""

import os
import yaml
import argparse
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union
from datetime import datetime
from pathlib import Path


@dataclass
class ModelConfig:
    """モデル設定"""
    name: str = "gpt-4o-mini"
    temperature: float = 1.0
    max_tokens: int = 8192
    max_new_tokens: int = 512
    top_p: float = 0.9
    top_k: int = 50
    device: str = "auto"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    trust_remote_code: bool = False
    torch_dtype: str = "auto"


@dataclass 
class AgentConfig:
    """エージェント設定"""
    type: str = "openai"  # "openai" or "huggingface"
    max_steps: int = 15
    do_sample: bool = True
    verbose: bool = False  # actionとobservationの詳細ログ出力


@dataclass
class EvaluationTaskConfig:
    """評価タスク設定"""
    test_task_limit: int = 10
    task_start_idx: int = 0
    task_end_idx: Optional[int] = None
    instruction_path: str = "eval_agent/prompt/instructions/webshop_inst.txt"
    icl_path: str = "eval_agent/prompt/icl_examples/webshop_icl.json"
    data_path: str = "envs/webshop/data/items_shuffle.json"


@dataclass
class ResultConfig:
    """結果保存設定"""
    results_dir: str = "webshop_evaluator/results"
    job_id: Optional[str] = None



@dataclass
class EvaluationConfig:
    """統合評価設定"""
    model: ModelConfig = field(default_factory=ModelConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    task: EvaluationTaskConfig = field(default_factory=EvaluationTaskConfig)
    result: ResultConfig = field(default_factory=ResultConfig)
    
    def __post_init__(self):
        """初期化後の処理"""
        # job_idが未設定の場合、自動生成
        if self.result.job_id is None:
            self.result.job_id = datetime.now().strftime("%m%d_%H%M")


class ConfigManager:
    """設定管理クラス"""
    
    @staticmethod
    def load_yaml_config(config_path: str) -> Dict[str, Any]:
        """YAML設定ファイルを読み込む"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config or {}
    
    @staticmethod
    def create_argument_parser() -> argparse.ArgumentParser:
        """コマンドライン引数パーサーを作成"""
        parser = argparse.ArgumentParser(
            description="WebShop評価スクリプト",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        # 設定ファイル
        parser.add_argument("--config", type=str, help="YAML設定ファイルのパス")
        
        # エージェント設定
        parser.add_argument("--agent-type", type=str, choices=["openai", "huggingface"], 
                          help="使用するエージェントタイプ")
        
        # モデル設定
        parser.add_argument("--model", type=str, help="使用するモデル名")
        parser.add_argument("--temperature", type=float, help="温度パラメータ")
        parser.add_argument("--max-tokens", type=int, help="最大トークン数 (OpenAI用)")
        parser.add_argument("--max-new-tokens", type=int, help="最大生成トークン数 (HuggingFace用)")
        parser.add_argument("--top-p", type=float, help="nucleus samplingのp値")
        parser.add_argument("--top-k", type=int, help="top-k samplingのk値")
        parser.add_argument("--device", type=str, help="使用するデバイス")
        parser.add_argument("--load-in-8bit", action="store_true", help="8bit量子化を使用")
        parser.add_argument("--load-in-4bit", action="store_true", help="4bit量子化を使用")
        parser.add_argument("--trust-remote-code", action="store_true", help="リモートコードの実行を許可")
        parser.add_argument("--torch-dtype", type=str, help="PyTorchのデータ型")
        
        # エージェント設定
        parser.add_argument("--max-steps", type=int, help="タスクあたりの最大ステップ数")
        parser.add_argument("--verbose", action="store_true", help="actionとobservationの詳細ログを出力")
        
        # タスク設定
        parser.add_argument("--test-task-limit", type=int, help="タスク数の上限")
        parser.add_argument("--task-start-idx", type=int, help="開始タスクインデックス")
        parser.add_argument("--task-end-idx", type=int, help="終了タスクインデックス")
        parser.add_argument("--instruction-path", type=str, help="インストラクションファイルのパス")
        parser.add_argument("--icl-path", type=str, help="ICL例ファイルのパス")
        parser.add_argument("--data-path", type=str, help="データファイルのパス")
        
        # 結果設定
        parser.add_argument("--results-dir", type=str, help="結果保存ディレクトリ")
        parser.add_argument("--job-id", type=str, help="バッチジョブID")
        
        return parser
    
    @staticmethod
    def merge_configs(yaml_config: Optional[Dict[str, Any]], 
                     args: argparse.Namespace) -> EvaluationConfig:
        """YAML設定とコマンドライン引数をマージしてEvaluationConfigを作成"""
        
        # デフォルト設定から開始
        config = EvaluationConfig()
        
        # YAML設定をマージ
        if yaml_config:
            ConfigManager._apply_yaml_config(config, yaml_config)
        
        # コマンドライン引数で上書き
        ConfigManager._apply_cli_args(config, args)
        
        return config
    
    @staticmethod
    def _apply_yaml_config(config: EvaluationConfig, yaml_config: Dict[str, Any]):
        """YAML設定をEvaluationConfigに適用"""
        
        # モデル設定の適用
        if 'model' in yaml_config:
            model_cfg = yaml_config['model']
            for key, value in model_cfg.items():
                if hasattr(config.model, key):
                    setattr(config.model, key, value)
        
        # エージェント設定の適用
        if 'agent' in yaml_config:
            agent_cfg = yaml_config['agent']
            for key, value in agent_cfg.items():
                if hasattr(config.agent, key):
                    setattr(config.agent, key, value)
        
        # 評価設定の適用
        if 'evaluation' in yaml_config:
            eval_cfg = yaml_config['evaluation']
            for key, value in eval_cfg.items():
                if hasattr(config.task, key):
                    setattr(config.task, key, value)
        
        # 結果設定の適用
        if 'batch' in yaml_config:
            batch_cfg = yaml_config['batch']
            if 'job_id' in batch_cfg:
                config.result.job_id = batch_cfg['job_id']
        
    @staticmethod
    def _apply_cli_args(config: EvaluationConfig, args: argparse.Namespace):
        """コマンドライン引数をEvaluationConfigに適用"""
        
        # エージェント設定
        if args.agent_type is not None:
            config.agent.type = args.agent_type
        if args.max_steps is not None:
            config.agent.max_steps = args.max_steps
        if args.verbose:
            config.agent.verbose = True
        
        # モデル設定
        if args.model is not None:
            config.model.name = args.model
        if args.temperature is not None:
            config.model.temperature = args.temperature
        if args.max_tokens is not None:
            config.model.max_tokens = args.max_tokens
        if args.max_new_tokens is not None:
            config.model.max_new_tokens = args.max_new_tokens
        if args.top_p is not None:
            config.model.top_p = args.top_p
        if args.top_k is not None:
            config.model.top_k = args.top_k
        if args.device is not None:
            config.model.device = args.device
        if args.load_in_8bit:
            config.model.load_in_8bit = True
        if args.load_in_4bit:
            config.model.load_in_4bit = True
        if args.trust_remote_code:
            config.model.trust_remote_code = True
        if args.torch_dtype is not None:
            config.model.torch_dtype = args.torch_dtype
        
        # タスク設定
        if args.test_task_limit is not None:
            config.task.test_task_limit = args.test_task_limit
        if args.task_start_idx is not None:
            config.task.task_start_idx = args.task_start_idx
        if args.task_end_idx is not None:
            config.task.task_end_idx = args.task_end_idx
        if args.instruction_path is not None:
            config.task.instruction_path = args.instruction_path
        if args.icl_path is not None:
            config.task.icl_path = args.icl_path
        if args.data_path is not None:
            config.task.data_path = args.data_path
        
        # 結果設定
        if args.results_dir is not None:
            config.result.results_dir = args.results_dir
        if args.job_id is not None:
            config.result.job_id = args.job_id
    
    @staticmethod
    def from_args(args: argparse.Namespace) -> EvaluationConfig:
        """コマンドライン引数からEvaluationConfigを作成"""
        yaml_config = None
        if args.config:
            yaml_config = ConfigManager.load_yaml_config(args.config)
        
        return ConfigManager.merge_configs(yaml_config, args)
    
    @staticmethod
    def ensure_paths_exist(config: EvaluationConfig):
        """必要なディレクトリの存在を確認・作成"""
        Path(config.result.results_dir).mkdir(parents=True, exist_ok=True)
        
        # 必要なファイルの存在確認
        required_files = [
            config.task.instruction_path,
            config.task.icl_path,
            config.task.data_path
        ]
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"必要なファイルが見つかりません: {file_path}") 