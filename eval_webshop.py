#!/usr/bin/env python3
"""
統一されたWebShop評価スクリプト

新たにモジュール化されたwebshop_evaluatorパッケージを使用して、
OpenAIとHuggingFaceの両方のモデルを統一的に評価するためのエントリーポイント。

使用例:
# 設定ファイルを使用した評価
python eval_webshop.py --config webshop_evaluator/configs/webshop_gpt-4o-mini.yaml

python eval_webshop.py --config webshop_evaluator/configs/webshop_llama_3_1_70B.yaml

# コマンドライン引数でHuggingFace評価
python eval_webshop.py --agent-type huggingface --model meta-llama/Llama-3.2-3B-Instruct --load-in-4bit

# バッチジョブ用（タスク範囲指定）
python eval_webshop.py --agent-type openai --model gpt-4o-mini --task-start-idx 0 --task-end-idx 50 --job-id batch001
"""

import sys
from dotenv import load_dotenv

# 必要なパッケージのインポート
from webshop_evaluator import (
    ConfigManager,
    EvaluationConfig,
    WebShopEvaluator,
    ResultProcessor,
    setup_logging,
    print_gpu_info,
    print_model_device_info
)

# .envファイルの読み込み
load_dotenv()





def main():
    """メイン実行関数"""
    # ログ設定の初期化
    setup_logging(level="INFO")
    
    # コマンドライン引数の解析
    parser = ConfigManager.create_argument_parser()
    args = parser.parse_args()
    
    try:
        # 設定の作成
        config = ConfigManager.from_args(args)
        
        # 必要なパスの存在確認
        ConfigManager.ensure_paths_exist(config)
        
        # 評価情報の表示
        print_evaluation_info(config, args.config)
        
        # GPU情報の詳細表示（HuggingFaceエージェントの場合）
        if config.agent.type == "huggingface":
            print_gpu_info()
        
        # 結果プロセッサーの作成
        result_processor = ResultProcessor(config)
        
        # 評価の実行
        evaluator = WebShopEvaluator(config)
        
        # モデル初期化後のデバイス配置情報表示（HuggingFaceエージェントの場合）
        if config.agent.type == "huggingface":
            print_model_device_info(evaluator)
        
        results, stats = evaluator.evaluate()
        
        # 結果の保存
        results_file_path = result_processor.save_results(results, stats)
        
        # サマリーの表示
        result_processor.print_summary(results, stats)
        
        # 完了メッセージ
        print(f"\n✅ 評価が完了しました！")
        print(f"📁 結果ファイル: {results_file_path}")
        
    except KeyboardInterrupt:
        print("\n⚠️ 評価が中断されました。")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 評価の実行に失敗しました: {e}")
        sys.exit(1)


def print_evaluation_info(config: EvaluationConfig, config_file_path: str = None):
    """評価情報を表示"""
    print("\n" + "="*60)
    print("🚀 WebShop評価を開始します")
    print("="*60)
    
    if config_file_path:
        print(f"📋 設定ファイル: {config_file_path}")
    
    print(f"🤖 エージェントタイプ: {config.agent.type}")
    print(f"🧠 モデル: {config.model.name}")
    
    if config.agent.type == "huggingface":
        print(f"💾 デバイス: {config.model.device}")
        print(f"⚙️ 量子化: 8bit={config.model.load_in_8bit}, 4bit={config.model.load_in_4bit}")
    
    print(f"📊 評価タスク数: {config.task.test_task_limit}")
    print(f"🏷️ ジョブID: {config.result.job_id}")
    print(f"📁 結果保存先: {config.result.results_dir}")
    
    if config.task.task_end_idx:
        print(f"📂 タスク範囲: {config.task.task_start_idx} - {config.task.task_end_idx}")
    
    print(f"🎯 最大ステップ数: {config.agent.max_steps}")
    
    print("="*60)


if __name__ == "__main__":
    main() 