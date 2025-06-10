"""
結果処理モジュール

評価結果の保存、フォーマット、統計計算を担当するモジュールです。
JSONLファイルへの保存やサマリー表示などの機能を提供します。
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from .evaluator import TaskResult
from .config import EvaluationConfig

logger = logging.getLogger(__name__)


class ResultProcessor:
    """結果処理クラス"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self._ensure_results_directory()
    
    def _ensure_results_directory(self) -> None:
        """結果保存ディレクトリの存在を確認・作成"""
        Path(self.config.result.results_dir).mkdir(parents=True, exist_ok=True)
    
    def save_results(self, results: List[TaskResult], stats: Dict[str, Any]) -> str:
        """評価結果をJSONLファイルに保存"""
        # ジョブIDからタイムスタンプを抽出、またはフォールバック
        timestamp = self._extract_timestamp_from_job_id()
        # モデル名の"/"を"-"に置き換え（ディレクトリ名として使用するため）
        safe_model_name = self.config.model.name.replace("/", "-")
        subdir = f"{timestamp}_{safe_model_name}"
        results_dir = Path(self.config.result.results_dir) / subdir
        filename = f"{self.config.task.task_start_idx}-{self.config.task.task_end_idx}.jsonl"
        
        # ディレクトリを作成
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file_path = results_dir / filename
        
        try:
            with open(results_file_path, "w", encoding="utf-8") as results_file:
                for result in results:
                    result_dict = self._task_result_to_dict(result)
                    results_file.write(json.dumps(result_dict, ensure_ascii=False) + "\n")
            
            logger.info(f"評価結果を {results_file_path} に保存しました")
            return str(results_file_path)
            
        except Exception as e:
            logger.error(f"結果の保存に失敗しました: {e}")
            raise
    
    def _extract_timestamp_from_job_id(self) -> str:
        """ジョブIDからタイムスタンプを抽出、またはフォールバックとして現在時刻を使用"""
        if self.config.result.job_id:
            # ジョブIDの形式: MMDD_HHMM_i (例: 0610_2247_0)
            # タイムスタンプ部分のみを抽出
            parts = self.config.result.job_id.split('_')
            if len(parts) >= 2:
                # 最初の2つの部分がタイムスタンプ（MMDD_HHMM）
                timestamp = f"{parts[0]}_{parts[1]}"
                logger.info(f"ジョブIDからタイムスタンプを抽出: {timestamp}")
                return timestamp
            else:
                logger.warning(f"ジョブIDの形式が期待されるものと異なります: {self.config.result.job_id}")
        
        # フォールバック: 現在時刻を使用
        timestamp = datetime.now().strftime('%m%d_%H%M')
        logger.info(f"フォールバックとして現在時刻を使用: {timestamp}")
        return timestamp
    
    def _task_result_to_dict(self, result: TaskResult) -> Dict[str, Any]:
        """TaskResultを辞書に変換"""
        return {
            "task_index": result.task_index,
            "task_query": result.task_query,
            "steps": result.steps,
            "success": result.success,
            "reward": result.reward,
            "intermediate_steps": result.intermediate_steps,
            "execution_time": result.execution_time,
            "error_message": result.error_message
        }
    
    def print_summary(self, results: List[TaskResult], stats: Dict[str, Any]) -> None:
        """評価結果のサマリーを表示"""
        print("\n" + "="*60)
        print("📊 評価結果サマリー")
        print("="*60)
        
        # 基本統計
        print(f"📋 総タスク数: {stats['total_tasks']}")
        print(f"✅ 成功タスク数: {stats['successful_tasks']}")
        print(f"📈 成功率: {stats['success_rate']:.2%}")
        
        if stats['error_tasks'] > 0:
            print(f"❌ エラータスク数: {stats['error_tasks']}")
            print(f"📉 エラー率: {stats['error_rate']:.2%}")
        
        print(f"\n⏱️  実行時間統計:")
        print(f"   総実行時間: {stats['total_execution_time']:.2f}秒")
        print(f"   平均実行時間: {stats['average_execution_time']:.2f}秒/タスク")
        
        print(f"\n🎯 タスク統計:")
        print(f"   平均ステップ数: {stats['average_steps']:.2f}")
        print(f"   平均リワード: {stats['average_reward']:.4f}")
        
        if 'cache_size' in stats:
            print(f"\n💾 キャッシュ統計:")
            print(f"   キャッシュサイズ: {stats['cache_size']}")
        
        # 設定情報
        print(f"\n⚙️  設定情報:")
        print(f"   モデル: {self.config.model.name}")
        print(f"   エージェントタイプ: {self.config.agent.type}")
        print(f"   最大ステップ数: {self.config.agent.max_steps}")
        if self.config.result.job_id:
            print(f"   ジョブID: {self.config.result.job_id}")
        
        print("="*60)
    
    def load_results(self, file_path: str) -> List[TaskResult]:
        """JSONLファイルから結果を読み込み"""
        results = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        result = TaskResult(
                            task_index=data['task_index'],
                            task_query=data['task_query'],
                            steps=data['steps'],
                            success=data['success'],
                            reward=data['reward'],
                            intermediate_steps=data['intermediate_steps'],
                            execution_time=data['execution_time'],
                            error_message=data.get('error_message')
                        )
                        results.append(result)
            
            logger.info(f"結果を {file_path} から読み込みました ({len(results)}件)")
            return results
            
        except Exception as e:
            logger.error(f"結果の読み込みに失敗しました: {e}")
            raise
    
    def merge_results(self, file_paths: List[str]) -> List[TaskResult]:
        """複数の結果ファイルをマージ"""
        all_results = []
        
        for file_path in file_paths:
            results = self.load_results(file_path)
            all_results.extend(results)
        
        # タスクインデックスでソート
        all_results.sort(key=lambda x: x.task_index)
        
        logger.info(f"{len(file_paths)}個のファイルから{len(all_results)}件の結果をマージしました")
        return all_results 