"""
評価実行モジュール

WebShopタスクの評価を実行するためのメインロジックを含むモジュールです。
タスクの読み込み、エージェントの実行、結果の収集を担当します。
"""

import time
import logging
from typing import List, Tuple, Any, Optional
from dataclasses import dataclass

from eval_agent.tasks.webshop import WebShopTask
from eval_agent.envs.webshop_env import WebShopEnv
from envs.webshop.src.webshop.web_agent_site.envs.web_agent_text_env import WebAgentTextEnv

from .config import EvaluationConfig
from .agents import AgentFactory

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """タスク結果を管理するデータクラス"""
    task_index: int
    task_query: str
    steps: int
    success: bool
    reward: float
    intermediate_steps: List[Any]
    execution_time: float
    error_message: Optional[str] = None


class TaskLoader:
    """タスク読み込み管理クラス"""
    
    @staticmethod
    def load_tasks(config: EvaluationConfig) -> Tuple[List[Any], int, int]:
        """WebShopタスクを読み込む（範囲指定対応）"""
        try:
            all_tasks, _ = WebShopTask.load_tasks(split="test", part_num=1)
            all_tasks = list(all_tasks)
            
            # タスク範囲の決定
            start_idx = config.task.task_start_idx
            if config.task.task_end_idx is not None:
                end_idx = min(config.task.task_end_idx, len(all_tasks))
            else:
                end_idx = min(start_idx + config.task.test_task_limit, len(all_tasks))
            
            # 指定範囲のタスクを取得
            tasks = all_tasks[start_idx:end_idx]
            n_tasks = len(tasks)
            
            logger.info(f"全タスク数: {len(all_tasks)}, 処理範囲: {start_idx}-{end_idx}, 処理タスク数: {n_tasks}")
            return tasks, n_tasks, start_idx
            
        except Exception as e:
            logger.error(f"タスクの読み込みに失敗しました: {e}")
            raise


class TaskExecutor:
    """タスク実行管理クラス"""
    
    def __init__(self, config: EvaluationConfig, agent: Any):
        self.config = config
        self.agent = agent
        # 共有環境インスタンスを作成（一度だけ）
        self.shared_env = WebAgentTextEnv(file_path=self.config.task.data_path)
    
    def execute_task(self, task_index: int, task: Any, n_tasks: int, global_index: int) -> TaskResult:
        """単一タスクの実行"""
        start_time = time.time()
        logger.info(f"タスク {task_index}/{n_tasks} (グローバルインデックス: {global_index}) を開始")
        
        try:
            # 既存の環境をリセット（新規作成の代わりに）
            self.shared_env.reset()
            
            webshop_env = WebShopEnv(
                task=task,
                env=self.shared_env,
                instruction_path=self.config.task.instruction_path,
                icl_path=self.config.task.icl_path,
                max_steps=self.config.agent.max_steps
            )
            webshop_env.reset()
            initial_observation = webshop_env.env.observation
            
            # タスク実行
            step = 0
            while not webshop_env.state.finished:
                step += 1
                
                # エージェントの呼び出し
                action = self.agent(webshop_env.state.history)
                
                # verboseが有効な場合、actionの内容をログ出力
                if self.config.agent.verbose:
                    logger.info(f"タスク {task_index} ステップ {step} - Action: {action}")
                
                observation, state = webshop_env.step(action)
                
                # verboseが有効な場合、observationの内容をログ出力
                if self.config.agent.verbose:
                    logger.info(f"タスク {task_index} ステップ {step} - Observation: {observation}")
                    logger.info(f"タスク {task_index} ステップ {step} - State finished: {state.finished}, success: {state.success}, reward: {state.reward}")
                
                if step >= self.config.agent.max_steps:
                    logger.warning(f"タスク {task_index} が最大ステップ数に到達しました")
                    break
            
            execution_time = time.time() - start_time
            result_status = "成功" if webshop_env.state.success else "失敗"
            logger.info(f"タスク {task_index}/{n_tasks} (グローバルインデックス: {global_index}) を完了: {result_status} (実行時間: {execution_time:.2f}秒)")
            
            return TaskResult(
                task_index=global_index,
                task_query=initial_observation,
                steps=step,
                success=webshop_env.state.success,
                reward=webshop_env.state.reward,
                intermediate_steps=webshop_env.state.history,
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"タスク {task_index} の評価中にエラーが発生しました: {e}")
            return TaskResult(
                task_index=global_index,
                task_query="",
                steps=0,
                success=False,
                reward=0.0,
                intermediate_steps=[],
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def cleanup(self) -> None:
        """共有環境のクリーンアップ"""
        if hasattr(self, 'shared_env') and self.shared_env:
            try:
                self.shared_env.close()
            except:
                pass


class WebShopEvaluator:
    """WebShop評価のメインクラス"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.agent = None
        self.executor = None
        
        # エージェントの作成
        self._initialize_agent()
    
    def _initialize_agent(self) -> None:
        """エージェントの初期化"""
        logger.info("エージェントを初期化しています...")
        
        try:
            self.agent = AgentFactory.create_agent(self.config)
            self.executor = TaskExecutor(self.config, self.agent)
            logger.info(f"{self.config.agent.type}エージェントを作成しました")
            
        except Exception as e:
            logger.error(f"エージェントの初期化に失敗しました: {e}")
            raise
    
    def evaluate(self) -> Tuple[List[TaskResult], dict]:
        """評価を実行"""
        logger.info("WebShop評価を開始します")
        if self.config.result.job_id:
            logger.info(f"ジョブID: {self.config.result.job_id}")
        
        total_start_time = time.time()
        
        try:
            # タスクの読み込み
            tasks, n_tasks, start_idx = TaskLoader.load_tasks(self.config)
            
            # 順次評価の実行
            all_results = []
            for task_index, task in enumerate(tasks, start=1):
                global_index = start_idx + task_index - 1
                result = self.executor.execute_task(task_index, task, n_tasks, global_index)
                all_results.append(result)
            
            # 統計の計算
            total_time = time.time() - total_start_time
            stats = self._calculate_statistics(all_results)
            stats["total_evaluation_time"] = total_time
            
            logger.info(f"評価統計: {stats}")
            logger.info(f"総評価時間: {total_time:.2f}秒")
            logger.info(f"タスクあたりの平均時間: {stats['average_execution_time']:.2f}秒")
            
            logger.info("WebShop評価が完了しました")
            return all_results, stats
            
        except Exception as e:
            logger.error(f"評価中にエラーが発生しました: {e}")
            raise
        
        finally:
            # クリーンアップ
            self._cleanup()
    
    def _calculate_statistics(self, results: List[TaskResult]) -> dict:
        """評価統計を計算"""
        total_tasks = len(results)
        successful_tasks = sum(1 for r in results if r.success)
        success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0.0
        average_steps = sum(r.steps for r in results) / total_tasks if total_tasks > 0 else 0.0
        average_reward = sum(r.reward for r in results) / total_tasks if total_tasks > 0 else 0.0
        total_execution_time = sum(r.execution_time for r in results)
        average_execution_time = total_execution_time / total_tasks if total_tasks > 0 else 0.0
        
        # エラーの統計
        error_tasks = sum(1 for r in results if r.error_message is not None)
        error_rate = error_tasks / total_tasks if total_tasks > 0 else 0.0
        
        return {
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "success_rate": success_rate,
            "error_tasks": error_tasks,
            "error_rate": error_rate,
            "average_steps": average_steps,
            "average_reward": average_reward,
            "total_execution_time": total_execution_time,
            "average_execution_time": average_execution_time
        }
    
    def _cleanup(self) -> None:
        """リソースのクリーンアップ"""
        if self.agent and hasattr(self.agent, '__del__'):
            try:
                self.agent.__del__()
            except:
                pass
        
        if self.executor:
            self.executor.cleanup() 