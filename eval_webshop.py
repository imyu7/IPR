import json
import logging
import os
import asyncio
import concurrent.futures
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from dotenv import load_dotenv
from eval_agent.tasks.webshop import WebShopTask
from eval_agent.agents.openai_lm_agent import OpenAILMAgent
from eval_agent.envs.webshop_env import WebShopEnv
from envs.webshop.src.webshop.web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
import threading
from datetime import datetime

# .envファイルの読み込み
load_dotenv()

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """評価設定を管理するデータクラス"""
    model_name: str = "gpt-4o-mini"
    temperature: float = 1.0
    max_tokens: int = 8192
    max_steps: int = 15
    max_workers: int = 3
    test_task_limit: int = 10
    instruction_path: str = "eval_agent/prompt/instructions/webshop_inst.txt"
    icl_path: str = "eval_agent/prompt/icl_examples/webshop_icl.json"
    data_path: str = "envs/webshop/data/items_shuffle.json"
    results_dir: str = "results"


@dataclass
class TaskResult:
    """タスク結果を管理するデータクラス"""
    task_index: int
    task_query: str
    steps: int
    success: bool
    reward: float
    intermediate_steps: List[Any]


class WebShopEvaluator:
    """WebShop環境での評価を管理するクラス"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.openai_config = self._load_openai_config()
        self._ensure_results_directory()
    
    def _load_openai_config(self) -> Dict[str, Any]:
        """OpenAI設定を読み込む"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEYが設定されていません。.envファイルを確認してください。")
        
        return {
            "model_name": self.config.model_name,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "api_key": api_key
        }
    
    def _ensure_results_directory(self) -> None:
        """結果保存ディレクトリの存在を確認・作成"""
        Path(self.config.results_dir).mkdir(exist_ok=True)
    
    def _load_tasks(self) -> Tuple[List[Any], int]:
        """WebShopタスクを読み込む"""
        try:
            tasks, _ = WebShopTask.load_tasks(split="test", part_num=1)
            tasks = list(tasks)[:self.config.test_task_limit]
            n_tasks = len(tasks)
            logger.info(f"{n_tasks}個のタスクを読み込みました")
            return tasks, n_tasks
        except Exception as e:
            logger.error(f"タスクの読み込みに失敗しました: {e}")
            raise
    
    def _initialize_environment(self) -> None:
        """重いリソースの事前初期化"""
        try:
            logger.info("事前に重いリソースを初期化しています...")
            dummy_env = WebAgentTextEnv(file_path=self.config.data_path)
            dummy_env.reset()
            del dummy_env
            logger.info("リソースの初期化が完了しました")
        except Exception as e:
            logger.error(f"環境の初期化に失敗しました: {e}")
            raise
    
    def _evaluate_single_task(self, task_index: int, task: Any, n_tasks: int) -> TaskResult:
        """単一タスクの評価を実行"""
        thread_name = threading.current_thread().name
        logger.info(f"スレッド {thread_name} がタスク {task_index}/{n_tasks} を開始")
        
        try:
            # 環境の作成
            env = WebAgentTextEnv(file_path=self.config.data_path)
            env.reset()
            
            webshop_env = WebShopEnv(
                task=task,
                env=env,
                instruction_path=self.config.instruction_path,
                icl_path=self.config.icl_path,
                max_steps=self.config.max_steps
            )
            webshop_env.reset()
            initial_observation = webshop_env.env.observation
            
            # エージェントの初期化
            agent = OpenAILMAgent(self.openai_config)
            
            # タスク実行
            step = 0
            while not webshop_env.state.finished:
                step += 1
                action = agent(webshop_env.state.history)
                observation, state = webshop_env.step(action)
                
                if step >= self.config.max_steps:
                    logger.warning(f"タスク {task_index} が最大ステップ数に到達しました")
                    break
            
            result_status = "成功" if webshop_env.state.success else "失敗"
            logger.info(f"スレッド {thread_name} でタスク {task_index}/{n_tasks} を完了: {result_status}")
            
            return TaskResult(
                task_index=task_index,
                task_query=initial_observation,
                steps=step,
                success=webshop_env.state.success,
                reward=webshop_env.state.reward,
                intermediate_steps=webshop_env.state.history
            )
            
        except Exception as e:
            logger.error(f"タスク {task_index} の評価中にエラーが発生しました: {e}")
            return TaskResult(
                task_index=task_index,
                task_query="",
                steps=0,
                success=False,
                reward=0.0,
                intermediate_steps=[]
            )
    
    def _save_results(self, results: List[TaskResult], n_tasks: int) -> str:
        """評価結果をJSONLファイルに保存"""
        timestamp = datetime.now().strftime('%m%d_%H%M')
        results_file_path = Path(self.config.results_dir) / f"eval_webshop_{n_tasks}_{timestamp}.jsonl"
        
        try:
            with open(results_file_path, "w", encoding="utf-8") as results_file:
                for result in results:
                    result_dict = {
                        "task_index": result.task_index,
                        "task_query": result.task_query,
                        "steps": result.steps,
                        "success": result.success,
                        "reward": result.reward,
                        "intermediate_steps": result.intermediate_steps
                    }
                    results_file.write(json.dumps(result_dict, ensure_ascii=False) + "\n")
            
            logger.info(f"評価結果を {results_file_path} に保存しました")
            return str(results_file_path)
            
        except Exception as e:
            logger.error(f"結果の保存に失敗しました: {e}")
            raise
    
    def _calculate_statistics(self, results: List[TaskResult]) -> Dict[str, Any]:
        """評価統計を計算"""
        total_tasks = len(results)
        successful_tasks = sum(1 for r in results if r.success)
        success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0
        average_steps = sum(r.steps for r in results) / total_tasks if total_tasks > 0 else 0
        average_reward = sum(r.reward for r in results) / total_tasks if total_tasks > 0 else 0
        
        return {
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "success_rate": success_rate,
            "average_steps": average_steps,
            "average_reward": average_reward
        }
    
    async def evaluate(self) -> str:
        """WebShop評価のメイン実行メソッド"""
        logger.info("WebShop評価を開始します")
        
        try:
            # タスクの読み込み
            tasks, n_tasks = self._load_tasks()
            
            # 環境の初期化
            self._initialize_environment()
            
            # 並列評価の実行
            loop = asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                evaluation_tasks = [
                    loop.run_in_executor(
                        executor, self._evaluate_single_task, i, task, n_tasks
                    )
                    for i, task in enumerate(tasks, start=1)
                ]
                results = await asyncio.gather(*evaluation_tasks)
            
            # 統計の計算
            stats = self._calculate_statistics(results)
            logger.info(f"評価統計: {stats}")
            
            # 結果の保存
            results_file_path = self._save_results(results, n_tasks)
            
            logger.info("WebShop評価が完了しました")
            return results_file_path
            
        except Exception as e:
            logger.error(f"評価中にエラーが発生しました: {e}")
            raise


async def main():
    """メイン実行関数"""
    config = EvaluationConfig()
    evaluator = WebShopEvaluator(config)
    
    try:
        results_file = await evaluator.evaluate()
        print(f"評価が完了しました。結果は {results_file} に保存されています。")
    except Exception as e:
        logger.error(f"評価の実行に失敗しました: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 