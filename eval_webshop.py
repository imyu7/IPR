import json
import logging
import os
import asyncio
import concurrent.futures
from dotenv import load_dotenv
from eval_agent.tasks.webshop import WebShopTask
from eval_agent.agents.openai_lm_agent import OpenAILMAgent
from eval_agent.envs.webshop_env import WebShopEnv
from envs.webshop.src.webshop.web_agent_site.envs.web_agent_text_env import WebAgentTextEnv

# .envファイルの読み込み
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _evaluate_single_task(i, task, config, data_path, n_tasks):
    # 環境の作成（各タスクごとに独立したインスタンスを作成）
    env = WebAgentTextEnv(file_path=data_path)
    env.reset()
    webshop_env = WebShopEnv(
        task=task,
        env=env,
        instruction_path="eval_agent/prompt/instructions/webshop_inst.txt",
        icl_path="eval_agent/prompt/icl_examples/webshop_icl.json",
        max_steps=20
    )
    webshop_env.reset()
    
    # エージェントの初期化
    agent = OpenAILMAgent(config)
    
    done = False
    step = 0
    while not done:
        print('-' * 40)
        step += 1
        print(f"step: {step}")
        action = agent(webshop_env.state.history)
        print(action)
        observation, state = webshop_env.step(action)
        print(observation)
        
        if state.finished:
            done = True
    
    print(f"タスク {i}/{n_tasks} が完了しました。")
    print('成功' if state.success else '失敗')
    print('-' * 100)
    
    # タスクの結果を返す
    result = {
        "task_index": i,
        "steps": step,
        "success": state.success,
        "reward": state.reward,
        "intermediate_steps": webshop_env.state.history
    }
    return result

async def evaluate_webshop():
    # OpenAI設定の読み込み
    config = {
        "model_name": "gpt-4o-mini",  
        "temperature": 0,
        "max_tokens": 8192,
        "api_key": os.getenv("OPENAI_API_KEY")
    }
    
    if not config["api_key"]:
        raise ValueError("OPENAI_API_KEYが設定されていません。.envファイルを確認してください。")
    
    # WebShopタスクのロード
    tasks, n_tasks = WebShopTask.load_tasks(split="test", part_num=1)
    # データファイルのパスを設定
    data_path = os.path.join("envs", "webshop", "data", "items_shuffle.json")
    
    # スレッドプールのスレッド数を固定
    max_workers = 1  # 必要に応じて変更してください
    
    loop = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        evaluation_tasks = [
            loop.run_in_executor(
                executor, _evaluate_single_task, i, task, config, data_path, n_tasks
            )
            for i, task in enumerate(tasks, start=1)
        ]
        results = await asyncio.gather(*evaluation_tasks)
    
    # 結果をjsonlファイルに保存
    results_file_path = "evaluation_results.jsonl"
    with open(results_file_path, "w", encoding="utf-8") as results_file:
        for result in results:
            results_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            logger.info(f"タスク {result['task_index']}/{n_tasks} の評価結果が保存されました。")

if __name__ == "__main__":
    asyncio.run(evaluate_webshop()) 