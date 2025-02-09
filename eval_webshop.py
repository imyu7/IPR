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
import threading
from datetime import datetime
# .envファイルの読み込み
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _evaluate_single_task(i, task, config, data_path, n_tasks):
    thread_name = threading.current_thread().name
    print(f"スレッド {thread_name} がタスク {i}/{n_tasks} を開始")
    
    # 環境の作成（各タスクごとに独立したインスタンスを作成）
    env = WebAgentTextEnv(file_path=data_path)
    env.reset()
    webshop_env = WebShopEnv(
        task=task,
        env=env,
        instruction_path="eval_agent/prompt/instructions/webshop_inst.txt",
        icl_path="eval_agent/prompt/icl_examples/webshop_icl.json",
        max_steps=15
    )
    webshop_env.reset()
    initial_observation = webshop_env.env.observation
    # print(f"[DEBUG] initial_observation: {initial_observation}")
    
    # エージェントの初期化
    agent = OpenAILMAgent(config)
    
    done = False
    step = 0
    while not done:
        # print('-' * 40)
        step += 1
        # print(f"step: {step}")
        action = agent(webshop_env.state.history)
        # print(action)
        observation, state = webshop_env.step(action)
        # print(observation)
        
        if state.finished:
            done = True
    
    print(f"スレッド {thread_name} でタスク {i}/{n_tasks} を完了")
    print('成功！！' if state.success else '失敗')
    # print('-' * 100)
    
    # タスクの結果を返す
    result = {
        "task_index": i,
        "task_query": initial_observation,
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
        "temperature": 1.0,
        "max_tokens": 8192,
        "api_key": os.getenv("OPENAI_API_KEY")
    }
    
    if not config["api_key"]:
        raise ValueError("OPENAI_API_KEYが設定されていません。.envファイルを確認してください。")
    
    # WebShopタスクのロード
    tasks, n_tasks = WebShopTask.load_tasks(split="test", part_num=1)
    # for test 用に generator を list に変換してからスライス
    tasks = list(tasks)
    tasks = tasks[:100]
    n_tasks = 100
    
    # データファイルのパスを設定
    data_path = os.path.join("envs", "webshop", "data", "items_shuffle.json")
    
    print("事前に重いリソースを初期化しています...")
    dummy_env = WebAgentTextEnv(file_path=data_path)
    dummy_env.reset()
    del dummy_env
    
    # スレッドプールのスレッド数を固定
    max_workers = 30  # 必要に応じて変更
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
    results_file_path = f"results/eval_webshop_{n_tasks}_{datetime.now().strftime('%m%d_%H%M')}.jsonl"
    with open(results_file_path, "w", encoding="utf-8") as results_file:
        for result in results:
            results_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            logger.info(f"タスク {result['task_index']}/{n_tasks} の評価結果が保存されました。")

if __name__ == "__main__":
    asyncio.run(evaluate_webshop()) 