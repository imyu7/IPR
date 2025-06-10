#!/usr/bin/env python3
"""
バッチジョブの結果を統合するスクリプト

使用例:
# 結果の統合または統計表示（merged.jsonlの有無で自動判定）
python merge_results.py webshop_evaluator/results/0610_2255_gpt-4.1-2025-04-14

# 統計情報をファイルに保存
python merge_results.py webshop_evaluator/results/0610_2255_gpt-4.1-2025-04-14 --stats-output statistics.json
"""

import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(description="バッチジョブ結果統合スクリプト")
    
    parser.add_argument(
        "results_dir",
        type=str,
        help="結果ファイルが保存されているディレクトリ"
    )
    parser.add_argument(
        "--stats-output",
        type=str,
        default=None,
        help="統計情報の出力ファイル名 (オプション)"
    )
    
    return parser.parse_args()


def load_merged_file(file_path: Path) -> List[Dict[str, Any]]:
    """統合済みのJSONLファイルを読み込む"""
    results = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        logger.info(f"ファイル {file_path.name} から {len(results)} 件の結果を読み込みました")
    except Exception as e:
        logger.error(f"ファイル {file_path} の読み込みに失敗しました: {e}")
    return results


def merge_results(results_dir: Path) -> List[Dict[str, Any]]:
    """ディレクトリ内のJSONLファイルを統合"""
    all_results = []
    result_files = list(results_dir.glob("*.jsonl"))
    
    # merged.jsonlは除外
    merged_file = results_dir / "merged.jsonl"
    if merged_file in result_files:
        result_files.remove(merged_file)
    
    if not result_files:
        logger.warning(f"ディレクトリ {results_dir} にJSONLファイルが見つかりません")
        return all_results
    
    logger.info(f"{len(result_files)} 個の結果ファイルを見つけました")
    
    # タスクインデックスでソートするための辞書
    task_dict = {}
    
    for file_path in sorted(result_files):
        results = load_merged_file(file_path)
        for result in results:
            task_idx = result.get('task_index')
            if task_idx is not None:
                # 重複チェック
                if task_idx in task_dict:
                    logger.warning(f"タスク {task_idx} が重複しています。新しい結果で上書きします。")
                task_dict[task_idx] = result
    
    # タスクインデックスでソート
    all_results = [task_dict[idx] for idx in sorted(task_dict.keys())]
    
    logger.info(f"合計 {len(all_results)} 件の結果を統合しました")
    
    # タスクの欠落をチェック
    task_indices = sorted(task_dict.keys())
    if task_indices:
        expected_indices = set(range(min(task_indices), max(task_indices) + 1))
        missing_indices = expected_indices - set(task_indices)
        if missing_indices:
            logger.warning(f"以下のタスクインデックスが欠落しています: {sorted(missing_indices)}")
    
    return all_results


def calculate_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """統合された結果から統計を計算"""
    if not results:
        return {}
    
    total_tasks = len(results)
    successful_tasks = sum(1 for r in results if r.get('success', False))
    success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0
    
    steps_list = [r.get('steps', 0) for r in results]
    rewards_list = [r.get('reward', 0.0) for r in results]
    execution_times = [r.get('execution_time', 0.0) for r in results]
    
    average_steps = sum(steps_list) / total_tasks if total_tasks > 0 else 0
    average_reward = sum(rewards_list) / total_tasks if total_tasks > 0 else 0
    total_execution_time = sum(execution_times)
    average_execution_time = total_execution_time / total_tasks if total_tasks > 0 else 0
    
    # タスクごとの成功率を計算（10タスクごとのグループ）
    group_size = 10
    group_stats = defaultdict(lambda: {'total': 0, 'success': 0})
    
    for result in results:
        task_idx = result.get('task_index', 0)
        group_idx = task_idx // group_size
        group_stats[group_idx]['total'] += 1
        if result.get('success', False):
            group_stats[group_idx]['success'] += 1
    
    group_success_rates = {}
    for group_idx, stats in sorted(group_stats.items()):
        start_idx = group_idx * group_size
        end_idx = start_idx + group_size - 1
        success_rate_group = stats['success'] / stats['total'] if stats['total'] > 0 else 0
        group_success_rates[f"tasks_{start_idx}-{end_idx}"] = {
            'total': stats['total'],
            'success': stats['success'],
            'success_rate': success_rate_group
        }
    
    return {
        'total_tasks': total_tasks,
        'successful_tasks': successful_tasks,
        'success_rate': success_rate,
        'average_steps': average_steps,
        'average_reward': average_reward,
        'total_execution_time': total_execution_time,
        'average_execution_time': average_execution_time,
        'min_steps': min(steps_list) if steps_list else 0,
        'max_steps': max(steps_list) if steps_list else 0,
        'min_reward': min(rewards_list) if rewards_list else 0,
        'max_reward': max(rewards_list) if rewards_list else 0,
        'group_statistics': group_success_rates
    }


def save_merged_results(results: List[Dict[str, Any]], output_path: Path):
    """統合結果をJSONLファイルとして保存"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        logger.info(f"統合結果を {output_path} に保存しました")
    except Exception as e:
        logger.error(f"結果の保存に失敗しました: {e}")
        raise


def save_statistics(stats: Dict[str, Any], output_path: Path):
    """統計情報を保存"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info(f"統計情報を {output_path} に保存しました")
    except Exception as e:
        logger.error(f"統計情報の保存に失敗しました: {e}")


def print_statistics(stats: Dict[str, Any]):
    """統計情報を表示"""
    print("\n=== 統計情報 ===")
    print(f"総タスク数: {stats.get('total_tasks', 0)}")
    print(f"成功タスク数: {stats.get('successful_tasks', 0)}")
    print(f"成功率: {stats.get('success_rate', 0):.2%}")
    print(f"平均ステップ数: {stats.get('average_steps', 0):.2f}")
    print(f"平均報酬: {stats.get('average_reward', 0):.4f}")
    print(f"総実行時間: {stats.get('total_execution_time', 0):.2f}秒")
    print(f"平均実行時間: {stats.get('average_execution_time', 0):.2f}秒/タスク")
    print(f"最小/最大ステップ数: {stats.get('min_steps', 0)} / {stats.get('max_steps', 0)}")
    print(f"最小/最大報酬: {stats.get('min_reward', 0):.4f} / {stats.get('max_reward', 0):.4f}")
    
    # グループ統計の表示
    group_stats = stats.get('group_statistics', {})
    if group_stats:
        print("\n=== グループ別成功率 ===")
        for group_name, group_data in sorted(group_stats.items()):
            print(f"{group_name}: {group_data['success']}/{group_data['total']} ({group_data['success_rate']:.2%})")


def main():
    """メイン実行関数"""
    args = parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logger.error(f"ディレクトリ {results_dir} が存在しません")
        return
    
    merged_file_path = results_dir / "merged.jsonl"
    
    # merged.jsonlの存在をチェックして動作モードを決定
    if merged_file_path.exists():
        # 統計情報表示モード
        logger.info("=== 統計情報表示モード ===")
        logger.info(f"統合済みファイル: {merged_file_path}")
        
        # 統合済みファイルを読み込み
        merged_results = load_merged_file(merged_file_path)
        
        if not merged_results:
            logger.warning("読み込む結果がありません")
            return
        
        # 統計の計算と表示
        stats = calculate_statistics(merged_results)
        print_statistics(stats)
        
        # 統計情報の保存（オプション）
        if args.stats_output:
            stats_path = Path(args.stats_output)
            if not stats_path.is_absolute():
                stats_path = results_dir / stats_path
            save_statistics(stats, stats_path)
        
        logger.info("統計情報表示が完了しました")
        
    else:
        # マージモード
        logger.info("=== バッチジョブ結果統合モード ===")
        logger.info(f"結果ディレクトリ: {results_dir}")
        logger.info(f"出力ファイル: {merged_file_path}")
        
        # 結果の統合
        merged_results = merge_results(results_dir)
        
        if not merged_results:
            logger.warning("統合する結果がありません")
            return
        
        # 統合結果の保存
        save_merged_results(merged_results, merged_file_path)
        
        # 統計の計算と表示
        stats = calculate_statistics(merged_results)
        print_statistics(stats)
        
        # 統計情報の保存（オプション）
        if args.stats_output:
            stats_path = Path(args.stats_output)
            if not stats_path.is_absolute():
                stats_path = results_dir / stats_path
            save_statistics(stats, stats_path)
        
        logger.info("統合処理が完了しました")


if __name__ == "__main__":
    main() 