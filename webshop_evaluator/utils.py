"""
ユーティリティモジュール

共通のヘルパー関数やログ設定などを提供するモジュールです。
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """ログ設定を初期化"""
    
    # ログレベルの設定
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # ログフォーマットの設定
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # ルートロガーの設定
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # 既存のハンドラーをクリア
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # コンソールハンドラーの追加
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # ファイルハンドラーの追加（指定がある場合）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def format_time(seconds: float) -> str:
    """秒を読みやすい時間形式に変換"""
    if seconds < 60:
        return f"{seconds:.2f}秒"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}分{remaining_seconds:.2f}秒"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        remaining_seconds = seconds % 60
        return f"{hours}時間{remaining_minutes}分{remaining_seconds:.2f}秒"


def format_percentage(value: float) -> str:
    """小数値をパーセンテージ文字列に変換"""
    return f"{value:.2%}"


def format_number(value: float, decimal_places: int = 2) -> str:
    """数値を指定された小数点以下桁数でフォーマット"""
    return f"{value:.{decimal_places}f}"


def validate_file_paths(*file_paths: str) -> None:
    """ファイルパスの存在を検証"""
    missing_files = []
    
    for file_path in file_paths:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        raise FileNotFoundError(
            f"以下のファイルが見つかりません: {', '.join(missing_files)}"
        )


def ensure_directories(*directory_paths: str) -> None:
    """ディレクトリの存在を確認し、必要に応じて作成"""
    for directory_path in directory_paths:
        Path(directory_path).mkdir(parents=True, exist_ok=True)


def truncate_string(text: str, max_length: int = 50, suffix: str = "...") -> str:
    """文字列を指定された長さで切り詰める"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def get_memory_usage() -> Optional[float]:
    """現在のメモリ使用量をMBで取得（psutilが利用可能な場合）"""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024  # バイトからMBに変換
    except ImportError:
        return None


def get_gpu_memory_usage() -> Optional[dict]:
    """GPU メモリ使用量を取得（pynvmlが利用可能な場合）"""
    try:
        import pynvml
        
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        
        gpu_info = {}
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            gpu_info[f"gpu_{i}"] = {
                "total_mb": memory_info.total / 1024 / 1024,
                "used_mb": memory_info.used / 1024 / 1024,
                "free_mb": memory_info.free / 1024 / 1024,
                "utilization": (memory_info.used / memory_info.total) * 100
            }
        
        return gpu_info
        
    except ImportError:
        return None


class ProgressTracker:
    """進捗追跡クラス"""
    
    def __init__(self, total: int, description: str = "Progress"):
        self.total = total
        self.current = 0
        self.description = description
        self.logger = logging.getLogger(__name__)
    
    def update(self, increment: int = 1) -> None:
        """進捗を更新"""
        self.current += increment
        percentage = (self.current / self.total) * 100
        
        self.logger.info(
            f"{self.description}: {self.current}/{self.total} ({percentage:.1f}%)"
        )
    
    def reset(self) -> None:
        """進捗をリセット"""
        self.current = 0
    
    def is_complete(self) -> bool:
        """完了状態を確認"""
        return self.current >= self.total 