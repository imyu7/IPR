"""
GPU情報取得ユーティリティモジュール

GPU使用状況の詳細確認とモデルのデバイス配置情報表示のためのユーティリティ関数を提供します。
"""

import torch
import psutil
import subprocess
from typing import Dict, Any, Optional


def print_gpu_info():
    """GPU情報を詳細に表示"""
    print("\n" + "="*60)
    print("🖥️  GPU使用状況の詳細確認")
    print("="*60)
    
    # PyTorchのCUDA情報
    if torch.cuda.is_available():
        print(f"✅ CUDA利用可能: {torch.cuda.is_available()}")
        print(f"🔢 CUDA利用可能デバイス数: {torch.cuda.device_count()}")
        print(f"🎯 現在のCUDAデバイス: {torch.cuda.current_device()}")
        print(f"📝 CUDAバージョン: {torch.version.cuda}")
        
        # 各GPUの詳細情報
        for i in range(torch.cuda.device_count()):
            device_name = torch.cuda.get_device_name(i)
            memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            memory_cached = torch.cuda.memory_reserved(i) / 1024**3
            
            print(f"🎮 GPU {i}: {device_name}")
            print(f"   💾 総メモリ: {memory_total:.2f} GB")
            print(f"   🔥 使用メモリ: {memory_allocated:.2f} GB")
            print(f"   📦 キャッシュメモリ: {memory_cached:.2f} GB")
            print(f"   📊 使用率: {(memory_allocated/memory_total)*100:.1f}%")
    else:
        print("❌ CUDAが利用できません")
    
    # nvidia-smi情報の取得（可能な場合）
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu', 
                               '--format=csv,noheader,nounits'], 
                               capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("\n🔍 nvidia-smi情報:")
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.strip():
                    parts = line.split(', ')
                    if len(parts) >= 5:
                        idx, name, mem_used, mem_total, util = parts[:5]
                        print(f"   GPU {idx}: {name} | メモリ: {mem_used}/{mem_total} MB | 使用率: {util}%")
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        print("⚠️ nvidia-smi情報を取得できませんでした")
    
    # システムリソース情報
    print(f"\n💻 システム情報:")
    print(f"   🧠 CPU数: {psutil.cpu_count()}")
    print(f"   💾 総メモリ: {psutil.virtual_memory().total / 1024**3:.2f} GB")
    print(f"   📊 メモリ使用率: {psutil.virtual_memory().percent}%")
    
    print("="*60)


def print_model_device_info(evaluator):
    """モデルの実際のデバイス配置情報を表示"""
    print("\n" + "="*60)
    print("🧠 モデルデバイス配置情報")
    print("="*60)
    
    try:
        # HuggingFaceエージェントの場合
        if hasattr(evaluator, 'agent') and hasattr(evaluator.agent, 'model'):
            model = evaluator.agent.model
            
            # モデルの各パラメータがどのデバイスにあるかを確認
            device_map = {}
            total_params = 0
            
            for name, param in model.named_parameters():
                device = str(param.device)
                if device not in device_map:
                    device_map[device] = {'count': 0, 'params': 0}
                device_map[device]['count'] += 1
                device_map[device]['params'] += param.numel()
                total_params += param.numel()
            
            print(f"📊 総パラメータ数: {total_params:,}")
            print(f"🎯 デバイス分散状況:")
            
            for device, info in device_map.items():
                param_ratio = (info['params'] / total_params) * 100
                print(f"   {device}: {info['count']} layers, {info['params']:,} params ({param_ratio:.1f}%)")
            
            # 各GPUのメモリ使用率情報を追加
            if torch.cuda.is_available():
                print(f"\n💾 GPUメモリ使用状況:")
                for i in range(torch.cuda.device_count()):
                    device_name = torch.cuda.get_device_name(i)
                    memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                    memory_cached = torch.cuda.memory_reserved(i) / 1024**3
                    memory_usage_percent = (memory_allocated / memory_total) * 100
                    
                    print(f"   🎮 GPU {i} ({device_name}): "
                          f"{memory_allocated:.2f}/{memory_total:.2f} GB "
                          f"({memory_usage_percent:.1f}%)")
                    print(f"      📦 キャッシュ: {memory_cached:.2f} GB")
            
            # モデルの実際のデバイス情報（可能な場合）
            if hasattr(model, 'hf_device_map'):
                print(f"\n🗺️ HuggingFace Device Map:")
                for layer, device in model.hf_device_map.items():
                    print(f"   {layer}: {device}")
                    
        else:
            print("⚠️ モデル情報を取得できませんでした")
            
    except Exception as e:
        print(f"❌ モデルデバイス情報の取得に失敗: {e}")
    
    print("="*60)


def get_gpu_memory_info() -> Dict[str, Dict[str, float]]:
    """GPUメモリ情報を辞書形式で取得"""
    gpu_info = {}
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device_name = torch.cuda.get_device_name(i)
            memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            memory_cached = torch.cuda.memory_reserved(i) / 1024**3
            memory_usage_percent = (memory_allocated / memory_total) * 100
            
            gpu_info[f"gpu_{i}"] = {
                "name": device_name,
                "total_gb": memory_total,
                "allocated_gb": memory_allocated,
                "cached_gb": memory_cached,
                "usage_percent": memory_usage_percent
            }
    
    return gpu_info


def get_model_device_distribution(model) -> Dict[str, Any]:
    """モデルのデバイス分散情報を取得"""
    device_map = {}
    total_params = 0
    
    if hasattr(model, 'named_parameters'):
        for name, param in model.named_parameters():
            device = str(param.device)
            if device not in device_map:
                device_map[device] = {'count': 0, 'params': 0, 'layers': []}
            device_map[device]['count'] += 1
            device_map[device]['params'] += param.numel()
            device_map[device]['layers'].append(name)
            total_params += param.numel()
    
    # パーセンテージを計算
    for device_info in device_map.values():
        device_info['params_percent'] = (device_info['params'] / total_params) * 100 if total_params > 0 else 0
    
    return {
        'total_params': total_params,
        'device_map': device_map
    }


def check_cuda_availability() -> Dict[str, Any]:
    """CUDA利用可能性の詳細情報を取得"""
    cuda_info = {
        'available': torch.cuda.is_available(),
        'device_count': 0,
        'current_device': None,
        'cuda_version': None,
        'devices': []
    }
    
    if torch.cuda.is_available():
        cuda_info['device_count'] = torch.cuda.device_count()
        cuda_info['current_device'] = torch.cuda.current_device()
        cuda_info['cuda_version'] = torch.version.cuda
        
        for i in range(torch.cuda.device_count()):
            device_info = {
                'index': i,
                'name': torch.cuda.get_device_name(i),
                'total_memory_gb': torch.cuda.get_device_properties(i).total_memory / 1024**3,
                'major': torch.cuda.get_device_properties(i).major,
                'minor': torch.cuda.get_device_properties(i).minor
            }
            cuda_info['devices'].append(device_info)
    
    return cuda_info 