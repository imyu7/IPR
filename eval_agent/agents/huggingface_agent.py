import logging
import torch
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gc
from .base import LMAgent

logger = logging.getLogger("agent_frame")


class HuggingFaceAgent(LMAgent):
    """Hugging Faceモデルを使用してローカルで推論を行うエージェント"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 必須パラメータの確認
        assert "model_name" in config, "model_nameが設定されていません"
        
        self.model_name = config["model_name"]
        self.temperature = config.get("temperature", 0.8)
        self.max_new_tokens = config.get("max_new_tokens", 512)
        self.top_p = config.get("top_p", 0.9)
        self.top_k = config.get("top_k", 50)
        self.do_sample = config.get("do_sample", True)
        self.device = config.get("device", "auto")
        self.load_in_8bit = config.get("load_in_8bit", False)
        self.load_in_4bit = config.get("load_in_4bit", False)
        self.trust_remote_code = config.get("trust_remote_code", False)
        self.torch_dtype = config.get("torch_dtype", "auto")
        
        logger.info(f"Hugging Faceモデル '{self.model_name}' を初期化しています...")
        
        # デバイスの設定
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 量子化設定
        quantization_config = None
        if self.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif self.load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # torch_dtypeの設定
        if self.torch_dtype == "auto":
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        else:
            torch_dtype = getattr(torch, self.torch_dtype)
        
        try:
            # トークナイザーの読み込み
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                padding_side="left"
            )
            
            # パディングトークンの設定
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # モデルの読み込み
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map=self.device if quantization_config is None else "auto",
                quantization_config=quantization_config,
                trust_remote_code=self.trust_remote_code,
                low_cpu_mem_usage=True
            )
            
            # 量子化を使用しない場合のデバイス移動
            if quantization_config is None and self.device != "auto":
                self.model = self.model.to(self.device)
            
            # 評価モードに設定
            self.model.eval()
            
            logger.info(f"モデルの初期化が完了しました。デバイス: {self.device}")
            
        except Exception as e:
            logger.error(f"モデルの初期化に失敗しました: {e}")
            raise
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """メッセージリストをプロンプト文字列に変換"""
        formatted_prompt = ""
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                formatted_prompt += f"System: {content}\n\n"
            elif role == "user":
                formatted_prompt += f"Human: {content}\n\n"
            elif role == "assistant":
                formatted_prompt += f"Assistant: {content}\n\n"
        
        # 最後にAssistantの応答を促すプロンプトを追加
        formatted_prompt += "Assistant:"
        
        return formatted_prompt
    
    def _generate_response(self, prompt: str) -> str:
        """プロンプトから応答を生成"""
        try:
            # model_max_lengthが非常に大きい場合の対処
            max_model_length = self.tokenizer.model_max_length
            if max_model_length > 100000:  # 異常に大きい値の場合
                max_model_length = 4096  # デフォルト値を使用
            
            # トークン化
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_model_length - self.max_new_tokens
            )
            
            # デバイスに移動
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # 生成設定
            generation_config = {
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "do_sample": self.do_sample,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            
            # ストップワードの設定
            if self.stop_words:
                stop_token_ids = []
                for stop_word in self.stop_words:
                    stop_tokens = self.tokenizer.encode(stop_word, add_special_tokens=False)
                    stop_token_ids.extend(stop_tokens)
                if stop_token_ids:
                    generation_config["eos_token_id"] = list(set(stop_token_ids + [self.tokenizer.eos_token_id]))
            
            # 推論実行
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_config
                )
            
            # 生成されたテキストをデコード
            generated_text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            
            # ストップワードで切り取り
            for stop_word in self.stop_words:
                if stop_word in generated_text:
                    generated_text = generated_text.split(stop_word)[0]
            
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"応答生成中にエラーが発生しました: {e}")
            return ""
        finally:
            # メモリクリーンアップ
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    def __call__(self, messages: List[Dict[str, str]]) -> str:
        """メッセージリストから応答を生成"""
        try:
            # メッセージをプロンプトに変換
            prompt = self._format_messages(messages)
            
            # 応答を生成
            response = self._generate_response(prompt)
            
            logger.debug(f"生成された応答: {response[:100]}...")
            
            return response
            
        except Exception as e:
            logger.error(f"応答生成に失敗しました: {e}")
            return ""
    
    def __del__(self):
        """デストラクタでリソースをクリーンアップ"""
        try:
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'tokenizer'):
                del self.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except:
            pass 