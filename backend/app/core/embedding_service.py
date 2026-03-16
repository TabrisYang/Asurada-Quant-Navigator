"""阿斯拉量化系統 — 本地語意嵌入服務

將文字轉為固定長度的向量，使語意相似的問句在向量空間中距離相近。
- 使用 sentence-transformers 的多語言模型（支援中英混合）
- 延遲載入：首次呼叫時才載入模型（約 3-5 秒）
- 向量已正規化：dot product = cosine similarity
- 若 sentence-transformers 未安裝，功能自動停用（graceful degradation）
"""

import threading
from typing import Optional

import numpy as np
from loguru import logger

_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_VECTOR_DIM = 384

_model = None
_available: bool | None = None  # None = 尚未嘗試載入
_lock = threading.Lock()


def _lazy_load():
    """延遲載入嵌入模型（thread-safe）"""
    global _model, _available

    if _available is not None:
        return

    with _lock:
        if _available is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"正在載入嵌入模型 {_MODEL_NAME}（首次載入需下載約 470MB）...")
            _model = SentenceTransformer(_MODEL_NAME)
            _available = True
            logger.info(f"嵌入模型已就緒: {_MODEL_NAME} ({_VECTOR_DIM} 維)")
        except ImportError:
            logger.warning(
                "sentence-transformers 未安裝，語意快取功能停用。"
                "安裝指令: pip install sentence-transformers"
            )
            _available = False
        except Exception as e:
            logger.error(f"嵌入模型載入失敗: {e}")
            _available = False


def is_available() -> bool:
    _lazy_load()
    return bool(_available)


def get_vector_dim() -> int:
    return _VECTOR_DIM


def encode(text: str) -> Optional[np.ndarray]:
    """
    將文字編碼為正規化向量。

    Returns:
        shape=(384,) 的 float32 numpy array，或 None（模型不可用時）
    """
    _lazy_load()
    if not _available or _model is None:
        return None

    try:
        vec = _model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        return vec.astype(np.float32)
    except Exception as e:
        logger.warning(f"嵌入編碼失敗: {e}")
        return None


def batch_encode(texts: list[str]) -> Optional[np.ndarray]:
    """
    批次編碼多段文字。

    Returns:
        shape=(N, 384) 的 float32 numpy array，或 None
    """
    _lazy_load()
    if not _available or _model is None:
        return None

    try:
        vecs = _model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vecs.astype(np.float32)
    except Exception as e:
        logger.warning(f"批次嵌入編碼失敗: {e}")
        return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """計算兩個正規化向量的餘弦相似度（= dot product）"""
    return float(np.dot(a, b))


def cosine_similarity_batch(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """計算 query 向量與多個候選向量的相似度，回傳相似度陣列"""
    return candidates @ query  # shape=(N,)
