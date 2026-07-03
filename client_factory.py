"""
Factory chọn LLM client theo backend cấu hình (mục 5 yêu cầu bổ sung: hỗ trợ llama.cpp
server như 1 backend thay thế Ollama). Cả 2 client (OllamaClient, LlamaCppClient) cùng
implement interface .embed() / .embed_batch() / .chat() / .judge() nên phần còn lại của hệ
thống (MemoryStore, QAService, ingest, scheduler, eval) không cần biết đang chạy backend nào.
"""
import config
from ollama_client import OllamaClient
from llamacpp_client import LlamaCppClient

BACKENDS = ("ollama", "llamacpp")


def get_client(backend: str = None, **kwargs):
    """
    backend: "ollama" | "llamacpp". Mặc định lấy config.LLM_BACKEND (env MEMOS_LLM_BACKEND).
    kwargs được truyền thẳng vào constructor của client tương ứng (vd host, chat_model...).
    """
    backend = (backend or config.LLM_BACKEND or "ollama").lower()
    if backend == "llamacpp":
        return LlamaCppClient(**kwargs)
    if backend == "ollama":
        return OllamaClient(**kwargs)
    raise ValueError(f"Backend không hợp lệ: {backend!r}. Chọn 1 trong {BACKENDS}")
