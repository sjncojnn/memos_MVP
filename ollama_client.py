"""
Client tối giản gọi Ollama local qua REST API (http://localhost:11434).
Không phụ thuộc SDK ngoài — chỉ dùng `requests` để dễ cài đặt trên macOS.

Yêu cầu trước khi chạy thật:
    ollama pull llama3.2:3b       # hoặc model chat khác nhẹ hơn/nặng hơn tuỳ máy
    ollama pull nomic-embed-text  # model embedding

Có STABLE system prompt cố định (SYSTEM_PROMPT) để tận dụng cơ chế cache prefix
nội bộ của Ollama/llama.cpp: khi phần đầu prompt giống hệt lần gọi trước, engine
tái sử dụng KV-cache đã tính thay vì tính lại toàn bộ -> giảm time-to-first-token.
Đây là điểm hiện thực cho yêu cầu "tái sử dụng tri thức ổn định dưới dạng trạng
thái suy luận (KV) để giảm độ trễ" trong Problem Statement mục 4.
"""
import json
import time
import numpy as np
import requests

import config

SYSTEM_PROMPT = (
    "Bạn là trợ lý hỏi đáp nghiệp vụ cho khách hàng của kênh bán ViettelPay Pro. "
    "Chỉ trả lời dựa trên NGỮ CẢNH được cung cấp bên dưới. "
    "Nếu ngữ cảnh không đủ thông tin, hãy nói rõ là chưa có dữ liệu, không suy đoán. "
    "Trả lời ngắn gọn, chính xác, đúng nghiệp vụ, bằng tiếng Việt."
)


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, host: str = None, chat_model: str = None, embed_model: str = None):
        self.host = (host or config.OLLAMA_HOST).rstrip("/")
        self.chat_model = chat_model or config.CHAT_MODEL
        self.embed_model = embed_model or config.EMBED_MODEL

    def embed(self, text: str) -> np.ndarray:
        """Trả về vector embedding (numpy float32) cho 1 đoạn text."""
        url = f"{self.host}/api/embeddings"
        resp = requests.post(url, json={"model": self.embed_model, "prompt": text}, timeout=60)
        if resp.status_code != 200:
            raise OllamaError(f"Embedding call failed ({resp.status_code}): {resp.text}")
        data = resp.json()
        vec = np.array(data["embedding"], dtype=np.float32)
        return vec

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [self.embed(t) for t in texts]

    def chat(self, user_prompt: str, context: str = "", temperature: float = 0.1) -> dict:
        """
        Gọi model chat với system prompt CỐ ĐỊNH (để hưởng lợi prefix-cache) + ngữ cảnh
        retrieved + câu hỏi. Trả về dict {answer, latency_sec}.
        """
        full_user_msg = f"NGỮ CẢNH:\n{context}\n\nCÂU HỎI: {user_prompt}"
        url = f"{self.host}/api/chat"
        payload = {
            "model": self.chat_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": full_user_msg},
            ],
            "stream": False,
            "options": {"temperature": temperature},
        }
        t0 = time.time()
        resp = requests.post(url, json=payload, timeout=180)
        latency = time.time() - t0
        if resp.status_code != 200:
            raise OllamaError(f"Chat call failed ({resp.status_code}): {resp.text}")
        data = resp.json()
        answer = data.get("message", {}).get("content", "").strip()
        return {"answer": answer, "latency_sec": latency}

    def judge(self, question: str, reference: str, candidate: str) -> float:
        """Dùng chính model local để chấm điểm 1-5 (LLM-judge, mục 5 Problem Statement)."""
        prompt = (
            "Bạn là giám khảo chấm chất lượng câu trả lời nghiệp vụ. "
            "Cho điểm 1-5 (5 là tốt nhất) dựa trên mức độ trùng khớp NỘI DUNG giữa "
            "câu trả lời tham chiếu và câu trả lời cần chấm. Chỉ trả về 1 số nguyên duy nhất.\n\n"
            f"Câu hỏi: {question}\nĐáp án tham chiếu: {reference}\nCâu trả lời cần chấm: {candidate}\n\nĐiểm:"
        )
        url = f"{self.host}/api/generate"
        resp = requests.post(
            url, json={"model": self.chat_model, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0.0}}, timeout=120
        )
        if resp.status_code != 200:
            raise OllamaError(f"Judge call failed ({resp.status_code}): {resp.text}")
        text = resp.json().get("response", "").strip()
        for token in text.split():
            token = token.strip(".,")
            if token.isdigit():
                return float(token)
        return 0.0
