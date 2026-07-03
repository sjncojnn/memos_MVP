"""
Client tối giản gọi llama.cpp server (llama-server) qua REST API.

Đây là backend LLM THAY THẾ cho Ollama (mục 5 yêu cầu bổ sung), cho phép demo cơ chế
"prefix/prompt cache" THẬT của llama.cpp thay vì chỉ semantic answer cache như hiện tại.

Cách chạy llama-server (build từ https://github.com/ggml-org/llama.cpp hoặc `brew install
llama.cpp` trên macOS):

    llama-server \
        -m ./models/qwen2.5-7b-instruct-q4_k_m.gguf \
        -c 8192 --port 8080 \
        --embeddings                       # BẮT BUỘC để bật endpoint embedding
    # Nếu dùng model embedding riêng (khuyến nghị), chạy 1 tiến trình llama-server thứ 2
    # trên port khác (vd 8081) chỉ để phục vụ embedding:
    llama-server -m ./models/nomic-embed-text-v1.5.Q8_0.gguf -c 2048 --port 8081 --embeddings

Cơ chế prefix cache (KV-cache reuse) THẬT trong llama.cpp server:
  - Server giữ 1 KV-cache riêng cho mỗi "slot" (số lượng slot cấu hình qua `--parallel N`,
    mặc định 1). Khi 2 request liên tiếp gửi vào CÙNG 1 slot có phần ĐẦU prompt (prefix)
    giống hệt nhau, server chỉ tính lại phần đuôi khác biệt -> giảm mạnh time-to-first-token.
  - Điều kiện để hưởng lợi: (1) `cache_prompt: true` trong request (mặc định true ở bản
    llama.cpp mới, nhưng client này set tường minh cho chắc), (2) route ổn định về cùng 1
    `id_slot` giữa các lượt hỏi liên tiếp — client này cố định `id_slot` theo
    `config.LLAMACPP_SLOT_ID` thay vì để server tự chọn slot rảnh (round-robin có thể phá
    cache nếu chạy nhiều slot song song).
  - System prompt CỐ ĐỊNH (SYSTEM_PROMPT, giống ollama_client.py) luôn được đặt ở đầu prompt
    -> đây chính là phần prefix lặp lại giữa các lượt hỏi, được hưởng lợi nhiều nhất từ cache.

Giới hạn đã biết: nếu context (system + retrieved context + câu hỏi) đổi ngay từ đầu (vd
retrieved context khác nhau mỗi câu hỏi vì nội dung tri thức khác nhau), phần prefix dùng
chung thực tế chỉ là SYSTEM_PROMPT — vẫn có lợi nhưng nhỏ hơn nhiều so với cache toàn bộ
prompt. Muốn tối đa hoá cache hit, có thể đặt các câu hỏi liên tiếp trong cùng 1 category để
context trùng nhau nhiều hơn. Nếu server/version hiện tại không hỗ trợ hoặc không cho hiệu
quả cache rõ ràng, hệ thống vẫn fallback về semantic QA cache (qa_service.py) như bình thường.
"""
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


class LlamaCppError(RuntimeError):
    pass


class LlamaCppClient:
    def __init__(self, host: str = None, embed_host: str = None, slot_id: int = None,
                 chat_model: str = None, embed_model: str = None):
        """
        host       : URL llama-server phục vụ chat/completion (mặc định config.LLAMACPP_HOST)
        embed_host : URL llama-server phục vụ embedding (mặc định = host, nếu bạn chạy 2 tiến
                     trình llama-server riêng cho chat và embedding thì truyền host riêng)
        slot_id    : id_slot cố định để tối đa hoá prefix-cache hit (mặc định config.LLAMACPP_SLOT_ID)
        chat_model / embed_model: llama.cpp server chỉ load 1 model / tiến trình nên tham số
                     này chỉ mang tính thông tin (không gửi lên server), giữ lại để tương thích
                     interface với OllamaClient.
        """
        self.host = (host or config.LLAMACPP_HOST).rstrip("/")
        self.embed_host = (embed_host or self.host).rstrip("/")
        self.slot_id = config.LLAMACPP_SLOT_ID if slot_id is None else slot_id
        self.chat_model = chat_model
        self.embed_model = embed_model

    # ------------------------------------------------------------- embed
    def embed(self, text: str) -> np.ndarray:
        """Gọi endpoint native /embedding (yêu cầu server chạy với --embeddings)."""
        url = f"{self.embed_host}/embedding"
        resp = requests.post(url, json={"content": text}, timeout=60)
        if resp.status_code != 200:
            raise LlamaCppError(f"Embedding call failed ({resp.status_code}): {resp.text}")
        data = resp.json()
        vec = self._extract_embedding(data)
        return np.array(vec, dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [self.embed(t) for t in texts]

    @staticmethod
    def _extract_embedding(data) -> list[float]:
        """
        llama.cpp server có vài biến thể format response tuỳ version/pooling:
          - {"embedding": [floats]}
          - {"embedding": [[floats]]}            (pooling='none' hoặc multi-token)
          - [{"embedding": [floats]}, ...]        (khi content là list)
        Hàm này cố gắng dò và trả về 1 vector float duy nhất.
        """
        if isinstance(data, list):
            if not data:
                raise LlamaCppError("Embedding response rỗng")
            data = data[0]
        vec = data.get("embedding")
        if vec is None:
            raise LlamaCppError(f"Không tìm thấy field 'embedding' trong response: {data}")
        if isinstance(vec, list) and vec and isinstance(vec[0], list):
            vec = vec[0]
        return vec

    # -------------------------------------------------------------- chat
    def chat(self, user_prompt: str, context: str = "", temperature: float = 0.1) -> dict:
        """
        Gọi endpoint OpenAI-compatible /v1/chat/completions với `cache_prompt: true` +
        `id_slot` cố định để tận dụng prefix-cache THẬT của llama.cpp server (khác với
        ollama_client.py chỉ dựa vào cache prefix nội bộ ẩn, không kiểm soát được slot).
        Trả về dict {answer, latency_sec}.
        """
        full_user_msg = f"NGỮ CẢNH:\n{context}\n\nCÂU HỎI: {user_prompt}"
        url = f"{self.host}/v1/chat/completions"
        payload = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": full_user_msg},
            ],
            "temperature": temperature,
            "stream": False,
            "cache_prompt": True,
            "id_slot": self.slot_id,
        }
        if self.chat_model:
            payload["model"] = self.chat_model
        t0 = time.time()
        resp = requests.post(url, json=payload, timeout=180)
        latency = time.time() - t0
        if resp.status_code != 200:
            raise LlamaCppError(f"Chat call failed ({resp.status_code}): {resp.text}")
        data = resp.json()
        try:
            answer = data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as e:
            raise LlamaCppError(f"Response không đúng định dạng OpenAI-chat: {data}") from e
        return {"answer": answer, "latency_sec": latency}

    # ------------------------------------------------------------- judge
    def judge(self, question: str, reference: str, candidate: str) -> float:
        """Dùng chính model local để chấm điểm 1-5 (LLM-judge, mục 5 Problem Statement)."""
        prompt = (
            "Bạn là giám khảo chấm chất lượng câu trả lời nghiệp vụ. "
            "Cho điểm 1-5 (5 là tốt nhất) dựa trên mức độ trùng khớp NỘI DUNG giữa "
            "câu trả lời tham chiếu và câu trả lời cần chấm. Chỉ trả về 1 số nguyên duy nhất.\n\n"
            f"Câu hỏi: {question}\nĐáp án tham chiếu: {reference}\nCâu trả lời cần chấm: {candidate}\n\nĐiểm:"
        )
        url = f"{self.host}/v1/chat/completions"
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "stream": False,
            "cache_prompt": True,
            "id_slot": self.slot_id,
        }
        resp = requests.post(url, json=payload, timeout=120)
        if resp.status_code != 200:
            raise LlamaCppError(f"Judge call failed ({resp.status_code}): {resp.text}")
        data = resp.json()
        try:
            text = data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError):
            return 0.0
        for token in text.split():
            token = token.strip(".,")
            if token.isdigit():
                return float(token)
        return 0.0

    # ------------------------------------------------------------- health
    def health_check(self) -> bool:
        try:
            resp = requests.get(f"{self.host}/health", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False
