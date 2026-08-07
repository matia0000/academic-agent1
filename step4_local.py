import re, time, json, requests
import numpy as np
import rag
from step3_agent import run_agent

OLLAMA = "http://localhost:11434"
SENSITIVE = ("20211234 김민준 학생이 지난 학기 F를 받았는데 "
             "재수강하면 성적이 대체되나요? 연락처는 010-1234-5678입니다.")

# ---------- 마스킹 ----------
def mask(text):
    text = re.sub(r"\b20\d{6}\b", "[학번]", text)
    text = re.sub(r"01[016-9]-?\d{3,4}-?\d{4}", "[연락처]", text)
    text = re.sub(r"[가-힣]{2,4} 학생", "[학생]", text)
    return text

# ---------- 로컬 임베딩 ----------
def embed_local(texts, model="bge-m3"):
    r = requests.post(f"{OLLAMA}/api/embed",
                      json={"model": model, "input": texts}, timeout=300)
    v = np.array(r.json()["embeddings"], dtype="float32")
    return v / np.linalg.norm(v, axis=1, keepdims=True)

# ---------- 로컬 생성 ----------
def chat_local(prompt, model="qwen2.5:7b"):
    t0 = time.time()
    r = requests.post(f"{OLLAMA}/api/chat", json={
        "model": model, "stream": False,
        "messages": [{"role": "user", "content": prompt}],
    }, timeout=300)
    return r.json()["message"]["content"], round(time.time() - t0, 1)


if __name__ == "__main__":
    print("원문  :", SENSITIVE)
    print("마스킹:", mask(SENSITIVE))

    print("\n--- ① 클라우드 Agent (마스킹 후) ---")
    print(run_agent(mask(SENSITIVE), verbose=False))

    print("\n--- ② 완전 로컬 RAG (원문 그대로, 외부 전송 0) ---")
    records = json.load(open("index.json", encoding="utf-8"))
    texts = [r["내용"] for r in records]
    M = embed_local(texts)                      # 로컬 인덱스 재구축
    qv = embed_local([SENSITIVE])[0]
    top = np.argsort(-(M @ qv))[:3]
    context = "\n---\n".join(f"[{records[i]['문서']}] {records[i]['내용']}" for i in top)
    ans, sec = chat_local(
        f"다음 학사 규정만 근거로 답하라. 규정에 없으면 모른다고 답하라.\n\n"
        f"{context}\n\n질문: {SENSITIVE}"
    )
    print(f"({sec}초)\n{ans}")