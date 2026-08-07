import os, json, glob
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM   = 1536          # 512로 줄이면 index.json 용량이 1/3 (7교시 팁 참고)
DOC_DIR     = "docs"
INDEX_PATH  = "index.json"


# ---------- 1) 청킹 ----------
def chunk_text(text, size=400, overlap=80):
    """단락(빈 줄 2개) 우선 + 길이 상한 + 중첩"""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if buf and len(buf) + len(p) + 1 > size:
            chunks.append(buf)
            buf = (buf[-overlap:] + "\n" + p) if overlap else p
        else:
            buf = f"{buf}\n{p}" if buf else p
    if buf:
        chunks.append(buf)
    return chunks


# ---------- 2) 임베딩 ----------
def embed(texts):
    """문자열 리스트 → L2 정규화된 벡터 배열 (한 번에 배치 호출)"""
    res = client.embeddings.create(
        model=EMBED_MODEL, input=texts, dimensions=EMBED_DIM
    )
    vecs = np.array([d.embedding for d in res.data], dtype="float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)   # 정규화 → 내적=코사인
    return vecs


# ---------- 3) 인덱스 구축 ----------
def build_index(size=400, overlap=80):
    records = []
    for path in sorted(glob.glob(os.path.join(DOC_DIR, "*.txt"))):
        name = os.path.basename(path)
        text = open(path, encoding="utf-8").read()
        for i, ch in enumerate(chunk_text(text, size, overlap)):
            records.append({"문서": name, "위치": f"청크 {i+1}", "내용": ch})

    print(f"총 청크 수: {len(records)}")
    vecs = embed([r["내용"] for r in records])
    for r, v in zip(records, vecs):
        r["임베딩"] = v.tolist()

    json.dump(records, open(INDEX_PATH, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"저장 완료 → {INDEX_PATH} "
          f"({os.path.getsize(INDEX_PATH)/1024/1024:.1f} MB)")
    return records


# ---------- 4) 검색 ----------
_INDEX, _MATRIX = None, None

def load_index():
    global _INDEX, _MATRIX
    _INDEX = json.load(open(INDEX_PATH, encoding="utf-8"))
    _MATRIX = np.array([r["임베딩"] for r in _INDEX], dtype="float32")
    print(f"인덱스 로딩: 청크 {len(_INDEX)}개")

def search_document(query: str, top_k: int = 3, min_score: float = 0.35):
    if _INDEX is None:
        load_index()
    qv = embed([query])[0]
    scores = _MATRIX @ qv
    order = np.argsort(-scores)[:top_k]
    hits = [{
        "문서": _INDEX[i]["문서"],
        "위치": _INDEX[i]["위치"],
        "유사도": round(float(scores[i]), 3),
        "내용": _INDEX[i]["내용"],
    } for i in order if scores[i] >= min_score]
    return hits or [{"문서": None, "유사도": 0, "내용": "검색 결과 없음"}]


if __name__ == "__main__":
    build_index()
    for q in ["복전하면 몇 학점 들어야 해요?", "수업 뺄 수 있는 기간", "군대 가면 휴학"]:
        print(f"\nQ: {q}")
        for h in search_document(q):
            print(f"  [{h['유사도']}] {h['문서']} {h['위치']} — {h['내용'][:60]}...")