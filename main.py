import os, uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import rag
from step5_agent_plus import run_agent

@asynccontextmanager
async def lifespan(app: FastAPI):
    rag.load_index()          # 서버 기동 시 인덱스 1회 로딩 (요청마다 재로딩 방지)
    yield

app = FastAPI(title="학사도우미 API", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],              # 교육용. 운영 시 Lovable 도메인만 허용
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

SESSIONS = {}                         # 교육용 임시 메모리 (재시작 시 초기화)

class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=500)
    session_id: str | None = None

class AskResponse(BaseModel):
    결론: str
    근거문서: str
    인용: str
    다음행동: str
    확신도: str
    session_id: str

@app.get("/health")
def health():
    return {"status": "ok",
            "chunks": len(rag._INDEX) if rag._INDEX else 0,
            "docs": len(os.listdir("docs"))}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    sid = req.session_id or str(uuid.uuid4())
    history = SESSIONS.get(sid, [])
    try:
        data, new_history = run_agent(req.question, history, verbose=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent 실행 실패: {e}")
    SESSIONS[sid] = new_history[-12:]
    return AskResponse(**data, session_id=sid)