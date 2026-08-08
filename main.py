import os, uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from supabase import create_client, Client
from datetime import datetime
import rag
from step5_agent_plus import run_agent

@asynccontextmanager
async def lifespan(app: FastAPI):
    rag.load_index()
    yield

app = FastAPI(title="학사도우미 API", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

SESSIONS = {}

# ──────────────────────────────────────────
# /ask 엔드포인트
# ──────────────────────────────────────────

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

# ──────────────────────────────────────────
# Supabase 상담 신청 엔드포인트
# ──────────────────────────────────────────

def get_supabase() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise HTTPException(status_code=500, detail="Supabase 환경변수 미설정")
    return create_client(url, key)

class CounselingRequest(BaseModel):
    name: str
    student_id: str
    phone: str | None = None
    email: str | None = None
    type: str
    date: str
    time_slot: str | None = None
    message: str

@app.post("/counseling")
def create_counseling(req: CounselingRequest):
    try:
        sb = get_supabase()
        data = {
            "name": req.name,
            "student_id": req.student_id,
            "type": req.type,
            "date": req.date,
            "message": req.message,
        }
        result = sb.table("counseling").insert(data).execute()
        saved_id = result.data[0]["id"] if result.data else None
        return {"status": "ok", "id": saved_id}
    except HTTPException:
        raise
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/counseling")
def get_counseling(student_id: str | None = None):
    try:
        sb = get_supabase()
        query = sb.table("counseling").select("*").order("created_at", desc=True)
        if student_id:
            query = query.eq("student_id", student_id)
        result = query.execute()
        return {"status": "ok", "data": result.data}
    except HTTPException:
        raise
    except Exception as e:
        return {"status": "error", "message": str(e)}
