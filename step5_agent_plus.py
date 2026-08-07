import json, time
from openai import RateLimitError, APITimeoutError, APIStatusError
from step1_call import client, MODEL, SYSTEM
from step3_agent import TOOLS, TOOL_MAP

# ---------- 응답 스키마 (Structured Outputs) ----------
ANSWER_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "academic_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "결론":     {"type": "string", "description": "한 문장 결론"},
                "근거문서": {"type": "string", "description": "문서명. 없으면 빈 문자열"},
                "인용":     {"type": "string", "description": "규정 원문 인용. 없으면 빈 문자열"},
                "다음행동": {"type": "string", "description": "신청 경로와 기한"},
                "확신도":   {"type": "string", "enum": ["높음", "보통", "근거없음"]},
            },
            "required": ["결론", "근거문서", "인용", "다음행동", "확신도"],
            "additionalProperties": False,
        },
    },
}

def call_with_retry(**kwargs):
    """429 / 5xx / timeout 대응 — 지수 백오프 3회"""
    for attempt in range(3):
        try:
            return client.chat.completions.create(**kwargs)
        except (RateLimitError, APITimeoutError) as e:
            wait = 2 ** attempt
            print(f"[재시도 {attempt+1}/3] {type(e).__name__} — {wait}초 대기")
            time.sleep(wait)
        except APIStatusError as e:
            if e.status_code in (500, 502, 503):
                time.sleep(2 ** attempt); continue
            raise
    raise RuntimeError("API 재시도 3회 실패")


def run_agent(question, history=None, max_turns=5, verbose=True):
    """history: system 제외한 이전 messages 배열"""
    messages = [{"role": "system", "content": SYSTEM}]
    messages += list(history or [])
    messages.append({"role": "user", "content": question})

    for turn in range(1, max_turns + 1):
        res = call_with_retry(
            model=MODEL, temperature=0.0, max_tokens=1500,
            messages=messages, tools=TOOLS,
            response_format=ANSWER_SCHEMA,
        )
        msg = res.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            data = json.loads(msg.content)          # 스키마 보장 → 파싱 실패 없음
            return data, messages[1:]               # system 제외하고 반환

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            if verbose:
                print(f"  [Act] {tc.function.name}({args})")
            try:
                out = TOOL_MAP[tc.function.name](**args)
            except Exception as e:
                out = [{"오류": str(e)}]
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": json.dumps(out, ensure_ascii=False)})

    return ({"결론": "최대 반복 초과", "근거문서": "", "인용": "",
             "다음행동": "질문을 나누어 다시 시도", "확신도": "근거없음"}, messages[1:])


if __name__ == "__main__":
    scenario = [
        "휴학하려면 어떻게 해야 하나요?",
        "저는 1학년 1학기 재학생인데요?",
        "그러면 언제부터 신청할 수 있나요?",
        "복학은요?",
    ]
    history = []
    for q in scenario:
        print(f"\n👤 {q}")
        data, history = run_agent(q, history)
        print(f"🤖 [{data['확신도']}] {data['결론']}")
        print(f"   근거: {data['근거문서']} — {data['인용'][:60]}")
        print(f"   다음: {data['다음행동']}")