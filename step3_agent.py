import json
from step1_call import client, MODEL, SYSTEM
from rag import search_document

TOOLS = [{
    "type": "function",
    "function": {
        "name": "search_document",
        "description": (
            "대학 학사 규정•안내 문서를 의미 기반으로 검색해 관련 원문 청크를 반환한다. "
            "졸업요건•수강신청•휴학•복학•성적평가•학사일정에 관한 질문에는 "
            "답변 전에 반드시 이 함수를 먼저 호출한다."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "검색할 문장. 사용자의 구어체를 규정 용어로 바꿔서 넣는다. "
                                   "예: '복전 몇 학점' → '복수전공 졸업 이수학점'"
                },
                "top_k": {"type": "integer", "description": "반환 청크 수 (기본 3)"}
            },
            "required": ["query"],
        },
    },
}]

TOOL_MAP = {"search_document": search_document}


def run_agent(question, max_turns=5, verbose=True):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]

    for turn in range(1, max_turns + 1):
        res = client.chat.completions.create(
            model=MODEL, temperature=0.0, max_tokens=1500,
            messages=messages, tools=TOOLS,
        )
        msg = res.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        # --- 최종 답변 ---
        if not msg.tool_calls:
            if verbose:
                print(f"[{turn}턴 Answer]")
            return msg.content

        # --- Act: 함수 실제 실행 ---
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            if verbose:
                print(f"[{turn}턴 Act]     {tc.function.name}({args})")
            try:
                output = TOOL_MAP[tc.function.name](**args)
            except Exception as e:
                output = [{"오류": str(e)}]
            if verbose:
                scores = [h.get("유사도") for h in output]
                print(f"[{turn}턴 Observe] {len(output)}건 / 유사도 {scores}")

            # --- Observe: 결과 재투입 ---
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(output, ensure_ascii=False),
            })

    return "최대 반복 횟수를 초과했습니다. 질문을 나누어 다시 시도해 주세요."


if __name__ == "__main__":
    for i, q in enumerate(open("test_questions.txt", encoding="utf-8"), 1):
        q = q.strip()
        if not q:
            continue
        print(f"\n{'='*60}\nQ{i}. {q}\n{'-'*60}")
        print(run_agent(q))