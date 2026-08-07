# step5_stream.py
from step1_call import client, MODEL

stream = client.chat.completions.create(
    model="gpt-4o-mini", stream=True,
    messages=[
        {"role": "user", "content": "대학교 RAG 시스템이 무엇인지 한국어로 200자 이상 설명해줘"}
    ],
)
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
print()