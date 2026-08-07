import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "gpt-4o-mini"   # ※ 당일 platform.openai.com 모델 목록으로 확인 후 확정
SYSTEM = open("system_prompt.txt", encoding="utf-8").read()

def ask(question, temperature=0.0):
    res = client.chat.completions.create(
        model=MODEL,
        temperature=temperature,
        max_tokens=1000,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
        ],
    )
    u = res.usage
    print(f"[토큰] 입력 {u.prompt_tokens} / 출력 {u.completion_tokens} "
          f"/ finish_reason={res.choices[0].finish_reason}")
    return res.choices[0].message.content

if __name__ == "__main__":
    q = "복수전공 학생의 졸업 최저 이수학점은 몇 학점인가요?"
    for t in (0.0, 1.0):
        print(f"\n===== temperature={t} =====")
        print(ask(q, t))