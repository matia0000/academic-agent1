import rag

QUERIES = ["복전하면 몇 학점 들어야 해요?", "조기졸업 조건", "성적 이의신청 언제"]

for size, overlap in [(200, 0), (400, 80), (1200, 0)]:
    rag.build_index(size, overlap)
    rag._INDEX = None                      # 캐시 초기화
    print(f"\n########## size={size}, overlap={overlap} ##########")
    for q in QUERIES:
        top = rag.search_document(q, top_k=1)[0]
        print(f"  {q} → [{top['유사도']}] {top['문서']} / {top['내용'][:50]}...")