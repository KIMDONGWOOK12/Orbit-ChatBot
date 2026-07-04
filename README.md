# Orbit-ChatBot

개발 기술 문서 RAG 에이전트 챗봇 — LangChain/LangGraph 기반, 직접 학습시킨 miniGPT 연동을 목표로 하는 개인 프로젝트입니다.

Dev-docs RAG agent chatbot built with LangChain/LangGraph, aiming to integrate a self-trained miniGPT as the generator.

---

## 1. 프로젝트 소개

한국어 위키백과 개발 기술 문서(파이썬, 자료구조, 알고리즘, 객체지향)를 검색해 근거 기반으로 답변하는 챗봇입니다.

단순 RAG에서 출발해, 질문 유형을 스스로 판단하는 Agent로 확장했고, 최종적으로 직접 학습시킨 miniGPT를 답변 생성 모델로 연동하는 것을 목표로 합니다.

핵심 특징:

- 질문을 보고 검색 필요 여부를 LLM이 스스로 판단 (LangGraph 조건부 분기)
- 자료에 없는 질문은 일반 지식으로 답하되 출처를 구분해 표시
- LangSmith Dataset 평가로 답변 품질을 수치로 검증 (키워드 매칭 vs LLM-as-Judge 비교)
- 환경변수 하나로 LLM 프로바이더 전환 (Google Gemini ↔ Anthropic Claude ↔ Ollama)

---

## 2. 아키텍처

```
사용자 질문
    ↓
[judge]  질문 유형 판단 (LLM)
    ↓
 조건부 분기 ─────────────┐
    ↓ 검색 필요            ↓ 검색 불필요
[search] 벡터 검색         │
    ↓                     ↓
[generate] 답변 생성 (문서 근거 / 일반 지식)
    ↓
답변 반환
```

| 구성 요소 | 기술 |
|---|---|
| 오케스트레이션 | LangGraph (StateGraph, Conditional Edges) |
| 검색 | Chroma + Google gemini-embedding-001 |
| 답변 생성 | Gemini / Claude (전환 가능) → miniGPT (목표) |
| 평가·추적 | LangSmith (Tracing, Dataset Evaluation) |
| API 서빙 | FastAPI |

---

## 3. 진행 상황

| 단계 | 내용 | 상태 |
|---|---|---|
| RAG 파이프라인 구축 | 문서 로딩 → 분할 → 임베딩 → 검색 → 생성 (LCEL) | 완료 |
| LangSmith 평가 | 키워드 매칭 + LLM-as-Judge 이중 평가 | 완료 |
| StateGraph 마이그레이션 | LCEL 체인 → State/Node/Edge 그래프 구조 | 완료 |
| Agent 확장 | judge 노드 + 조건부 엣지로 검색 여부 자율 판단 | 완료 |
| LLM 프로바이더 전환 구조 | Gemini ↔ Claude ↔ Ollama 환경변수 전환 | 완료 |
| FastAPI 배포 | Agent를 REST API로 래핑 | 진행 중 |
| miniGPT 연동 | 직접 학습시킨 모델을 생성기로 통합 | 예정 |

---

## 4. 로드맵

부트캠프 커리큘럼을 따라 단계적으로 발전시킵니다.

### Phase 1 — RAG & Agent (완료)
- LangChain LCEL 기반 RAG 파이프라인
- LangSmith Tracing + Dataset 평가
- LangGraph StateGraph 마이그레이션
- Conditional Edges 기반 Agent (검색 여부 자율 판단)

### Phase 2 — Agent 고도화 (진행 중)
- FastAPI REST API 배포
- Tool Calling / ReAct Pattern 적용 검토
- Plan-and-Execute Pattern 적용 검토
- Checkpointer(대화 상태 저장), Human-in-the-Loop 검토
- MCP(Model Context Protocol) 연동 검토

### Phase 3 — 나만의 모델 연동 (핵심 목표)
- 직접 학습시킨 miniGPT를 답변 생성기로 통합 (LangChain Custom LLM)
- 검색(임베딩)과 생성(miniGPT)의 역할 분리 운영
- LLM 경량화 적용 검토: Quantization(INT8/INT4, GGUF), PEFT(LoRA/QLoRA)

### Phase 4 — 서빙 & 배포
- 서빙 엔진 검토: vLLM / Ollama (Modelfile)
- Docker 컨테이너화, Docker Compose
- CI/CD: GitHub Actions, 무중단 배포(Blue/Green, Rolling)
- 클라우드 배포: AWS (EC2, S3, ELB 등)

---

## 5. 평가 결과 (현재 버전)

LangSmith Dataset(개발 문서 3문항)에 두 가지 평가자를 적용했습니다.

| 평가자 | 방식 | 점수 |
|---|---|---|
| contains_expected_keyword | 기대 답변 키워드 포함 여부 (글자 매칭) | 1.00 |
| llm_judge_semantic_match | 의미 일치 여부를 LLM이 판단 | 1.00 |

과정에서 얻은 인사이트: 키워드 매칭은 "데이터"와 "자료"처럼 뜻이 같아도 표현이 다르면 0점을 주지만, LLM-as-Judge는 의미를 보고 판단합니다. 단순 글자 비교의 한계와 의미 기반 평가의 필요성을 점수로 직접 확인했습니다.

---

## 6. 실행 방법

### 6.1 환경 구성 (uv)

```bash
uv sync
```

### 6.2 `.env` 설정

```env
GOOGLE_API_KEY=your_gemini_api_key        # 임베딩용
ANTHROPIC_API_KEY=your_claude_api_key     # 생성용 (선택)
LLM_PROVIDER=google                        # google / anthropic / ollama
GOOGLE_MODEL=gemini-2.5-flash

LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=orbit-chatbot
```

### 6.3 실행

```bash
# RAG Agent + 평가
uv run baseline.py

# FastAPI 서버
uv run uvicorn main:app --reload
# Swagger UI: http://127.0.0.1:8000/docs

# 터미널 대화형 챗봇
uv run chat.py
```

---

## 7. 트러블슈팅 기록

| 문제 | 원인 | 해결 |
|---|---|---|
| 위키 "파이썬" 검색 시 답변 불가 | URL이 동물(비단뱀) 페이지 | `파이썬_(프로그래밍_언어)`로 교체 — Garbage In, Garbage Out |
| 임베딩 429 (분당 한도) | chunk 수백 개 일괄 임베딩 | chunk 수 제한 |
| 생성 429 (일일 한도) | Agent 전환으로 질문당 LLM 호출 2배 (judge+generate) | LLM 프로바이더를 Claude로 전환 — build_llm() 분기 구조 활용 |
| 조건부 엣지 노드 이름 오류 | 라우팅 함수 반환값과 등록 노드 불일치 | 반환 가능한 노드 이름을 목록으로 제한 |

---

## 8. 배운 것

- RAG는 "학습"이 아니라 "검색"이다 — 모델은 그대로 두고 자료를 연결하는 구조
- Agent = 판단이 추가된 워크플로 — LLM을 생성뿐 아니라 의사결정에도 사용
- 판단 노드는 유연함을 주는 대신 호출 비용이 늘어난다 (트레이드오프)
- 좋은 입력이 좋은 결과를 만든다 — 자료 선택과 평가 설계가 품질의 출발점
