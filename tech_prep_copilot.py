# ============================================================
# Tech-Prep Copilot — Google Colab + Gradio 구현
# AI 기반 맞춤형 기술 면접 & 기업 분석 에이전트
#
# [Colab 실행 순서]
# 1. 아래 셀 1(패키지 설치)을 먼저 실행
# 2. 셀 2(앱 실행)를 실행하면 Gradio 퍼블릭 URL 출력
# 3. 출력된 URL로 서비스 접속
# ============================================================

# ─────────────────────────────────────────────────────────────
# 셀 1: 패키지 설치 (Colab에서 이 블록만 별도 실행)
#
# ※ gradio>=5.0 을 사용합니다. Colab의 최신 huggingface_hub와 호환됩니다.
# ※ 설치 후 "dependency conflicts" 경고는 무시하세요 (Colab 기본 패키지 충돌).
# ※ 설치 완료 후 [런타임 → 세션 다시 시작] 후 셀 2를 실행하세요.
# ─────────────────────────────────────────────────────────────
# !pip install -q \
#   "gradio>=5.0" \
#   "langchain-openai>=0.2" \
#   "langchain-community>=0.3" \
#   "langchain-chroma>=0.1.4" \
#   "langchain-text-splitters>=0.3" \
#   pypdf \
#   beautifulsoup4

# ─────────────────────────────────────────────────────────────
# 셀 2: 앱 실행 (아래 코드 전체를 한 셀에서 실행)
# ─────────────────────────────────────────────────────────────

import gradio as gr
import json
import os
import re
import io
import textwrap
from typing import List, Tuple, Dict, Optional

import requests
from bs4 import BeautifulSoup
import pypdf

# ─── 전역 세션 상태 ───────────────────────────────────────────
_state: Dict = {
    "resume_text": "",
    "jd_text": "",
    "gap_report": None,
    "interview_system_prompt": "",
    "vectorstores": {},        # company_id -> Chroma 인스턴스
    "available_companies": [], # 등록된 기업명 목록
    "llm": None,
    "embeddings": None,
}

# ─── 유틸: LLM 초기화 ─────────────────────────────────────────
def initialize_llm(api_key: str) -> str:
    if not api_key.strip():
        return "❌ API 키를 입력하세요."
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        os.environ["OPENAI_API_KEY"] = api_key.strip()
        _state["llm"] = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        _state["embeddings"] = OpenAIEmbeddings(model="text-embedding-3-small")
        return "✅ API 키 설정 완료. 이제 다른 탭을 이용할 수 있습니다."
    except Exception as e:
        return f"❌ 초기화 실패: {e}"

# ─── 유틸: PDF 파싱 ───────────────────────────────────────────
def _parse_pdf(file_obj) -> str:
    reader = pypdf.PdfReader(file_obj)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

# ─── 탭 1: 초기 설정 — 이력서 & JD 저장 ─────────────────────
def save_inputs(pdf_file, jd_text: str) -> Tuple[str, str]:
    resume_msg, jd_msg = "", ""

    if pdf_file is not None:
        with open(pdf_file, "rb") as f:
            text = _parse_pdf(f)
        _state["resume_text"] = text
        resume_msg = f"✅ 이력서 파싱 완료 — {len(text):,}자 추출"
    else:
        resume_msg = "⚠️ PDF 파일을 업로드하세요."

    if jd_text.strip():
        _state["jd_text"] = jd_text.strip()
        jd_msg = f"✅ JD 저장 완료 — {len(jd_text):,}자"
    else:
        jd_msg = "⚠️ 채용공고 텍스트를 입력하세요."

    return resume_msg, jd_msg

# ─── 탭 2: GAP 분석 ───────────────────────────────────────────
_GAP_SYSTEM = textwrap.dedent("""\
    당신은 10년 경력의 IT 채용 컨설턴트입니다.
    주어진 이력서와 채용공고(JD)를 비교 분석하여 직무 적합성을 평가하세요.

    반드시 아래 JSON 형식으로만 응답하세요 (마크다운 코드블록 없이 순수 JSON):
    {
      "overall_score": <0~100 정수>,
      "summary": "<한두 문장 종합 평가>",
      "strengths": ["<강점1>", "<강점2>", "<강점3>"],
      "weaknesses": ["<보완점1>", "<보완점2>", "<보완점3>"],
      "recommendations": ["<추천 키워드1>", "<추천 키워드2>", "<추천 키워드3>"]
    }

    규칙:
    - 제공된 문서에 없는 수치나 기술 사실을 임의로 생성하지 마세요.
    - 근거가 부족하면 '정보가 부족하여 판단이 어렵습니다'라고 명시하세요.
""")

def run_gap_analysis() -> Tuple[str, str, str, str, str]:
    """5개 출력: score, summary, strengths, weaknesses, recommendations"""
    empty = ("", "", "", "", "")

    if not _state["llm"]:
        return ("❌ API 키를 먼저 설정하세요.",) + ("",) * 4
    if not _state["resume_text"]:
        return ("❌ 이력서를 먼저 업로드하세요.",) + ("",) * 4
    if not _state["jd_text"]:
        return ("❌ JD를 먼저 입력하세요.",) + ("",) * 4

    from langchain_core.messages import HumanMessage, SystemMessage

    user_msg = (
        f"## 이력서\n{_state['resume_text'][:3000]}\n\n"
        f"## 채용공고(JD)\n{_state['jd_text'][:2000]}\n\n"
        "위 두 문서를 비교하여 GAP 분석 JSON을 출력하세요."
    )

    try:
        resp = _state["llm"].invoke([
            SystemMessage(content=_GAP_SYSTEM),
            HumanMessage(content=user_msg),
        ])
        raw = resp.content.strip()
        # JSON 추출 (마크다운 감쌌을 경우 대비)
        m = re.search(r"\{[\s\S]+\}", raw)
        if not m:
            return (f"❌ JSON 파싱 실패:\n{raw}",) + ("",) * 4

        data = json.loads(m.group())
        _state["gap_report"] = data

        score   = f"{data.get('overall_score', '?')} / 100"
        summary = data.get("summary", "")
        strengths = "\n".join(f"• {s}" for s in data.get("strengths", []))
        weaknesses = "\n".join(f"• {w}" for w in data.get("weaknesses", []))
        recs = "\n".join(f"• {r}" for r in data.get("recommendations", []))
        return score, summary, strengths, weaknesses, recs

    except Exception as e:
        return (f"❌ 분석 중 오류: {e}",) + ("",) * 4

# ─── 탭 3: 기업 블로그 크롤링 & 벡터 인덱싱 ─────────────────
def crawl_and_index(company_name: str, blog_url: str) -> str:
    if not _state["embeddings"]:
        return "❌ API 키를 먼저 설정하세요."
    if not company_name.strip() or not blog_url.strip():
        return "❌ 기업명과 블로그 URL을 모두 입력하세요."

    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; TechPrepBot/1.0)"}
        res = requests.get(blog_url.strip(), headers=headers, timeout=15)
        res.raise_for_status()

        soup = BeautifulSoup(res.content, "html.parser")
        # 불필요한 태그 제거
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        raw_text = soup.get_text(separator="\n", strip=True)

        if len(raw_text) < 100:
            return "⚠️ 페이지에서 충분한 텍스트를 추출하지 못했습니다. URL을 확인하세요."

        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_chroma import Chroma

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        docs = splitter.create_documents(
            [raw_text],
            metadatas=[{"source": blog_url.strip(), "company": company_name.strip()}],
        )

        company_id = re.sub(r"[^a-z0-9_]", "_", company_name.strip().lower())

        if company_id in _state["vectorstores"]:
            _state["vectorstores"][company_id].add_documents(docs)
        else:
            vs = Chroma.from_documents(docs, _state["embeddings"])
            _state["vectorstores"][company_id] = vs

        if company_name.strip() not in _state["available_companies"]:
            _state["available_companies"].append(company_name.strip())

        return (
            f"✅ [{company_name}] 크롤링 완료\n"
            f"   • 수집 텍스트: {len(raw_text):,}자\n"
            f"   • 인덱싱 청크: {len(docs)}개\n"
            f"   • URL: {blog_url.strip()}"
        )

    except requests.exceptions.RequestException as e:
        return f"❌ 네트워크 오류: {e}"
    except Exception as e:
        return f"❌ 크롤링 실패: {e}"

def get_company_choices() -> List[str]:
    return _state["available_companies"] if _state["available_companies"] else []

# ─── 탭 4: 모의 면접 ──────────────────────────────────────────
_INTERVIEW_SYSTEM_TEMPLATE = textwrap.dedent("""\
    당신은 {company}의 시니어 엔지니어 면접관입니다.
    아래 이력서와 기업 기술 블로그 정보를 바탕으로 심층 기술 면접을 진행합니다.

    ## 면접 원칙
    1. 이력서의 실제 프로젝트·기술 경험에 대해 구체적으로 질문하세요.
    2. 기업 기술 블로그 내용과 연결하여 꼬리 질문을 던지세요.
    3. 한 번에 질문 하나만 하세요.
    4. 사용자 답변 후에는 먼저 **[피드백]** 섹션(기술 정확성 + 블로그 근거 인용)을
       제공하고, 이어서 **[다음 질문]** 섹션을 작성하세요.
    5. 제공된 문서에 없는 사실을 임의로 만들지 마세요.

    ## 이력서 (요약)
    {resume}

    ## 기업 기술 블로그 참고 정보
    {context}
""")

def start_interview(company_name: str) -> Tuple[List, str, str]:
    """면접 시작 → (chat_history, status_msg, system_prompt_state)"""
    if not _state["llm"]:
        return [], "❌ API 키를 먼저 설정하세요.", ""
    if not _state["resume_text"]:
        return [], "❌ 이력서를 먼저 업로드하세요.", ""
    if not company_name:
        return [], "❌ 기업을 선택하세요.", ""

    from langchain_core.messages import HumanMessage, SystemMessage

    company_id = re.sub(r"[^a-z0-9_]", "_", company_name.lower())
    context_chunks = []

    if company_id in _state["vectorstores"]:
        docs = _state["vectorstores"][company_id].similarity_search(
            _state["resume_text"][:500], k=3
        )
        context_chunks = [d.page_content for d in docs]

    context_text = (
        "\n\n---\n\n".join(context_chunks)
        if context_chunks
        else "(수집된 블로그 정보 없음 — 이력서 기반으로 일반 면접 진행)"
    )

    system_prompt = _INTERVIEW_SYSTEM_TEMPLATE.format(
        company=company_name,
        resume=_state["resume_text"][:2500],
        context=context_text,
    )
    _state["interview_system_prompt"] = system_prompt

    # 첫 질문 생성
    resp = _state["llm"].invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content="면접을 시작해주세요. 첫 번째 기술 질문을 해주세요."),
    ])
    first_q = resp.content.strip()

    chat_history = [{"role": "assistant", "content": first_q}]
    status = f"✅ [{company_name}] 면접 시작 — 답변을 입력하세요."
    return chat_history, status, system_prompt

def send_answer(
    user_answer: str,
    chat_history: List,
    system_prompt: str,
    company_name: str,
) -> Tuple[str, List]:
    if not user_answer.strip():
        return "", chat_history
    if not _state["llm"]:
        return "", chat_history + [{"role": "assistant", "content": "❌ API 키를 설정하세요."}]

    from langchain_core.messages import HumanMessage, SystemMessage

    company_id = re.sub(r"[^a-z0-9_]", "_", company_name.lower()) if company_name else ""

    # 답변과 관련된 컨텍스트 재검색
    context_text = ""
    if company_id in _state["vectorstores"]:
        docs = _state["vectorstores"][company_id].similarity_search(user_answer, k=3)
        context_text = "\n\n---\n\n".join(d.page_content for d in docs)

    # LangChain 메시지 재구성
    messages = [SystemMessage(content=system_prompt)]
    for msg in chat_history:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(SystemMessage(content=f"[면접관 발언]: {content}"))

    feedback_prompt = (
        f"## 지원자 답변\n{user_answer.strip()}\n\n"
        f"## 관련 기업 기술 블로그 참고 (피드백 근거로 활용)\n"
        f"{context_text if context_text else '(참고 정보 없음)'}\n\n"
        "위 답변에 대해 **[피드백]** 과 **[다음 질문]** 두 섹션으로 응답하세요."
    )
    messages.append(HumanMessage(content=feedback_prompt))

    resp = _state["llm"].invoke(messages)
    ai_content = resp.content.strip()

    updated_history = chat_history + [
        {"role": "user", "content": user_answer.strip()},
        {"role": "assistant", "content": ai_content},
    ]
    return "", updated_history

# ─── Gradio UI 구성 ───────────────────────────────────────────
CSS = """
h1 { text-align: center; }
.tab-nav button { font-size: 15px; }
.score-box { font-size: 2rem; font-weight: bold; text-align: center; color: #2563eb; }
"""

with gr.Blocks(title="Tech-Prep Copilot", theme=gr.themes.Soft(), css=CSS) as demo:

    # ── 헤더 ──────────────────────────────────────────────────
    gr.Markdown("""
# 🎯 Tech-Prep Copilot
### AI 기반 맞춤형 기술 면접 & 기업 분석 에이전트

> LangChain RAG를 활용해 채용공고·기업 기술 블로그·이력서를 통합 분석하고
> 맞춤형 모의 면접을 진행합니다.
""")

    # ── 탭 1: 초기 설정 ───────────────────────────────────────
    with gr.Tab("⚙️ 초기 설정"):
        gr.Markdown("### 1️⃣ OpenAI API 키 설정")
        with gr.Row():
            api_key_input = gr.Textbox(
                label="OpenAI API Key",
                type="password",
                placeholder="sk-proj-...",
                scale=4,
            )
            api_btn = gr.Button("키 등록", variant="primary", scale=1)
        api_status = gr.Textbox(label="상태", interactive=False)

        gr.Markdown("---\n### 2️⃣ 이력서 & 채용공고(JD) 입력")
        with gr.Row():
            with gr.Column(scale=1):
                resume_file = gr.File(
                    label="📄 이력서 PDF 업로드",
                    file_types=[".pdf"],
                    file_count="single",
                )
                resume_status = gr.Textbox(label="이력서 상태", interactive=False)
            with gr.Column(scale=2):
                jd_input = gr.Textbox(
                    label="📋 채용공고(JD) 텍스트",
                    lines=12,
                    placeholder=(
                        "채용공고 전문을 여기에 붙여넣기 하세요.\n\n"
                        "예)\n[직무 요건]\n- Python/Java 백엔드 개발 3년 이상\n"
                        "- MSA 아키텍처 설계·운영 경험\n- AWS/GCP 클라우드 환경 경험 우대"
                    ),
                )
                jd_status = gr.Textbox(label="JD 상태", interactive=False)

        save_btn = gr.Button("💾 저장", variant="primary")

        api_btn.click(initialize_llm, inputs=[api_key_input], outputs=[api_status])
        save_btn.click(
            save_inputs,
            inputs=[resume_file, jd_input],
            outputs=[resume_status, jd_status],
        )

    # ── 탭 2: GAP 분석 리포트 ─────────────────────────────────
    with gr.Tab("📊 GAP 분석 리포트"):
        gr.Markdown("""\
### 이력서 vs 채용공고 GAP 분석
이력서와 JD를 AI가 직접 비교하여 **강점 / 보완점 / 추천 학습 키워드**를 제시합니다.
*분석 시간: 약 15~30초*
""")
        gap_btn = gr.Button("🔍 GAP 분석 시작", variant="primary", size="lg")

        with gr.Row():
            score_out = gr.Textbox(
                label="📈 직무 적합도 점수",
                interactive=False,
                elem_classes=["score-box"],
            )
            summary_out = gr.Textbox(
                label="💬 종합 평가",
                interactive=False,
                lines=3,
            )

        with gr.Row():
            strengths_out = gr.Textbox(
                label="✅ 강점 (Strengths)",
                interactive=False,
                lines=6,
            )
            weaknesses_out = gr.Textbox(
                label="⚠️ 보완점 (Weaknesses)",
                interactive=False,
                lines=6,
            )

        recs_out = gr.Textbox(
            label="📚 추천 학습 키워드 (Recommendations)",
            interactive=False,
            lines=3,
        )

        gap_btn.click(
            run_gap_analysis,
            outputs=[score_out, summary_out, strengths_out, weaknesses_out, recs_out],
        )

    # ── 탭 3: 기업 데이터 수집 ────────────────────────────────
    with gr.Tab("🏢 기업 데이터 수집"):
        gr.Markdown("""\
### 기업 기술 블로그 크롤링 & 벡터 DB 구축
기술 블로그 URL을 등록하면 RAG용 벡터 DB를 자동으로 구축합니다.
여러 URL을 반복 등록하면 같은 기업 컬렉션에 누적 인덱싱됩니다.
""")

        with gr.Row():
            company_name_in = gr.Textbox(
                label="기업명",
                placeholder="예: 카카오",
                scale=1,
            )
            blog_url_in = gr.Textbox(
                label="기술 블로그 URL",
                placeholder="https://tech.kakao.com/posts/600",
                scale=3,
            )
        crawl_btn = gr.Button("📥 크롤링 & 인덱싱", variant="primary")
        crawl_status = gr.Textbox(label="진행 상태", interactive=False, lines=5)

        gr.Markdown("""\
---
#### 📌 추천 기술 블로그 목록
| 기업 | URL 예시 |
|------|---------|
| 카카오 | https://tech.kakao.com/blog/ |
| 네이버 D2 | https://d2.naver.com/helloworld |
| 토스 | https://toss.tech/article |
| 우아한형제들 | https://techblog.woowahan.com/ |
| 라인 | https://engineering.linecorp.com/ko/blog |
| 당근마켓 | https://medium.com/daangn |

> **팁**: 특정 게시글 URL도 가능합니다. 팀원별로 다른 기업 블로그를 분담해 등록하세요.
""")

        crawl_btn.click(
            crawl_and_index,
            inputs=[company_name_in, blog_url_in],
            outputs=[crawl_status],
        )

    # ── 탭 4: 모의 면접 ───────────────────────────────────────
    with gr.Tab("🎤 모의 면접"):
        gr.Markdown("""\
### AI 맞춤형 모의 면접
기업 기술 블로그(RAG)와 이력서를 기반으로 실전 꼬리 질문을 경험합니다.
AI 면접관이 **[피드백]** 과 **[다음 질문]** 을 교대로 제공합니다.
""")

        # 면접 설정
        _system_prompt_state = gr.State("")

        with gr.Row():
            company_dd = gr.Dropdown(
                label="🏢 면접 기업 선택",
                choices=[],
                interactive=True,
                scale=3,
            )
            refresh_btn = gr.Button("🔄 목록 갱신", scale=1)
            start_btn = gr.Button("▶ 면접 시작", variant="primary", scale=1)

        interview_status = gr.Textbox(label="상태", interactive=False)

        # 채팅 인터페이스
        chatbot = gr.Chatbot(
            label="면접 채팅",
            height=520,
            type="messages",
            show_copy_button=True,
        )

        with gr.Row():
            answer_box = gr.Textbox(
                label="💬 답변 입력 (Shift+Enter로 줄바꿈, Enter로 전송)",
                placeholder="답변을 입력하세요...",
                lines=4,
                scale=5,
            )
            send_btn = gr.Button("전송", variant="primary", scale=1, min_width=80)

        clear_btn = gr.Button("🗑️ 면접 초기화", size="sm")

        # 이벤트 연결
        def refresh_dd():
            choices = get_company_choices()
            return gr.Dropdown(choices=choices, value=choices[0] if choices else None)

        refresh_btn.click(refresh_dd, outputs=[company_dd])

        start_btn.click(
            start_interview,
            inputs=[company_dd],
            outputs=[chatbot, interview_status, _system_prompt_state],
        )

        send_btn.click(
            send_answer,
            inputs=[answer_box, chatbot, _system_prompt_state, company_dd],
            outputs=[answer_box, chatbot],
        )
        answer_box.submit(
            send_answer,
            inputs=[answer_box, chatbot, _system_prompt_state, company_dd],
            outputs=[answer_box, chatbot],
        )
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, answer_box])

    # ── 푸터 ──────────────────────────────────────────────────
    gr.Markdown("""\
---
**⚠️ 주의사항**
- 이력서 원본은 서버에 저장되지 않으며 세션 메모리에만 임시 보관됩니다.
- LLM 응답에는 환각(Hallucination) 가능성이 있습니다. 중요한 기술 사실은 반드시 공식 문서로 재확인하세요.
- `gpt-4o-mini` 기준 GAP 분석 1회 약 0.01~0.03 USD, 면접 1턴 약 0.005~0.01 USD 소요됩니다.
""")

# ─────────────────────────────────────────────────────────────
# 서비스 실행
# Colab에서는 share=True 로 퍼블릭 URL 생성
# 로컬에서는 share=False 로 localhost 접근
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        share=True,          # Colab 환경에서 외부 접속 허용
        debug=False,
        show_error=True,
        server_name="0.0.0.0",
    )
