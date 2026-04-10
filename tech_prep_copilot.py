# ============================================================
# Tech-Prep Copilot — Google Colab + Gradio 구현
# AI 기반 맞춤형 기술 면접 & 기업 분석 에이전트
#
# [Colab 실행 순서]
# 1. 셀 1(패키지 설치) 실행
# 2. [런타임 → 세션 다시 시작]
# 3. 셀 2(앱 실행) 실행 → 출력된 URL 접속
# ============================================================

# ─────────────────────────────────────────────────────────────
# 셀 1: 패키지 설치
# ※ 설치 후 "dependency conflicts" 경고는 무시하세요.
# ※ 설치 완료 후 [런타임 → 세션 다시 시작] 필수!
# ─────────────────────────────────────────────────────────────
# !pip install -q \
#   "gradio>=5.0" \
#   "langchain-groq>=0.2" \
#   "langchain-community>=0.3" \
#   "langchain-chroma>=0.1.4" \
#   "langchain-text-splitters>=0.3" \
#   "langchain-huggingface>=0.1" \
#   "sentence-transformers>=3.0" \
#   pypdf \
#   beautifulsoup4

# ─────────────────────────────────────────────────────────────
# 셀 2: 앱 실행 (아래 코드 전체를 한 셀에서 실행)
# ─────────────────────────────────────────────────────────────

import gradio as gr
import json, os, re, textwrap
from typing import List, Tuple, Dict

import requests
from bs4 import BeautifulSoup
import pypdf

# ── 전역 상태 ──────────────────────────────────────────────────
_state: Dict = {
    "resume_text": "",
    "jd_text": "",
    "gap_report": None,
    "interview_system_prompt": "",
    "vectorstores": {},
    "available_companies": [],
    "llm": None,
    "embeddings": None,
}

# ── LLM 초기화 (Groq + HuggingFace 임베딩) ────────────────────
def initialize_llm(api_key: str) -> str:
    if not api_key.strip():
        return "❌ Groq API 키를 입력하세요."
    try:
        from langchain_groq import ChatGroq
        from langchain_huggingface import HuggingFaceEmbeddings

        os.environ["GROQ_API_KEY"] = api_key.strip()
        _state["llm"] = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=2048,
        )
        # 로컬 임베딩 (무료, 첫 실행 시 모델 다운로드 약 90MB)
        _state["embeddings"] = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
        return "✅ Groq API 키 설정 완료. 임베딩 모델 로드 완료."
    except Exception as e:
        return f"❌ 초기화 실패: {e}"

# ── PDF 파싱 ───────────────────────────────────────────────────
def _parse_pdf(path: str) -> str:
    with open(path, "rb") as f:
        reader = pypdf.PdfReader(f)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

# ── 이력서 & JD 저장 ───────────────────────────────────────────
def save_inputs(pdf_file, jd_text: str) -> Tuple[str, str]:
    resume_msg = "⚠️ PDF 파일을 업로드하세요."
    jd_msg     = "⚠️ 채용공고 텍스트를 입력하세요."

    if pdf_file is not None:
        text = _parse_pdf(pdf_file)
        _state["resume_text"] = text
        resume_msg = f"✅ 이력서 파싱 완료 — {len(text):,}자 추출"

    if jd_text.strip():
        _state["jd_text"] = jd_text.strip()
        jd_msg = f"✅ JD 저장 완료 — {len(jd_text):,}자"

    return resume_msg, jd_msg

# ── GAP 분석 ───────────────────────────────────────────────────
_GAP_SYSTEM = textwrap.dedent("""\
    당신은 10년 경력의 IT 채용 컨설턴트입니다.
    이력서와 채용공고(JD)를 비교 분석하여 직무 적합성을 평가하세요.

    반드시 아래 JSON 형식으로만 응답하세요 (마크다운 코드블록 없이 순수 JSON):
    {
      "overall_score": <0~100 정수>,
      "summary": "<한두 문장 종합 평가>",
      "strengths": ["<강점1>", "<강점2>", "<강점3>"],
      "weaknesses": ["<보완점1>", "<보완점2>", "<보완점3>"],
      "recommendations": ["<추천 키워드1>", "<추천 키워드2>", "<추천 키워드3>"]
    }

    규칙: 제공된 문서에 없는 수치나 기술 사실을 임의로 생성하지 마세요.
""")

def run_gap_analysis() -> Tuple[str, str, str, str, str]:
    if not _state["llm"]:
        return ("❌ API 키를 먼저 설정하세요.", "", "", "", "")
    if not _state["resume_text"]:
        return ("❌ 이력서를 먼저 업로드하세요.", "", "", "", "")
    if not _state["jd_text"]:
        return ("❌ JD를 먼저 입력하세요.", "", "", "", "")

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
        m = re.search(r"\{[\s\S]+\}", resp.content.strip())
        if not m:
            return (f"❌ JSON 파싱 실패:\n{resp.content}", "", "", "", "")

        data = json.loads(m.group())
        _state["gap_report"] = data

        score      = f"{data.get('overall_score', '?')} / 100"
        summary    = data.get("summary", "")
        strengths  = "\n".join(f"• {s}" for s in data.get("strengths", []))
        weaknesses = "\n".join(f"• {w}" for w in data.get("weaknesses", []))
        recs       = "\n".join(f"• {r}" for r in data.get("recommendations", []))
        return score, summary, strengths, weaknesses, recs
    except Exception as e:
        return (f"❌ 분석 중 오류: {e}", "", "", "", "")

# ── 기업 블로그 크롤링 & 벡터 인덱싱 ──────────────────────────
def crawl_and_index(company_name: str, blog_url: str) -> str:
    if not _state["embeddings"]:
        return "❌ API 키를 먼저 설정하세요."
    if not company_name.strip() or not blog_url.strip():
        return "❌ 기업명과 블로그 URL을 모두 입력하세요."
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(blog_url.strip(), headers=headers, timeout=15)
        res.raise_for_status()

        soup = BeautifulSoup(res.content, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        raw_text = soup.get_text(separator="\n", strip=True)

        if len(raw_text) < 100:
            return "⚠️ 텍스트 추출 실패. URL을 확인하세요."

        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_chroma import Chroma

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        docs = splitter.create_documents(
            [raw_text],
            metadatas=[{"source": blog_url.strip(), "company": company_name.strip()}],
        )

        cid = re.sub(r"[^a-z0-9_]", "_", company_name.strip().lower())
        if cid in _state["vectorstores"]:
            _state["vectorstores"][cid].add_documents(docs)
        else:
            _state["vectorstores"][cid] = Chroma.from_documents(docs, _state["embeddings"])

        if company_name.strip() not in _state["available_companies"]:
            _state["available_companies"].append(company_name.strip())

        return (
            f"✅ [{company_name}] 크롤링 완료\n"
            f"   • 수집 텍스트: {len(raw_text):,}자\n"
            f"   • 인덱싱 청크: {len(docs)}개"
        )
    except requests.exceptions.RequestException as e:
        return f"❌ 네트워크 오류: {e}"
    except Exception as e:
        return f"❌ 크롤링 실패: {e}"

# ── 모의 면접 ──────────────────────────────────────────────────
_INTERVIEW_TPL = textwrap.dedent("""\
    당신은 {company}의 시니어 엔지니어 면접관입니다.
    아래 이력서와 기업 기술 블로그 정보를 바탕으로 심층 기술 면접을 진행합니다.

    ## 면접 원칙
    1. 이력서의 실제 프로젝트·기술 경험에 대해 구체적으로 질문하세요.
    2. 기업 블로그 내용과 연결하여 꼬리 질문을 던지세요.
    3. 한 번에 질문 하나만 하세요.
    4. 답변 후에는 **[피드백]** 섹션(기술 정확성 + 블로그 근거 인용)과
       **[다음 질문]** 섹션으로 응답하세요.
    5. 제공된 문서에 없는 사실을 임의로 만들지 마세요.

    ## 이력서
    {resume}

    ## 기업 기술 블로그 참고 정보
    {context}
""")

def start_interview(company_name: str) -> Tuple[List, str, str]:
    if not _state["llm"]:
        return [], "❌ API 키를 먼저 설정하세요.", ""
    if not _state["resume_text"]:
        return [], "❌ 이력서를 먼저 업로드하세요.", ""
    if not company_name:
        return [], "❌ 기업을 선택하세요.", ""

    from langchain_core.messages import HumanMessage, SystemMessage

    cid = re.sub(r"[^a-z0-9_]", "_", company_name.lower())
    context_text = "(수집된 블로그 정보 없음 — 이력서 기반 일반 면접 진행)"
    if cid in _state["vectorstores"]:
        docs = _state["vectorstores"][cid].similarity_search(_state["resume_text"][:500], k=3)
        if docs:
            context_text = "\n\n---\n\n".join(d.page_content for d in docs)

    system_prompt = _INTERVIEW_TPL.format(
        company=company_name,
        resume=_state["resume_text"][:2500],
        context=context_text,
    )
    _state["interview_system_prompt"] = system_prompt

    resp = _state["llm"].invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content="면접을 시작해주세요. 첫 번째 기술 질문을 해주세요."),
    ])
    chat = [{"role": "assistant", "content": resp.content.strip()}]
    return chat, f"✅ [{company_name}] 면접 시작 — 답변을 입력하세요.", system_prompt

def send_answer(user_answer: str, chat_history: List, system_prompt: str, company_name: str) -> Tuple[str, List]:
    if not user_answer.strip():
        return "", chat_history
    if not _state["llm"]:
        return "", chat_history + [{"role": "assistant", "content": "❌ API 키를 설정하세요."}]

    from langchain_core.messages import HumanMessage, SystemMessage

    cid = re.sub(r"[^a-z0-9_]", "_", company_name.lower()) if company_name else ""
    context_text = ""
    if cid in _state["vectorstores"]:
        docs = _state["vectorstores"][cid].similarity_search(user_answer, k=3)
        context_text = "\n\n---\n\n".join(d.page_content for d in docs)

    messages = [SystemMessage(content=system_prompt)]
    for msg in chat_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(SystemMessage(content=f"[면접관]: {msg['content']}"))

    messages.append(HumanMessage(content=(
        f"## 지원자 답변\n{user_answer.strip()}\n\n"
        f"## 관련 블로그 참고\n{context_text or '(없음)'}\n\n"
        "**[피드백]** 과 **[다음 질문]** 두 섹션으로 응답하세요."
    )))

    resp = _state["llm"].invoke(messages)
    updated = chat_history + [
        {"role": "user",      "content": user_answer.strip()},
        {"role": "assistant", "content": resp.content.strip()},
    ]
    return "", updated

# ── Gradio UI ──────────────────────────────────────────────────
with gr.Blocks(title="Tech-Prep Copilot", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
# 🎯 Tech-Prep Copilot
### AI 기반 맞춤형 기술 면접 & 기업 분석 에이전트
LangChain RAG + **Groq (무료)** 로 채용공고·기업 기술 블로그·이력서를 분석하고 맞춤형 모의 면접을 진행합니다.
""")

    # 탭 1 ── 초기 설정
    with gr.Tab("⚙️ 초기 설정"):
        gr.Markdown("""\
### 1️⃣ Groq API 키
무료 발급: https://console.groq.com → API Keys → Create API Key
""")
        with gr.Row():
            api_key_input = gr.Textbox(label="Groq API Key", type="password",
                                       placeholder="gsk_...", scale=4)
            api_btn = gr.Button("등록", variant="primary", scale=1)
        api_status = gr.Textbox(label="상태", interactive=False)

        gr.Markdown("---\n### 2️⃣ 이력서 & 채용공고 입력")
        with gr.Row():
            with gr.Column(scale=1):
                resume_file   = gr.File(label="📄 이력서 PDF", file_types=[".pdf"])
                resume_status = gr.Textbox(label="이력서 상태", interactive=False)
            with gr.Column(scale=2):
                jd_input  = gr.Textbox(label="📋 채용공고(JD) 텍스트", lines=12,
                                       placeholder="채용공고 전문을 붙여넣기 하세요...")
                jd_status = gr.Textbox(label="JD 상태", interactive=False)
        save_btn = gr.Button("💾 저장", variant="primary")

        api_btn.click(initialize_llm, inputs=[api_key_input], outputs=[api_status])
        save_btn.click(save_inputs, inputs=[resume_file, jd_input],
                       outputs=[resume_status, jd_status])

    # 탭 2 ── GAP 분석
    with gr.Tab("📊 GAP 분석 리포트"):
        gr.Markdown("### 이력서 vs 채용공고 GAP 분석\n분석 시간 약 10~20초")
        gap_btn = gr.Button("🔍 GAP 분석 시작", variant="primary", size="lg")
        with gr.Row():
            score_out   = gr.Textbox(label="📈 적합도 점수", interactive=False)
            summary_out = gr.Textbox(label="💬 종합 평가",   interactive=False, lines=3)
        with gr.Row():
            strengths_out  = gr.Textbox(label="✅ 강점",   interactive=False, lines=6)
            weaknesses_out = gr.Textbox(label="⚠️ 보완점", interactive=False, lines=6)
        recs_out = gr.Textbox(label="📚 추천 학습 키워드", interactive=False, lines=3)

        gap_btn.click(run_gap_analysis,
                      outputs=[score_out, summary_out, strengths_out, weaknesses_out, recs_out])

    # 탭 3 ── 기업 데이터 수집
    with gr.Tab("🏢 기업 데이터 수집"):
        gr.Markdown("### 기술 블로그 크롤링 & 벡터 DB 구축\n같은 기업 URL을 여러 번 등록하면 누적 인덱싱됩니다.")
        with gr.Row():
            company_name_in = gr.Textbox(label="기업명", placeholder="예: 카카오", scale=1)
            blog_url_in     = gr.Textbox(label="기술 블로그 URL",
                                         placeholder="https://tech.kakao.com/posts/600", scale=3)
        crawl_btn    = gr.Button("📥 크롤링 & 인덱싱", variant="primary")
        crawl_status = gr.Textbox(label="진행 상태", interactive=False, lines=4)

        gr.Markdown("""---
| 기업 | URL 예시 |
|------|---------|
| 카카오 | https://tech.kakao.com/blog/ |
| 네이버 D2 | https://d2.naver.com/helloworld |
| 토스 | https://toss.tech/article |
| 우아한형제들 | https://techblog.woowahan.com/ |
| 라인 | https://engineering.linecorp.com/ko/blog |
| 당근마켓 | https://medium.com/daangn |
""")
        crawl_btn.click(crawl_and_index,
                        inputs=[company_name_in, blog_url_in], outputs=[crawl_status])

    # 탭 4 ── 모의 면접
    with gr.Tab("🎤 모의 면접"):
        gr.Markdown("### AI 맞춤형 모의 면접\nRAG 기반 꼬리 질문 → 답변 → 블로그 근거 인용 피드백")

        _sys_state = gr.State("")

        with gr.Row():
            company_dd  = gr.Dropdown(label="🏢 면접 기업 선택", choices=[], interactive=True, scale=3)
            refresh_btn = gr.Button("🔄 목록 갱신", scale=1)
            start_btn   = gr.Button("▶ 면접 시작", variant="primary", scale=1)
        interview_status = gr.Textbox(label="상태", interactive=False)

        chatbot = gr.Chatbot(label="면접 채팅", height=500, type="messages", show_copy_button=True)

        with gr.Row():
            answer_box = gr.Textbox(label="💬 답변 입력 (Enter 전송)", lines=4,
                                    placeholder="답변을 입력하세요...", scale=5)
            send_btn = gr.Button("전송", variant="primary", scale=1, min_width=80)
        clear_btn = gr.Button("🗑️ 면접 초기화", size="sm")

        def refresh_dd():
            c = _state["available_companies"]
            return gr.Dropdown(choices=c, value=c[0] if c else None)

        refresh_btn.click(refresh_dd, outputs=[company_dd])
        start_btn.click(start_interview, inputs=[company_dd],
                        outputs=[chatbot, interview_status, _sys_state])
        send_btn.click(send_answer,
                       inputs=[answer_box, chatbot, _sys_state, company_dd],
                       outputs=[answer_box, chatbot])
        answer_box.submit(send_answer,
                          inputs=[answer_box, chatbot, _sys_state, company_dd],
                          outputs=[answer_box, chatbot])
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, answer_box])

    gr.Markdown("""\
---
⚠️ 이력서 원본은 서버에 저장되지 않으며 세션 메모리에만 임시 보관됩니다.
LLM 응답에는 환각 가능성이 있으므로 중요한 기술 사실은 공식 문서로 재확인하세요.
""")

demo.launch(share=True, show_error=True)
