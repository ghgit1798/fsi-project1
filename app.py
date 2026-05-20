import os

import streamlit as st
from dotenv import load_dotenv
from langgraph.errors import GraphRecursionError

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
hf_token = os.getenv("HF_TOKEN")  # GPQA 등 gated 데이터셋 접근에 사용

# .env 로드 후에 에이전트 모듈을 import해야 ChatOpenAI가 키를 안정적으로 참조할 수 있음
from agents.specialists import ROUTE_TITLE_MAP
from benchmarks.gpqa_benchmark import evaluate_gpqa
from benchmarks.mmlu_benchmark import evaluate_mmlu
from benchmarks.truthfulqa_benchmark import evaluate_truthfulqa
from graph.workflow import run_security_workflow

st.set_page_config(page_title="보안 상담 챗봇", page_icon="🔐")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Noto+Sans+KR:wght@400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: "Noto Sans KR", sans-serif;
    }

    .stApp {
        background:
          radial-gradient(circle at 10% 10%, rgba(255, 179, 71, 0.14), transparent 40%),
          radial-gradient(circle at 90% 15%, rgba(74, 144, 226, 0.16), transparent 45%),
          linear-gradient(180deg, #f7f9fc 0%, #eef3f8 100%);
    }

    .hero-card {
        border-radius: 16px;
        padding: 1.05rem 1.15rem;
        background: linear-gradient(135deg, #10203a 0%, #1f4b7a 65%, #2b6d8c 100%);
        color: #f9fbff;
        box-shadow: 0 12px 30px rgba(16, 32, 58, 0.22);
        margin-bottom: 1rem;
        overflow: hidden;
    }

    .hero-title {
        font-family: "Space Grotesk", sans-serif;
        font-size: clamp(1.2rem, 2.3vw, 1.45rem);
        font-weight: 700;
        line-height: 1.25;
        margin-bottom: 0.3rem;
    }

    .hero-body {
        font-size: clamp(0.85rem, 1.45vw, 0.92rem);
        line-height: 1.5;
        opacity: 0.95;
        word-break: keep-all;
        overflow-wrap: anywhere;
    }

    .hint-wrap {
        display: flex;
        flex-wrap: wrap;
        gap: 0.28rem;
        margin-top: 0.22rem;
    }

    .hint-chip {
        display: inline-flex;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.16);
        padding: 0.24rem 0.55rem;
        font-size: clamp(0.73rem, 1.2vw, 0.8rem);
        max-width: 100%;
        line-height: 1.35;
        word-break: keep-all;
        overflow-wrap: anywhere;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-card">
      <div class="hero-title">🔐 보안 상담 챗봇</div>
      <div class="hero-body">
        일상에서 자주 겪는 보안 고민을 쉽게 설명하고, 지금 바로 실천할 수 있는 대응 방법을 안내합니다.
        질문 내용에 따라 <b>AI 보안</b>, <b>단말 보안</b>, <b>계정 보안</b>, <b>일반 보안</b> 상담으로 자동 분기됩니다.
      </div>
      <div class="hint-wrap">
        <span class="hint-chip">예시: "수상한 로그인 알림이 왔어요"</span>
        <span class="hint-chip">예시: "스마트폰 해킹이 의심돼요"</span>
        <span class="hint-chip">예시: "프롬프트 인젝션이 뭐예요?"</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not api_key:
    st.error("OPENAI_API_KEY를 찾을 수 없습니다. .env를 확인해 주세요.")
    st.stop()

st.caption("아래 입력창에 궁금한 보안 질문을 자유롭게 입력해 주세요.")
user_question = st.chat_input("예: 내 계정이 털린 것 같은데 먼저 뭘 해야 해?")

if user_question:
    st.write(f"질문: {user_question}")
    try:
        route, answer = run_security_workflow(user_question)
        st.write(f"라우터 분류 결과: `{route}`")
        st.write(f"### {ROUTE_TITLE_MAP.get(route, '보안 상담 답변')}")
        st.write(answer)
    except GraphRecursionError:
        st.error("워크플로우 실행 중 반복 오류가 발생했습니다. 다시 시도해 주세요.")

def render_benchmark_summary(state_key: str):
    summary = st.session_state.get(state_key)
    if not summary:
        return

    st.success("벤치마크 실행 완료")
    c1, c2, c3 = st.columns(3)
    c1.metric("샘플 수", summary["samples"])
    if "avg_score" in summary:
        c2.metric("평균 점수", f"{summary['avg_score']:.2f} / 5.0")
        c3.metric("Pass Rate(>=4)", f"{summary['pass_rate']:.4f}")
    else:
        c2.metric("정답 수", summary["correct"])
        c3.metric("정확도", f"{summary['accuracy']:.4f}")
    st.caption(f"결과 저장 경로: {summary['output_path']}")

    if "avg_score" in summary:
        preview_rows = [
            {
                "index": r["index"],
                "score": r["score"],
                "route": r["route"],
                "prediction": r["prediction"][:120],
            }
            for r in summary["results"]
        ]
    else:
        preview_rows = [
            {
                "index": r["index"],
                "gold": r["gold"],
                "pred": r["pred"],
                "correct": r["correct"],
                "route": r["route"],
            }
            for r in summary["results"]
        ]
    with st.expander("샘플별 결과 보기"):
        st.dataframe(preview_rows, use_container_width=True)


st.divider()
with st.sidebar.expander("참고", expanded=False):
    st.caption("필요할 때만 실행하세요. 랜덤 시드 42, 샘플 30개 고정입니다.")

    if st.button("MMLU 벤치마크 실행 (30문항)"):
        with st.spinner("MMLU 벤치마크 실행 중..."):
            try:
                summary = evaluate_mmlu(
                    split="test",
                    output_path="benchmarks/results/mmlu_results.json",
                    verbose=False,
                )
                st.session_state["mmlu_summary"] = summary
            except Exception as exc:  # noqa: BLE001
                st.error(f"벤치마크 실행 실패: {type(exc).__name__} - {exc}")

    if st.button("TruthfulQA 벤치마크 실행 (30문항)"):
        with st.spinner("TruthfulQA 벤치마크 실행 중..."):
            try:
                summary = evaluate_truthfulqa(
                    split="validation",
                    output_path="benchmarks/results/truthfulqa_results.json",
                    verbose=False,
                )
                st.session_state["truthfulqa_summary"] = summary
            except Exception as exc:  # noqa: BLE001
                st.error(f"벤치마크 실행 실패: {type(exc).__name__} - {exc}")

    if st.button("GPQA 벤치마크 실행 (30문항)"):
        with st.spinner("GPQA 벤치마크 실행 중..."):
            try:
                summary = evaluate_gpqa(
                    split="train",
                    output_path="benchmarks/results/gpqa_results.json",
                    verbose=False,
                )
                st.session_state["gpqa_summary"] = summary
            except Exception as exc:  # noqa: BLE001
                st.error(f"벤치마크 실행 실패: {type(exc).__name__} - {exc}")

    st.markdown("#### MMLU 결과")
    render_benchmark_summary("mmlu_summary")

    st.markdown("#### TruthfulQA 결과")
    render_benchmark_summary("truthfulqa_summary")

    st.markdown("#### GPQA 결과")
    render_benchmark_summary("gpqa_summary")
