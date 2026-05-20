from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class RouteDecision(BaseModel):
    route: str = Field(
        description=(
            "반드시 다음 중 하나: "
            "ai_security, device_security, account_security, general_security"
        )
    )


VALID_ROUTES = {
    "ai_security",
    "device_security",
    "account_security",
    "general_security",
}

router_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(
    RouteDecision
)
router_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "너는 보안 상담 라우터다. 사용자 질문을 정확히 하나의 route로 분류해라.\n"
                "- ai_security: AI 악용/프롬프트 인젝션/모델 보안/LLM 보안\n"
                "- device_security: PC/모바일/단말기 분실/악성앱/OS 보안\n"
                "- account_security: 계정 탈취/비밀번호/2FA/인증/로그인 문제\n"
                "- general_security: 나머지 일반 보안 이슈\n"
                "반드시 route 필드만 반환하라."
            ),
        ),
        ("human", "{question}"),
    ]
)


def router_agent(question: str) -> str:
    decision = (router_prompt | router_llm).invoke({"question": question})
    return decision.route if decision.route in VALID_ROUTES else "general_security"
