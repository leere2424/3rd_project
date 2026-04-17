import json
from typing import Dict

from openai import OpenAI
from langchain.tools import tool

from .config import SETTINGS

# llm_tool 내부 기본 모델(기존 동작 유지)
DEFAULT_FIXED_SEARCH_MODEL = SETTINGS.fixed_search_model

# 모듈 단위에서 OpenAI 클라이언트를 1회만 재사용한다.
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        # .env 로드 및 환경값 조회는 config 모듈에서 처리한다.
        if not SETTINGS.openai_api_key:
            raise ValueError("OPENAI_API_KEY 환경변수가 설정되어 있지 않습니다.")
        _client = OpenAI(api_key=SETTINGS.openai_api_key)
    return _client

prompt = """너는 사용자의 검색 문장을 분석하는 분류기다.
반드시 JSON 객체만 반환해.
출력 형식은 정확히 {"restaurant": "...", "menu": "...", "user": "..."} 이어야 한다.

판단 규칙:
1) 식당, 음식점, 카페, 맛집, 브랜드, 지역 등 식당/매장 중심 검색이면 restaurant에 넣고 나머지는 빈 문자열로 둔다.
2) 특정 음식, 메뉴명, 요리 이름, 음료 등 메뉴 중심 검색이면 menu에 넣고 나머지는 빈 문자열로 둔다.
3) 리뷰어, 작성자, 유저, 닉네임, 블로거, 인플루언서 등 사람/리뷰 작성자 중심 검색이면 user에 넣고 나머지는 빈 문자열로 둔다.
4) 식당, 메뉴, 유저가 동시에 등장하면 각각 해당 필드에 채운다.
5) 확실하지 않으면 추측하지 말고 해당 필드는 빈 문자열로 둔다.
6) 설명, 마크다운, 코드블록 없이 JSON만 출력한다."""

def _make_fixed_search_json(instr: str, model: str = DEFAULT_FIXED_SEARCH_MODEL) -> str:
    """
    1단계:
    사용자 입력을 분석해서
    {"restaurant": "...", "menu": "...", "user": "..."}
    형태의 JSON string만 반환받는다.
    """
    completion = _get_client().chat.completions.create(
        model=model,
        messages=[
            {
                "role": "developer",
                "content": prompt,
            },
            {
                "role": "user",
                "content": instr,
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "fixed_search_result",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "restaurant": {"type": "string"},
                        "menu": {"type": "string"},
                        "user": {"type": "string"},
                    },
                    "required": ["restaurant", "menu", "user"],
                    "additionalProperties": False,
                },
            },
        },
        temperature=0,
        max_completion_tokens=120,
        top_p=1,
    )

    content = completion.choices[0].message.content
    if not content:
        raise ValueError("모델이 빈 응답을 반환했습니다.")

    return content


def _parse_fixed_search_json(raw_json: str) -> Dict[str, str]:
    """
    2단계:
    1단계에서 받은 JSON string을 파싱해서
    최종 dict로 반환한다.
    """
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 파싱 실패: {e}") from e

    if not isinstance(data, dict):
        raise ValueError("모델 응답이 JSON object가 아닙니다.")

    restaurant = data.get("restaurant", "")
    menu = data.get("menu", "")
    user = data.get("user", "")

    if restaurant is None:
        restaurant = ""
    if menu is None:
        menu = ""
    if user is None:
        user = ""

    if not isinstance(restaurant, str) or not isinstance(menu, str) or not isinstance(user, str):
        raise ValueError("restaurant, menu, user 값은 문자열이어야 합니다.")

    return {
        "restaurant": restaurant.strip(),
        "menu": menu.strip(),
        "user": user.strip(),
    }


@tool("fixed_search")
def fixed_search(instr: str) -> Dict[str, str]:
    """
    Tool 역할:
    사용자 검색 문장을 restaurant/menu/user 3개 슬롯으로 분류해 반환한다.

    처리 단계:
    1) `_make_fixed_search_json`에서 LLM으로 구조화 JSON 생성
    2) `_parse_fixed_search_json`에서 타입/형식 검증 및 정규화
    3) 최종 dict 반환

    Args:
        instr: 사용자의 원본 검색 문장

    Returns:
        {"restaurant": "...", "menu": "...", "user": "..."}
    """
    instr = (instr or "").strip()
    if not instr:
        # 빈 입력은 LLM 호출 없이 즉시 빈 슬롯 구조를 반환한다.
        return {"restaurant": "", "menu": "", "user": ""}

    raw_json = _make_fixed_search_json(instr)
    parsed = _parse_fixed_search_json(raw_json)
    return parsed