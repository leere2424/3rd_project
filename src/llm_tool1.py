# 기본 라이브러리 및 LangChain 구성요소 import
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain.tools import tool
import json

from .config import SETTINGS

OPENAI_API_KEY = SETTINGS.openai_api_key

# `embedding_search` tool이 참조하는 1단계 지시문:
# - 사용자 자연어를 검색 가능한 requirement 목록으로 구조화한다.
# 1단계 프롬프트: 사용자 문장을 field+requirement JSON으로 구조화
prompt1 = """
당신은 식당 검색 시스템의 1단계 분석기입니다.

역할:
사용자의 자연어 요청을 읽고, 식당 검색에 사용할 수 있는 "요구사항 단위"로 분해한 뒤,
각 요구사항을 아래 5개의 검색 field 중 하나로 분류하세요.

검색 field 정의:
- category: 식당을 한 단어 또는 짧은 표현으로 설명하는 카테고리
  예: 일식, 고깃집, 브런치 카페, 술집
- tag: 식당의 분위기, 상황, 이용 목적, 방문 형태를 설명하는 특징
  예: 조용한, 혼밥하기 좋은, 접대용, 데이트하기 좋은
- review: 이용자 리뷰에서 자주 언급될 만한 평가 요소
  예: 서비스가 친절한, 양이 많은, 바삭한, 가성비가 좋은
- menu: 메뉴명 또는 메뉴 설명 수준의 구체적인 음식/조리/맛 특성
  예: 치즈가 많이 들어간 메뉴, 기름진 음식, 매운 국물 요리
- food: 음식의 대분류
  예: 돈가스, 국밥, 파스타, 면 요리, 튀김류

목표:
1. 사용자 입력에서 식당 검색에 의미 있는 요구사항만 추출하세요.
2. 하나의 요구사항은 가능한 한 하나의 검색 의도만 담도록 짧게 분리하세요.
3. 각 요구사항을 가장 적절한 field 하나에만 할당하세요.
4. 아직 검색 query를 확장하거나 유의어를 만들지 마세요.
5. 사용자가 직접 말하지 않은 특정 음식명이나 메뉴명을 추론해서 requirement에 넣지 마세요.
6. 사용자의 표현을 가능한 한 그대로 유지하되, 검색 요구사항 형태로만 최소한 정리하세요.

중요 규칙:
- 아직 검색 query를 생성하는 단계가 아닙니다.
- requirement는 원문의 핵심 의미를 유지한 짧은 문장 또는 구여야 합니다.
- 직접 언급되지 않은 음식명으로 바꾸지 마세요.
  예를 들어 사용자가 "기름진 음식"이라고 말했으면 requirement를 "돈가스"로 바꾸지 마세요.
- 여러 의미가 섞인 긴 문장은 분리하세요.
- 식당 검색과 무관한 표현은 제외하세요.
- 같은 의미의 중복 requirement는 만들지 마세요.
- 하나의 requirement는 반드시 하나의 field만 가져야 합니다.

출력 형식:
반드시 JSON 배열만 출력하세요.
각 원소는 아래 형식을 따르세요.

[
  {
    "field": "category | tag | review | menu | food",
    "requirement": "검색 요구사항"
  }
]

출력 예시 1:
사용자 입력:
조용하고 혼밥하기 좋으면서 기름진 음식 먹고 싶어

출력:
[
  {"field": "tag", "requirement": "조용한 곳"},
  {"field": "tag", "requirement": "혼밥하기 좋은 곳"},
  {"field": "menu", "requirement": "기름진 음식"}
]

출력 예시 2:
사용자 입력:
바삭한 튀김이 맛있고 양 많은 돈가스집 찾아줘

출력:
[
  {"field": "review", "requirement": "바삭한 튀김이 맛있는 곳"},
  {"field": "review", "requirement": "양이 많은 곳"},
  {"field": "food", "requirement": "돈가스"},
  {"field": "category", "requirement": "돈가스집"}
]

출력 예시 3:
사용자 입력:
접대하기 좋고 조용한 일식집

출력:
[
  {"field": "tag", "requirement": "접대하기 좋은 곳"},
  {"field": "tag", "requirement": "조용한 곳"},
  {"field": "category", "requirement": "일식집"}
]

이제 사용자 입력을 분석하세요.
사용자 입력:
{user_input}
"""

# `embedding_search` tool이 참조하는 2단계 지시문:
# - requirement 1건을 실제 임베딩 검색 query 후보(1~3개)로 확장한다.
# 2단계 프롬프트: requirement를 임베딩 검색용 query 리스트로 확장
prompt2 = """
당신은 식당 검색 시스템의 쿼리 분석기입니다.

역할:
입력으로 주어진 requirement를 읽고,
해당 requirement를 식당 검색 DB의 임베딩 검색에 사용할 수 있는 짧은 query 문자열 리스트로 변환하세요.

입력 형식:
{
  "field": "category | tag | review | menu | food",
  "requirement": "검색 요구사항"
}

field 정의:
- category: 식당의 유형 (예: 일식, 고깃집, 브런치 카페)
- tag: 분위기, 이용 상황, 방문 목적 (예: 조용한, 혼밥하기 좋은, 데이트하기 좋은)
- review: 리뷰에서 나타나는 평가 요소 (예: 양이 많은, 서비스가 친절한, 바삭한)
- menu: 메뉴의 특징, 조리 방식, 맛, 재료 (예: 기름진 음식, 매운 국물 요리, 치즈가 많은 메뉴)
- food: 음식의 대분류 (예: 돈가스, 국밥, 파스타, 면 요리, 튀김류)

목표:
1. requirement의 핵심 의미를 유지한 짧은 query를 생성하세요.
2. query는 검색에 적합한 짧은 구 형태로 작성하세요.
3. field에 맞는 표현 스타일로 query를 생성하세요.
4. query 개수는 1~3개로 제한하세요.
5. 같은 의미 범위 내에서만 표현을 다양화하세요.

중요 규칙:
1. 사용자가 특정 음식명 또는 메뉴명을 직접 언급한 경우에만 해당 표현을 query에 사용할 수 있습니다.
   - 예: requirement가 "돈가스" → ["돈가스"]

2. 사용자가 특정 음식명/메뉴명을 직접 언급하지 않은 경우,
   임의로 특정 음식명/메뉴명으로 구체화하지 마세요.
   - 예: "기름진 음식" → 가능
   - 예: "튀겨서 만든 요리" → 가능
   - 예: "돈가스" → 불가능

3. requirement를 더 짧고 검색 친화적인 표현으로 바꾸는 것은 허용되지만,
   원래 의미를 벗어나면 안 됩니다.

4. 과도한 유의어 확장은 금지합니다.
   - 의미 범위를 벗어나는 query는 만들지 마세요.

5. 문장 대신 짧은 구를 사용하세요.
   - 좋은 예: "조용한", "혼밥하기 좋은", "기름진 음식"
   - 나쁜 예: "혼자 가서 편하게 먹을 수 있는 식당"

6. field에 맞는 표현을 사용하세요:
   - category: 식당 유형 중심 (예: 일식집, 고깃집)
   - tag: 분위기/이용 상황 중심 (예: 조용한, 혼밥하기 좋은)
   - review: 평가/후기 표현 (예: 양이 많은, 바삭한)
   - menu: 음식 특성/조리 방식 (예: 기름진 음식, 매운 국물 요리)
   - food: 음식 분류 (예: 돈가스, 면 요리)

출력 형식:
반드시 JSON 배열(list[str])만 출력하세요.

예:
["조용한", "대화 소음이 적은"]

예시 1
입력:
{
  "field": "tag",
  "requirement": "조용한 곳"
}

출력:
["조용한", "대화 소음이 적은"]

예시 2
입력:
{
  "field": "tag",
  "requirement": "혼밥하기 좋은 곳"
}

출력:
["혼밥하기 좋은", "1인 방문하기 편한"]

예시 3
입력:
{
  "field": "menu",
  "requirement": "기름진 음식"
}

출력:
["기름진 음식", "튀겨서 만든 요리"]

예시 4
입력:
{
  "field": "food",
  "requirement": "돈가스"
}

출력:
["돈가스"]

예시 5
입력:
{
  "field": "review",
  "requirement": "양이 많은 곳"
}

출력:
["양이 많은", "푸짐한"]

이제 아래 입력을 변환하세요.

입력:
{requirement_json}
"""

@tool("embedding_search")
def embedding_search(user_input: str):
    # Tool 역할:
    # 사용자 문장을 2-stage LLM 처리하여
    # "requirement별 임베딩 검색 query 리스트"를 생성한다.
    #
    # 입력:
    # - user_input: 사용자의 원본 검색 요청 문장
    #
    # 출력:
    # - list[list[str]]
    #   예) [["조용한", "대화 소음이 적은"], ["기름진 음식", "튀겨서 만든 요리"]]
    def _parse_json_payload(raw_text: str, expected_type: type):
        """
        LLM이 코드블록(```json ... ```)으로 감싸 반환하는 경우를 포함해
        JSON payload를 안전하게 파싱한다.
        """
        text = (raw_text or "").strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        parsed = json.loads(text)
        if not isinstance(parsed, expected_type):
            raise ValueError(
                f"LLM JSON 타입이 올바르지 않습니다. expected={expected_type.__name__}"
            )
        return parsed

    # 전체 처리 흐름:
    # 사용자 문장 -> (1단계) requirement 분해 -> (2단계) 임베딩 검색 query 리스트 생성

    # 낮은 temperature로 샘플링 변동성을 줄입니다.
    llm = ChatOpenAI(model=SETTINGS.llm_model, temperature=0)
    output_parser = StrOutputParser()

    # 1단계 체인 정의
    chain1 = (
        RunnableLambda(lambda x: prompt1.replace("{user_input}", x["user_input"]))
        | llm
        | output_parser
    )

    # 2단계 체인 정의
    chain2 = (
        RunnableLambda(
            lambda x: prompt2.replace("{requirement_json}", x["requirement_json"])
        )
        | llm
        | output_parser
    )

    # 1단계 실행: 사용자 입력을 field/requirement JSON 문자열로 변환
    response1_raw = chain1.invoke({"user_input": user_input})

    # 1단계 결과를 JSON 배열로 파싱한 뒤, item별로 2단계를 고정 순서로 실행합니다.
    requirements = _parse_json_payload(response1_raw, list)
    response2 = []
    for item in requirements:
        if not isinstance(item, dict):
            raise ValueError("1단계 결과의 각 항목은 JSON object여야 합니다.")
        # item 하나를 2단계 입력 포맷으로 직렬화
        requirement_json = json.dumps(item, ensure_ascii=False)

        # 2단계 실행: requirement 하나 -> query 후보 리스트
        response2_raw = chain2.invoke({"requirement_json": requirement_json})
        response2.append(_parse_json_payload(response2_raw, list))

    # 최종 결과: requirement별 query 후보 리스트의 리스트
    return response2