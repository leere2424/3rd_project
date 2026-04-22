"""질의응답 파이프라인(LangGraph) + CLI 진입점.

- `run_qa(question, session_id)` : 외부 호출용 API
- `python -m src.pipeline`       : 터미널에서 대화 테스트
"""
from __future__ import annotations

from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from .generator import generate_response
from database.sql.utils import db_fixed_search, db_embedding_search
from .router import decide_route
from .slot_extractor import embedding_slot_extract, fixed_search


class GraphState(TypedDict, total=False):
    question: str
    session_id: str

    route: Literal["embedding", "fixed"]
    route_payload: dict[str, str]

    # utils(db connector) 반환값: list[dict]
    restaurant_list: list[dict[str, Any]]

    answer: str


_graph = None


def _normalize_restaurant_list(result: Any) -> list[dict[str, Any]]:
    """
    utils 함수 반환값을 restaurant_list 형태로 정규화한다.

    허용:
    1) list[dict]
    2) {"restaurant_list": list[dict]}  # 혹시 utils 쪽이 나중에 이렇게 바뀌어도 방어
    """
    if result is None:
        return []

    if isinstance(result, list):
        return result

    if isinstance(result, dict):
        restaurant_list = result.get("restaurant_list", [])
        if restaurant_list is None:
            return []
        if not isinstance(restaurant_list, list):
            raise ValueError("connector_response['restaurant_list']는 list 형태여야 합니다.")
        return restaurant_list

    raise ValueError("connector/utils 반환값은 list 또는 {'restaurant_list': list} 형태여야 합니다.")


def route_node(state: GraphState) -> GraphState:
    return {"route": decide_route(state["question"])}


def embedding_slot_node(state: GraphState) -> GraphState:
    payload = embedding_slot_extract.invoke(state["question"])
    return {"route_payload": payload}


def fixed_slot_node(state: GraphState) -> GraphState:
    payload = fixed_search.invoke(state["question"])
    return {"route_payload": payload}


def connector_search_node(state: GraphState) -> GraphState:
    route = state["route"]
    route_payload = state.get("route_payload", {})

    if route == "embedding":
        raw_result = db_embedding_search(route_payload)
    elif route == "fixed":
        raw_result = db_fixed_search(route_payload)
    else:
        raise ValueError(f"알 수 없는 route입니다: {route}")

    restaurant_list = _normalize_restaurant_list(raw_result)
    return {"restaurant_list": restaurant_list}


def generate_node(state: GraphState) -> GraphState:
    question = state["question"]
    session_id = state.get("session_id", "default")
    route = state.get("route", "embedding")
    route_payload = state.get("route_payload", {})
    restaurant_list = state.get("restaurant_list", [])

    connector_meta = {
        "restaurant_count": len(restaurant_list)
    }

    answer = generate_response(
        question=question,
        restaurant_list=restaurant_list,
        route=route,
        session_id=session_id,
        route_payload=route_payload,
        connector_meta=connector_meta,
    )
    return {"answer": answer}


def route_condition(state: GraphState) -> str:
    return state["route"]


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("route_node", route_node)
    graph.add_node("embedding_slot_node", embedding_slot_node)
    graph.add_node("fixed_slot_node", fixed_slot_node)
    graph.add_node("connector_search_node", connector_search_node)
    graph.add_node("generate_node", generate_node)

    graph.add_edge(START, "route_node")

    graph.add_conditional_edges(
        "route_node",
        route_condition,
        {
            "embedding": "embedding_slot_node",
            "fixed": "fixed_slot_node",
        },
    )

    graph.add_edge("embedding_slot_node", "connector_search_node")
    graph.add_edge("fixed_slot_node", "connector_search_node")

    graph.add_edge("connector_search_node", "generate_node")
    graph.add_edge("generate_node", END)

    return graph.compile()


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_qa(
    question: str,
    session_id: str = "default",
) -> dict[str, Any]:
    graph = get_graph()

    result = graph.invoke(
        {
            "question": question,
            "session_id": session_id,
        }
    )

    return {
        "question": question,
        "route": result.get("route"),
        "route_payload": result.get("route_payload", {}),
        "restaurant_list": result.get("restaurant_list", []),
        "answer": result.get("answer", ""),
    }


def main() -> None:
    """CLI 대화형 테스트 진입점.

    빈 Enter로 종료.
    route와 route_payload를 바탕으로 utils(db_fixed_search / db_embedding_search)를 직접 호출한다.
    """
    print("=" * 60)
    print("맛집 추천 CLI 테스트 (빈 Enter 입력시 종료)")
    print("=" * 60)

    while True:
        try:
            question = input("\n질문 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            break

        if not question:
            print("종료합니다.")
            break

        result = run_qa(
            question=question,
            session_id="default",
        )

        print("\n[답변]")
        print(result["answer"])


if __name__ == "__main__":
    main()