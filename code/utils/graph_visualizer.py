"""
LangGraph 图结构可视化与美化工具包
提供自动语义着色、自定义节点样式增强功能
"""

import re
from typing import Any, Dict, Optional, Union

# 默认语义化配色方案 (柔和现代配色体系)
DEFAULT_SEMANTIC_PALETTE: Dict[str, str] = {
    # LLM / 推理相关节点 (经典科技蓝)
    "model": "fill:#E3F2FD,stroke:#1E88E5,stroke-width:2px,color:#0D47A1",
    "agent": "fill:#E3F2FD,stroke:#1E88E5,stroke-width:2px,color:#0D47A1",
    "llm": "fill:#E3F2FD,stroke:#1E88E5,stroke-width:2px,color:#0D47A1",
    "chat": "fill:#E3F2FD,stroke:#1E88E5,stroke-width:2px,color:#0D47A1",
    # 工具 / 外部执行相关节点 (活力森林绿)
    "tools": "fill:#E8F5E9,stroke:#43A047,stroke-width:2px,color:#1B5E20",
    "tool": "fill:#E8F5E9,stroke:#43A047,stroke-width:2px,color:#1B5E20",
    "action": "fill:#E8F5E9,stroke:#43A047,stroke-width:2px,color:#1B5E20",
    "execute": "fill:#E8F5E9,stroke:#43A047,stroke-width:2px,color:#1B5E20",
    # 中间件 / 人机协同 / 审批拦截 (暖调预警橙)
    "middleware": "fill:#FFF3E0,stroke:#FB8C00,stroke-width:2px,color:#E65100",
    "human": "fill:#FFF3E0,stroke:#FB8C00,stroke-width:2px,color:#E65100",
    "approval": "fill:#FFF3E0,stroke:#FB8C00,stroke-width:2px,color:#E65100",
    "interrupt": "fill:#FFF3E0,stroke:#FB8C00,stroke-width:2px,color:#E65100",
    # 路由 / 条件判断 / 审核 (优雅深邃紫)
    "router": "fill:#F3E5F5,stroke:#8E24AA,stroke-width:2px,color:#4A148C",
    "condition": "fill:#F3E5F5,stroke:#8E24AA,stroke-width:2px,color:#4A148C",
    "judge": "fill:#F3E5F5,stroke:#8E24AA,stroke-width:2px,color:#4A148C",
    "eval": "fill:#F3E5F5,stroke:#8E24AA,stroke-width:2px,color:#4A148C",
    # 知识检索 / RAG / 搜索 (清新科技青)
    "retrieve": "fill:#E0F7FA,stroke:#00ACC1,stroke-width:2px,color:#006064",
    "search": "fill:#E0F7FA,stroke:#00ACC1,stroke-width:2px,color:#006064",
    "rag": "fill:#E0F7FA,stroke:#00ACC1,stroke-width:2px,color:#006064",
    # 数据转换 / 处理 (简约烟灰色)
    "format": "fill:#ECEFF1,stroke:#78909C,stroke-width:2px,color:#263238",
    "process": "fill:#ECEFF1,stroke:#78909C,stroke-width:2px,color:#263238",
}


# 语义匹配优先级与关键词定义（按从具体到通用排序）
SEMANTIC_PRIORITY_GROUPS = [
    # 1. 中间件 / 人机拦截 / 审批
    (["middleware", "human", "approval", "interrupt", "hitl"], "middleware"),
    # 2. 路由 / 条件分支 / 判定
    (["router", "condition", "judge", "eval", "check", "decision"], "router"),
    # 3. 检索 / 知识库 / 搜索
    (["retrieve", "search", "rag", "retriever", "kb"], "retrieve"),
    # 4. 工具 / 外部执行
    (["tools", "tool", "action", "execute"], "tools"),
    # 5. 模型 / 推理智能体
    (["model", "agent", "llm", "chat"], "model"),
    # 6. 数据处理 / 格式化
    (["format", "process", "transform"], "format"),
]


def escape_mermaid_node_id(node_id: str) -> str:
    """转义 Mermaid 节点标识符中的特殊符号（如点号转为 \\2e）"""
    return node_id.replace(".", r"\2e")


def extract_node_ids(mermaid_code: str) -> list[str]:
    """从 Mermaid 流程图代码中提取所有自定义节点 ID（排除系统保留的 __start__ 和 __end__）"""
    node_ids = set()
    # 匹配类似 node_id(label) 或 node_id[label] 或 node_id{label} 的定义行
    pattern = re.compile(
        r"^\s*([A-Za-z0-9_\\.]+)(?:\(.*?\)|\{.*?\}|\[.*?\])", re.MULTILINE
    )
    for match in pattern.finditer(mermaid_code):
        node = match.group(1).strip()
        if node not in ["__start__", "__end__", "graph", "subgraph"]:
            node_ids.add(node)
    return list(node_ids)


def match_semantic_style(node_id: str) -> Optional[str]:
    """根据节点名称关键词及优先级，自动匹配对应的语义化颜色样式"""
    normalized_name = node_id.replace(r"\2e", ".").lower()
    for keywords, category in SEMANTIC_PRIORITY_GROUPS:
        for kw in keywords:
            if kw in normalized_name:
                return DEFAULT_SEMANTIC_PALETTE.get(category)
    return None


def format_style_dict(style: Union[str, Dict[str, str]]) -> str:
    """将样式字典或样式字符串格式化为标准 Mermaid style 属性串"""
    if isinstance(style, dict):
        return ",".join(f"{k}:{v}" for k, v in style.items())
    return str(style)


def colorize_mermaid(
    mermaid_code: str,
    custom_styles: Optional[Dict[str, Union[Dict[str, str], str]]] = None,
    auto_semantic: bool = True,
) -> str:
    """
    为 Mermaid 流程图注入节点颜色样式
    
    :param mermaid_code: 原始 Mermaid 代码字符串
    :param custom_styles: 用户自定义节点样式映射表，例如：
                          {"model": "fill:#E3F2FD,stroke:#1E88E5", "tools": {"fill": "#E8F5E9"}}
    :param auto_semantic: 是否开启基于节点名称关键词的自动语义着色（默认为 True）
    :return: 增强着色后的 Mermaid 代码字符串
    """
    applied_styles: Dict[str, str] = {}
    node_ids = extract_node_ids(mermaid_code)

    # 1. 自动语义化匹配
    if auto_semantic:
        for node_id in node_ids:
            semantic_style = match_semantic_style(node_id)
            if semantic_style:
                applied_styles[node_id] = semantic_style

    # 2. 用户自定义样式覆盖
    if custom_styles:
        for raw_node, style in custom_styles.items():
            escaped_node = escape_mermaid_node_id(raw_node)
            applied_styles[escaped_node] = format_style_dict(style)

    if not applied_styles:
        return mermaid_code

    style_lines = [
        f"        style {node_id} {style_val}"
        for node_id, style_val in applied_styles.items()
    ]
    return mermaid_code.rstrip() + "\n\n" + "\n".join(style_lines) + "\n"


def draw_colorized_mermaid(
    graph_or_agent: Any,
    custom_styles: Optional[Dict[str, Union[Dict[str, str], str]]] = None,
    auto_semantic: bool = True,
    print_code: bool = True,
    **mermaid_kwargs: Any,
) -> str:
    """
    一步生成并着色 LangGraph 图的 Mermaid 流程图代码
    
    :param graph_or_agent: LangGraph StateGraph/CompiledStateGraph 或包含 get_graph() 方法的 Agent 对象
    :param custom_styles: 用户自定义样式字典
    :param auto_semantic: 是否开启自动语义着色（默认为 True）
    :param print_code: 是否直接在终端打印生成的 Mermaid 字符串（默认为 True）
    :param mermaid_kwargs: 传递给底层 draw_mermaid() 的额外参数（如 xray=True 等）
    :return: 着色后的 Mermaid 字符串
    """
    if hasattr(graph_or_agent, "get_graph"):
        graph = graph_or_agent.get_graph()
    else:
        graph = graph_or_agent

    raw_mermaid = graph.draw_mermaid(**mermaid_kwargs)
    colorized_code = colorize_mermaid(
        raw_mermaid, custom_styles=custom_styles, auto_semantic=auto_semantic
    )

    if print_code:
        print(colorized_code)

    return colorized_code
