# knowledge_tool.py
# 原 knowledge_agent.py, 重构为 LangChain Tool

from neo4j import GraphDatabase
import json
import config  # 导入配置
import os, re, base64, mimetypes
import random

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field  # 修复弃用
from typing import Type, Optional, Dict, Any


class KnowledgeRetrievalInput(BaseModel):
    """用于知识检索工具的输入模型"""
    theme: str = Field(description="需要检索的知识主题，例如 '《莲年有余》'")


class KnowledgeRetrievalTool(BaseTool):
    """
    一个 LangChain 工具，负责从 Neo4j 知识图谱中检索与特定主题相关的知识。
    """
    name: str = "knowledge_retrieval_tool"
    description: str = "用于从Neo4j知识图谱中检索特定主题（如作品名称）的详细信息。"
    args_schema: Type[BaseModel] = KnowledgeRetrievalInput

    _driver: Any = None  # Neo4j 驱动

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._driver = None
        try:
            self._driver = GraphDatabase.driver(
                config.NEO4J_URI,
                auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
                database=config.NEO4J_DATABASE
            )
            self._driver.verify_connectivity()
            print("KnowledgeTool: Neo4j 驱动已初始化并连接成功。")
        except Exception as e:
            print(f"KnowledgeTool: Neo4j 驱动初始化失败: {e}")
            self._driver = None

    def close(self):
        if self._driver:
            self._driver.close()
            print("KnowledgeTool: Neo4j 驱动已关闭。")

    def _execute_query(self, query, params={}):
        if not self._driver:
            raise ConnectionError("KnowledgeTool: Neo4j 驱动未初始化。")
        with self._driver.session() as session:
            result = session.run(query, params)
            # 使用 list(result) 来消费所有记录
            return [r.data() for r in result]

    def _extract_keyword_from_theme(self, theme: str) -> str:
        """
        (新增) 从用户输入中提取核心关键词。
        这是一个简化的实现，优先使用引号内的内容。
        """
        # 匹配单引号、双引号、中文引号
        match = re.search(r"['\"‘“](.+?)['\"’”]", theme)
        if match:
            keyword = match.group(1).strip()
            print(f"KnowledgeTool: 从主题中提取到引号内关键词: '{keyword}'")
            return keyword

        # 如果没有引号，则作简单清理后返回
        cleaned_input = re.sub(r"^(生成一个|创建一个|描述一个)\s*", "", theme.strip())
        cleaned_input = re.sub(r"\s*(的动作|的镜头|的场景)$", "", cleaned_input)
        print(f"KnowledgeTool: 未在输入中找到引号，使用清理后的输入作为关键词: '{cleaned_input}'")
        return cleaned_input

    def _run(self, theme: str) -> dict:
        print(f"KnowledgeTool: 收到检索指令，主题(关键词): '{theme}'")
        if not self._driver:
            return {"error": "Neo4j 驱动未初始化，无法执行查询。"}

        # 重写 Cypher 查询以获取文本描述和评估用的图片URL
        query = """
        MATCH (n)
        WHERE n.name CONTAINS $keyword
        OPTIONAL MATCH (n)-[:HAS_MEDIA]->(m:Media)
        RETURN
            // 尝试从多个可能的属性中获取文本描述
            COALESCE(n.description, n.feature, n.introduction, n.name) as textual_knowledge,
            // 获取用于评估的图片URL
            COALESCE(m.url, m.media_url, m.name) as ground_truth_url
        LIMIT 1
        """
        params = {"keyword": theme}

        try:
            rows = self._execute_query(query, params)
            if rows:
                result = rows[0]

                knowledge_packet = {
                    "textual_knowledge": result.get("textual_knowledge"),
                    "ground_truth_url": result.get("ground_truth_url")
                }

                # 添加日志和回退逻辑
                if not knowledge_packet["textual_knowledge"]:
                    print(f"KnowledgeTool: 找到了节点，但缺少文本描述。")
                    knowledge_packet["textual_knowledge"] = f"关于 {theme} 的通用描述。"
                if not knowledge_packet["ground_truth_url"]:
                    print(f"KnowledgeTool: 找到了节点，但缺少关联的图片URL。")
                    knowledge_packet["ground_truth_url"] = getattr(config, 'DASHSCOPE_FALLBACK_IMAGE_URL', 'https://example.com/placeholder.png')

                print(f"KnowledgeTool: 成功检索到知识: {knowledge_packet}")
                return knowledge_packet
            else:
                print(f"KnowledgeTool: 未找到与关键词 '{theme}' 相关的知识。")
                return {"error": f"未找到与关键词 '{theme}' 相关的知识。"}
        except Exception as e:
            print(f"KnowledgeTool: 查询执行失败: {e}")
            return {"error": f"查询执行失败: {e}"}

    async def _arun(self, theme: str) -> dict:
        """异步执行入口"""
        pass


# --- Test ---
if __name__ == "__main__":
    tool = KnowledgeRetrievalTool()
    # test_theme = "《金蟾童乐》"
    test_theme = "《莲年有余》" # 使用你更新了 URL 的节点测试
    packet = tool.invoke({"theme": test_theme})
    print("\n--- 知识包预览 ---")
    print(json.dumps(packet, indent=2, ensure_ascii=False))
    print("------------------")
    tool.close()
