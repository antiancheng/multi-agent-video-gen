# guardian_tool.py
# 原 guardian_agent.py, 重构为 LangChain Tool

from neo4j import GraphDatabase
import json
import re
import config  # 导入配置

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, List, Dict, Any, Tuple


class FactVerificationInput(BaseModel):
    """用于事实审核工具的输入模型"""
    story_draft_scenes: List[Dict[str, Any]] = Field(description="从故事模板中提取的场景列表，用于审核")


class FactVerificationTool(BaseTool):
    """
    一个 LangChain 工具，负责审核故事模板中的事实，确保文化保真性。
    （重构：新增多模态评估功能）
    """
    name: str = "fact_verification_tool"
    description: str = "审核故事模板或评估生成视频的文化保真度。"
    args_schema: Type[BaseModel] = FactVerificationInput

    _driver: Any = None
    _vlm_llm: Any = None # 新增：用于VLM调用的LLM客户端

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # ... (Neo4j驱动初始化保持不变)

        # 初始化VLM客户端 (例如 qwen-vl-max)
        try:
            from langchain_openai import ChatOpenAI
            self._vlm_llm = ChatOpenAI(
                api_key=config.QWEN_API_KEY,
                base_url=config.QWEN_BASE_URL,
                model_name="qwen-vl-max", # 指定VLM模型
                temperature=0.0
            )
            print("FactVerificationTool: VLM 客户端 (qwen-vl-max) 已初始化。")
        except Exception as e:
            print(f"FactVerificationTool: 初始化 VLM 客户端失败: {e}")
            self._vlm_llm = None

    # ... (close, _execute_query, _extract_facts_from_draft, _verify_fact 保持不变)

    def evaluate_video_fidelity(self, generated_video_path: str, ground_truth_knowledge: Dict[str, Any]) -> Dict[str, Any]:
        """
        （新增）评估生成视频的文化保真度。
        """
        if not self._vlm_llm:
            return {"error": "VLM客户端未初始化，无法进行评估。"}

        key_elements = ground_truth_knowledge.get("textual_knowledge_list", [])
        ground_truth_url = ground_truth_knowledge.get("ground_truth_url")

        if not key_elements:
            return {"error": "知识包中缺少 'textual_knowledge_list'。"}

        passed_elements = 0
        failed_elements = 0

        # 评估方法1：文本-视频
        for element in key_elements:
            prompt = f"看这个视频，视频中的人物是否符合 '{element}' 的特征？"
            # 此处需要实现调用VLM并传入视频和文本的逻辑
            # ... (调用VLM的代码) ...
            # 假设VLM返回 "是" 或 "否"
            vlm_response = "是" # 伪代码
            if vlm_response == "是":
                passed_elements += 1
            else:
                failed_elements += 1

        # 评估方法2：图-视频
        if ground_truth_url:
            prompt = f"对比这个视频（A）和这张标准答案图片（B），它们的 '服饰' 是否一致？"
            # 此处需要实现调用VLM并传入视频和图片的逻辑
            # ... (调用VLM的代码) ...
            vlm_response = "是" # 伪代码
            if vlm_response == "是":
                passed_elements += 1
            else:
                failed_elements += 1

        total_elements = len(key_elements) + (1 if ground_truth_url else 0)
        fidelity_score = passed_elements / total_elements if total_elements > 0 else 0

        return {
            "fidelity_score": fidelity_score,
            "passed_elements": passed_elements,
            "failed_elements": failed_elements
        }

    def _run(self, story_draft_scenes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """LangChain Tool的执行入口 (旧逻辑，保留用于兼容)。"""
        print("FactVerificationTool: 收到场景列表，开始审核...")
        if not self._driver:
            return {"status": "Fail", "feedback": ["Neo4j 驱动未初始化，无法审核。"]}

        facts_to_verify = self._extract_facts_from_draft(story_draft_scenes)
        if not facts_to_verify:
            print("FactVerificationTool: 未提取到需要核查的事实，审核通过。")
            return {"status": "Pass", "feedback": "未提取到需要核查的事实。"}

        errors = []
        verified_count = 0
        for fact in facts_to_verify:
            is_valid, message = self._verify_fact(fact)
            print(f"FactVerificationTool: 核查 '{fact.get('original_text', fact)}' -> {message}")
            if not is_valid:
                errors.append(message)
            verified_count += 1

        if errors:
            print(f"FactVerificationTool: 审核未通过，发现 {len(errors)} 处事实错误。")
            return {"status": "Fail", "feedback": errors}
        else:
            print(f"FactVerificationTool: 审核通过，所有 {verified_count} 个事实均与知识库一致。")
            return {"status": "Pass", "feedback": f"所有 {verified_count} 个事实均核查通过。"}

    async def _arun(self, story_draft_scenes: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self._run(story_draft_scenes)


# --- Test ---
if __name__ == "__main__":
    tool = FactVerificationTool()
    mock_draft_pass = [
        {"scene": 1, "description_template": "画师使用了经典的四色套印技法。"},
        {"scene": 2, "description_template": "这个技法叫半印半绘。"}
    ]

    print("\n--- 测试审核通过场景 ---")
    result_pass = tool.invoke({"story_draft_scenes": mock_draft_pass})
    print(json.dumps(result_pass, indent=2, ensure_ascii=False))

    tool.close()
