# planner_agent_langgraph.py
# 原 planner_agent.py, 使用 LangGraph 重构

import json
import re
import config
from typing import TypedDict, List, Dict, Any, Optional
import os  # 新增：导入 os 模块以便获取环境变量

# 新增: 导入各工具类，若失败则提供占位以避免 NameError
try:
    from knowledge_agent import KnowledgeRetrievalTool
    from storyteller_agent import StoryDraftingTool
    from guardian_agent import FactVerificationTool
    from video_generation_tool import VideoGenerationTool
except Exception as e:  # noqa
    print(f"[WARN] 工具模块导入失败，进入占位模式: {e}")
    class KnowledgeRetrievalTool:  # 占位，保持接口一致
        def invoke(self, *args, **kwargs): return {"error": "KnowledgeRetrievalTool 未可用"}
        def close(self): pass
    class StoryDraftingTool:
        def invoke(self, *args, **kwargs): return {"error": "StoryDraftingTool 未可用"}
        def close(self): pass
    class FactVerificationTool:
        def invoke(self, *args, **kwargs): return {"status": "Pass", "feedback": "占位审核通过 (真实工具未加载)"}
        def close(self): pass
    class VideoGenerationTool:
        def invoke(self, *args, **kwargs): return {"scene_id": 0, "video_url": "", "status": "PLACEHOLDER", "model": "none", "resolution": "480P", "duration": 10}
        def close(self): pass

# 安全导入 LangChain / LangGraph，不可用时使用占位实现保证主流程不崩溃
try:
    from langchain_openai import ChatOpenAI  # 可以兼容通义千问V1
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langgraph.graph import StateGraph, END
    LC_AVAILABLE = True
    LG_AVAILABLE = True
except Exception as e:
    print(f"[WARN] LangChain/LangGraph 依赖未完全可用，进入降级模式: {e}")
    LC_AVAILABLE = False
    LG_AVAILABLE = False
    # 占位类与函数
    class ChatOpenAI:  # noqa
        def __init__(self, *args, **kwargs):
            pass
        def invoke(self, *args, **kwargs):
            class R:
                content = ""
            return R()
    class ChatPromptTemplate:  # noqa
        @classmethod
        def from_messages(cls, messages):
            class Dummy:
                def __or__(self, other):
                    return self
                def invoke(self, d):
                    return ""
            return Dummy()
    class StrOutputParser:  # noqa
        def __or__(self, other):
            return self
        def invoke(self, x):
            return ""
    class StateGraph:  # noqa
        def __init__(self, *args, **kwargs):
            pass
        def add_node(self, *a, **k):
            pass
        def set_entry_point(self, *a, **k):
            pass
        def add_edge(self, *a, **k):
            pass
        def add_conditional_edges(self, *a, **k):
            pass
        def compile(self):
            class DummyApp:
                def invoke(self, initial_state, opts):
                    return {"error_message": "依赖缺失，无法执行真实流程"}
            return DummyApp()
    END = "END_PLACEHOLDER"


# --- 1. 定义 LangGraph 的状态 ---
class AgentState(TypedDict):
    """
    定义工作流中传递的状态 (仅提示词生成模式)。
    """
    user_input: str
    intent: List[str]
    theme: str
    original_action_request: Optional[str]
    knowledge_packet: Optional[Dict[str, Any]]
    supplemented_knowledge: Optional[Dict[str, Any]]
    master_prompt: Optional[str]
    negative_prompt: Optional[str]
    audit_result: Optional[Dict[str, Any]]
    final_package: Optional[Dict[str, Any]]
    error_message: Optional[str]


# --- 2. 定义 PlannerAgent 类 ---
class PlannerAgent:
    """
    使用 LangGraph 重构的规划Agent。
    它负责构建和编译工作流图。
    """

    def __init__(self):
        print("PlannerAgent: 初始化...")

        # 初始化 LLM (用于知识补充 和 输入解析)
        try:
            self.llm = ChatOpenAI(
                api_key=config.QWEN_API_KEY,
                base_url=config.QWEN_BASE_URL,
                model_name=config.PLANNER_LLM_MODEL,
                temperature=0.0
            )
            print("PlannerAgent: LLM 客户端 (for Qwen) 已初始化。")
        except Exception as e:
            print(f"PlannerAgent: 初始化 LLM 客户端失败: {e}")
            self.llm = None

        # --- 新增：为 parse_input_node 定义一个专用的提示词和链 ---
        self.input_parser_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "你是一个擅长理解用户意图的助手。"
             "用户的输入是一个关于非遗动作或镜头的请求。"
             "你的任务是：从这个请求中提取出最核心的、可用于数据库检索的【实体关键词】。"
             "【实体关键词】应该是一个具体的名词或短语，比如 '抱鱼娃娃', '三岔口', '开脸'。"
             "不要返回完整的句子或动作描述。"
             "只返回提取到的【实体关键词】。如果找不到，就返回 '未知'。"),
            ("user", "{user_input}")
        ])
        self.input_parser_chain = self.input_parser_prompt | self.llm | StrOutputParser()
        # --- 新增结束 ---

        # 初始化工具（移除视频工具）
        self.knowledge_tool = KnowledgeRetrievalTool()
        self.story_tool = StoryDraftingTool()
        self.guardian_tool = FactVerificationTool()
        # self.video_tool = VideoGenerationTool()  # 已禁用视频生成

        # 构建并编译 LangGraph 工作流
        self.app = self.build_graph()
        print("PlannerAgent: LangGraph 工作流已编译。")

    def _load_narrative_templates(self, filename=config.NARRATIVE_TEMPLATE_FILE):
        """加载叙事模板��JSON文件"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                templates = data.get("narrative_templates", [])
                print(f"PlannerAgent: 成功加载 {len(templates)} 个叙事模板。")
                return templates
        except Exception as e:
            print(f"PlannerAgent: 错误 - 加载叙事模板失败: {e}")
            return []

    def close_connections(self):
        """关闭所有子工具持有的连接"""
        print("PlannerAgent: 关闭子工具连接...")
        if hasattr(self.knowledge_tool, 'close'):
            self.knowledge_tool.close()
        if hasattr(self.guardian_tool, 'close'):
            self.guardian_tool.close()
        print("PlannerAgent: 所有连接已关闭。")

    def run(self, user_input: str) -> dict:
        """
        运行 LangGraph 工作流，增加健壮性处理，防止返回 None 导致调用端崩溃。
        返回统一字典结构：{error: msg} 或最终包。
        """
        print(f"\nPlannerAgent: 收到新请求: '{user_input}'")
        initial_state = {"user_input": user_input}
        try:
            final_state = self.app.invoke(initial_state, {"recursion_limit": 10})
        except Exception as e:
            print(f"[ERROR] LangGraph invoke 异常: {e}")
            return {"error": f"工作流执行异常: {e}"}

        if not final_state or not isinstance(final_state, dict):
            print("[ERROR] 工作流返回空或非字典状态。")
            return {"error": "工作流未产生有效状态 (final_state 为空或类型错误)。"}

        if final_state.get("error_message"):
            return {"error": final_state["error_message"], "details": final_state.get("audit_result", {})}
        pkg = final_state.get("final_package")
        if isinstance(pkg, dict):
            return pkg
        return {"error": "工作流执行完毕，但未生成最终结果。"}

    def _extract_entity_from_input(self, user_input: str) -> str:
        """
        (新增) 从用户输入中提取核心实体。
        优先匹配引号中的内容，若无则返回整个字符串。
        """
        # 匹配单引号、双引号、中文引号
        match = re.search(r"['\"‘“](.+?)['\"’”]", user_input)
        if match:
            entity = match.group(1).strip()
            print(f"从输入中提取到核心实体: '{entity}'")
            return entity

        # 如果没有引号，则作简单清理后返回
        # (例如去除 "生成一个"、"的动作" 等)
        cleaned_input = re.sub(r"^(生成一个|创建一个|描述一个)\s*", "", user_input.strip())
        cleaned_input = re.sub(r"\s*(的动作|的镜头|的场景)$", "", cleaned_input)
        print(f"未在输入中找到引号，使用清理后的输入作为实体: '{cleaned_input}'")
        return cleaned_input

    # --- 3. 定义 LangGraph 的节点 ---

    def parse_user_input_node(self, state: AgentState) -> Dict[str, Any]:
        """
        节点1: 解析用户输入（已升级：同时提取关键词和保存原始动作）。
        """
        print("--- (节点: 解析输入) ---")
        user_input = state["user_input"]

        try:
            # (这里的 self.input_parser_chain 是你调用LLM提取关键词的链)
            theme_keyword = self.input_parser_chain.invoke({"user_input": user_input}).strip()

            # (简单容错)
            if not theme_keyword or theme_keyword == "未知" or len(theme_keyword) > 20:
                theme_keyword = user_input.split("的")[0][-4:] # 简易回退

            print(f"从输入中提取到核心实体: '{theme_keyword}'")

            # --- ↓↓↓ 关键修改：同时保存原始输入和提取的关键词 ↓↓↓ ---
            return {
                "intent": ["动作生成"],
                "theme": theme_keyword, # 用于知识检索
                "original_action_request": user_input # 用于最终编剧
            }
            # --- ↑↑↑ 关键修改：同时保存原始输入和提取的关键词 ↑↑↑ ---

        except Exception as e:
            print(f"解析用户输入时出错: {e}")
            return {"error_message": "无法解析用户输入。"}

    def retrieve_knowledge_node(self, state: AgentState) -> Dict[str, Any]:
        """
        节点2: 调用 KnowledgeRetrievalTool 获取知识。
        """
        print("--- (节点: 检索知识) ---")
        theme = state["theme"]
        knowledge_packet = self.knowledge_tool.invoke({"theme": theme})

        if knowledge_packet.get("error"):
            print(f"知识库检索失败: {knowledge_packet.get('error')}. 尝试网络补充。")
            # 即使失败，也创建一个基础包以便网络补充
            knowledge_packet = {"artwork_name": theme.replace("《", "").replace("》", "")}

        return {"knowledge_packet": knowledge_packet}

    def _call_llm_search(self, query: str, expected_format: str) -> str:
        """(辅助函数) 调用LLM进行网络搜索来回答一个具体问题。"""
        if not self.llm: return ""

        system_prompt = f"""
你是一个信息检索助手。请根据用户的问题，提供一个简短、事实准确的答案。
你的答案必须严格符合以下格式：{expected_format}。
- 如果找到了信息，请只返回该信息 (例如 "张三", "清代", "吉祥如意")。
- 绝对不要添加任何解释或前言 (例如 "根据我的搜索...")。
- 如果你找不到准确的信息，请只返回一个空字符串 ""。
"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt.strip()),
            ("user", "{query}")
        ])
        chain = prompt | self.llm | StrOutputParser()

        try:
            answer = chain.invoke({"query": query}).strip()

            # 避免LLM返回 "我不知道" 之类的无效答案
            if ("不知道" in answer or "找不到" in answer or "无法" in answer or
                    "抱歉" in answer or "没有记载" in answer or "不详" in answer or
                    "根据" in answer or "资料" in answer or "作品" in answer):
                return ""
            return answer
        except Exception as e:
            print(f"PlannerAgent: [LLM 搜索] 调用失败: {e}")
            return ""

    def supplement_knowledge_node(self, state: AgentState) -> Dict[str, Any]:
        """
        节点3: 检查知识包，对缺失的关键信息进行网络搜索补充。
        增强：新增 creator_name / era / creation_year / creation_region 的严格检索与验证；
        若无法可靠获取则留空，由后续 prompt 使用安全占位，不臆造。
        """
        print("--- (节点: 补充知识) ---")
        knowledge_packet = state["knowledge_packet"].copy()  # 复制一份进行修改
        theme = state["theme"]
        search_theme = theme.replace("《", "").replace("》", "")

        # 允许的典型年代/朝代关键词，用于简单校验避免幻觉
        allowed_era_tokens = [
            "清", "清代", "乾隆", "嘉庆", "道光", "咸丰", "光绪", "民国", "晚清", "近代"
        ]
        def _validate_era(val: str) -> str:
            if not val: return ""
            for token in allowed_era_tokens:
                if token in val:
                    return token if len(val) > 8 else val  # 保留精简
            # 若回答包含"代"或数字年份区间也接受
            if any(ch.isdigit() for ch in val) and '年' in val:
                return val[:12]
            return ""  # 不在允许集合则丢弃

        def _extract_year(val: str) -> str:
            if not val: return ""
            import re
            m = re.search(r"(1|2)\d{3}", val)
            if m:
                year = m.group(0)
                y_int = int(year)
                if 1500 <= y_int <= 2025:
                    return year
            return ""

        # 1. 作者 (已有则不覆盖)
        if not knowledge_packet.get("creator_name"):
            q = f"杨柳青年画《{search_theme}》的创作者或常见署名是谁？只回答名字或'佚名'。"
            candidate = self._call_llm_search(q, "姓名或佚名")
            if candidate and len(candidate) <= 12:
                knowledge_packet["creator_name"] = candidate

        # 2. 时代 / 年代
        if not knowledge_packet.get("era"):
            q = f"杨柳青年画《{search_theme}》成画的大致年代是什么？回答如: 清代/光绪/民国/晚清。"
            era_ans = self._call_llm_search(q, "年代")
            era_clean = _validate_era(era_ans)
            if era_clean:
                knowledge_packet["era"] = era_clean

        # 3. 创作年份 (精确年份可选)
        if not knowledge_packet.get("creation_year"):
            q = f"杨柳青年画《{search_theme}》是否有明确的创作年份记录？若有给出阿拉伯数字年份，没有则留空。"
            year_ans = self._call_llm_search(q, "四位数字或空")
            year_clean = _extract_year(year_ans)
            if year_clean:
                knowledge_packet["creation_year"] = year_clean

        # 4. 流行地域/产地产地 (region)
        if not knowledge_packet.get("creation_region"):
            q = f"杨柳青年画《{search_theme}》主要流行或制作的地域是哪里？回答包含'杨柳青'或省市名。"
            region_ans = self._call_llm_search(q, "地域")
            if region_ans and ("杨柳青" in region_ans or len(region_ans) <= 10):
                knowledge_packet["creation_region"] = region_ans[:12]

        # 5. 原有寓意补充逻辑 (保持不变，下方复用原对象、人物补齐)
        objects = knowledge_packet.get("objects", []) or []
        has_symbol = any(o.get("object_symbol") for o in objects if isinstance(o, dict))
        if not has_symbol:
            q = f"杨柳青年画《{search_theme}》核心寓意之一是什么？回答如: 连年有余/吉祥如意。"
            symbol = self._call_llm_search(q, "寓意词")
            if symbol:
                if not objects:
                    objects = [{"object_name": "核心元素", "object_symbol": symbol[:24]}]
                else:
                    if isinstance(objects[0], dict):
                        objects[0]["object_symbol"] = objects[0].get("object_symbol") or symbol[:24]
                knowledge_packet["objects"] = objects

        # 若 objects 数量极少，尝试批量补充 (保持原逻辑，略微收缩长度)
        if len([o for o in objects if isinstance(o, dict) and o.get('object_name')]) < 2 and self.llm:
            try:
                sys_prompt = "你是一名严谨的非遗知识助理。只输出 JSON 数组，不添加解释。"
                user_prompt = (
                    f"围绕杨柳青年画《{search_theme}》，列出 3 个常见具象元素及其象征寓意。"
                    f"输出 JSON 数组，每项 {{'object_name':'元素','object_symbol':'寓意'}}，寓意简洁。"
                )
                chain = ChatPromptTemplate.from_messages([
                    ("system", sys_prompt),
                    ("user", "{q}")
                ]) | self.llm | StrOutputParser()
                raw_json = chain.invoke({"q": user_prompt}).strip()
                raw_json = raw_json[raw_json.find('['): raw_json.rfind(']')+1] if '[' in raw_json and ']' in raw_json else '[]'
                import json as _json
                extra_objs = _json.loads(raw_json)
                cleaned = []
                for item in extra_objs:
                    if not isinstance(item, dict):
                        continue
                    n = (item.get('object_name') or '').strip()[:10]
                    s = (item.get('object_symbol') or '').strip()[:20]
                    if n and s and all(n != old.get('object_name') for old in objects if isinstance(old, dict)):
                        cleaned.append({"object_name": n, "object_symbol": s})
                if cleaned:
                    knowledge_packet.setdefault('objects', [])
                    knowledge_packet['objects'].extend(cleaned)
            except Exception as e:
                print(f"[WARN] 批量补充对象寓意失败: {e}")

        # 人物/形象补齐 (保持)
        figures_list = knowledge_packet.get('figures', []) or []
        if not any(f.get('figure_name') for f in figures_list if isinstance(f, dict)) and self.llm:
            try:
                sys_prompt = "你是一名非遗年画知识整理助手。只输出 JSON 数组，无解释。"
                user_prompt = (
                    f"列出杨柳青年画《{search_theme}》或同题材常见的1个典型人物/形象。"
                    f"输出 JSON 数组, 每项 {{'figure_name':'名称','figure_symbol':'寓意(可为空)'}}。"
                )
                chain = ChatPromptTemplate.from_messages([
                    ("system", sys_prompt),
                    ("user", "{q}")
                ]) | self.llm | StrOutputParser()
                raw_json = chain.invoke({"q": user_prompt}).strip()
                raw_json = raw_json[raw_json.find('['): raw_json.rfind(']')+1] if '[' in raw_json and ']' in raw_json else '[]'
                import json as _json
                extra_figs = _json.loads(raw_json)
                cleaned_f = []
                for item in extra_figs:
                    if not isinstance(item, dict):
                        continue
                    n = (item.get('figure_name') or '').strip()[:10]
                    s = (item.get('figure_symbol') or '').strip()[:24]
                    if n and all(n != old.get('figure_name') for old in figures_list if isinstance(old, dict)):
                        cleaned_f.append({"figure_name": n, "figure_symbol": s})
                if cleaned_f:
                    knowledge_packet.setdefault('figures', [])
                    knowledge_packet['figures'].extend(cleaned_f)
            except Exception as e:
                print(f"[WARN] 补充 figures 失败: {e}")

        print("知识补充完毕。")
        return {"supplemented_knowledge": knowledge_packet}

    def select_template_node(self, state: AgentState) -> Dict[str, Any]:
        """
        节点4: 根据意图和*可用知识*选择最匹配的叙事模板。
        (原 _select_narrative_template)
        """
        print("--- (节点: 选择模板) ---")
        user_intent = state["intent"]
        knowledge_packet = state["supplemented_knowledge"]

        if not self.narrative_templates:
            return {"error_message": "叙事模板库为空，无法选择模板。"}

        # 分析知识包里有哪些“关键”知识
        available_knowledge_keys = set()
        if knowledge_packet.get("creator_name"): available_knowledge_keys.add("protagonist_identity")
        if knowledge_packet.get("techniques"): available_knowledge_keys.add("key_technique_description")
        if knowledge_packet.get("figures"): available_knowledge_keys.add("character_name_in_artwork")
        if any(o.get("object_symbol") for o in knowledge_packet.get("objects", []) if isinstance(o, dict)):
            available_knowledge_keys.add("artwork_meaning_text")

        print(f"分析得到可用关键知识: {available_knowledge_keys}")

        best_template = None
        max_score = -999

        for template in self.narrative_templates:
            intent_score = 0
            for intent in user_intent:
                if intent in template.get("applicable_intent", []): intent_score += 1
                if intent == template.get("video_knowledge_type", ""): intent_score += 1

            required_slots_list = template.get("knowledge_slots", [])
            knowledge_score = 0
            if required_slots_list:
                for slot in required_slots_list:
                    if slot in available_knowledge_keys:
                        knowledge_score += 1
                knowledge_match_ratio = knowledge_score / len(required_slots_list)
            else:
                knowledge_match_ratio = 0.5  # 模板不需要特定知识

            final_score = intent_score
            if knowledge_match_ratio > 0:
                final_score += (knowledge_match_ratio * 5)
            elif required_slots_list:
                final_score -= 10  # 惩罚需要知识但知识不匹配的模板

            if final_score > max_score:
                max_score = final_score
                best_template = template

        if best_template:
            print(f"选中模板: '{best_template.get('short_video_paradigm', '未知')}' (得分: {max_score:.2f})")
            return {"selected_template": best_template}
        else:
            return {"error_message": "无法根据可用知识选择合适的叙事模板。"}

    def draft_story_node(self, state: AgentState) -> Dict[str, Any]:
        """
        节点5: (重构) 调用 StoryDraftingTool 生成 master_prompt 和 negative_prompt。
        """
        print("--- (节点: 起草故事/生成提示词) ---")

        # --- ↓↓↓ 关键修改：传入两个参数 ↓↓↓ ---
        result = self.story_tool.invoke({
            "knowledge_packet": state["supplemented_knowledge"],
            "original_action_request": state["original_action_request"] # 把保存的动作请求传进去
        })
        # --- ↑↑↑ 关键修改：传入两个参数 ↑↑↑ ---

        if result.get("error"):
            return {"error_message": result["error"]}

        # (后续逻辑保持不变，它会自动存入 AgentState)
        return {
            "master_prompt": result.get("master_prompt"),
            "negative_prompt": result.get("negative_prompt")
        }

    def audit_draft_node(self, state: AgentState) -> Dict[str, Any]:
        """
        节点6: 调用 FactVerificationTool 审核故事。
        """
        print("--- (节点: 审核事实) ---")
        scenes = state.get("scenes_for_audit")
        if not scenes:
            print("[WARN] 审核节点未找到'scenes_for_audit'，跳过审核。")
            return {"audit_result": {"status": "Pass", "feedback": "无场景可审核，自动通过。"}}

        audit_result = self.guardian_tool.invoke({"story_draft_scenes": scenes})

        print(f"审核结果: {audit_result.get('status')}")
        return {"audit_result": audit_result}

    def trim_storyboard_node(self, state: AgentState) -> Dict[str, Any]:
        """
        新增节点: 强制裁剪 storyboard 到最多3个场景。
        """
        print("--- (节点: 裁剪分镜) ---")
        storyboard = state.get("final_storyboard")
        if storyboard and len(storyboard) > 3:
            print(f"分镜数量 {len(storyboard)} > 3，裁剪为前3个。")
            trimmed_storyboard = storyboard[:3]
            return {"final_storyboard": trimmed_storyboard}
        elif storyboard:
            print(f"分镜数量 {len(storyboard)} <= 3，无需裁剪。")
        return {} # 不需修改状态

    # --- 调试: 通用节点包装器 ---
    def _ensure_state(self, state, node_name: str) -> Dict[str, Any]:
        if state is None:
            print(f"[FATAL] 节点 {node_name} 收到的 state 为 None。")
            return {"error_message": f"内部错误：节点 {node_name} 收到空状态。"}
        if not isinstance(state, dict):
            print(f"[FATAL] 节点 {node_name} 收到的 state 类型异常: {type(state)}")
            return {"error_message": f"内部错误：节点 {node_name} 状态类型异常。"}
        return {}

    def generate_video_node(self, state: AgentState) -> Dict[str, Any]:
        pass  # 保留空壳占位，已弃用

    def assemble_package_node(self, state: AgentState) -> Dict[str, Any]:
        print("--- (节点: 组装提示词包) ---")
        master_prompt = state.get("master_prompt")
        if not master_prompt:
            return {"error_message": "组装失败：'master_prompt' 在状态中缺失。"}
        rich_prompt_package = {
            "master_prompt": master_prompt,
            "negative_prompt": state.get("negative_prompt"),
            "_source_knowledge_packet": state.get("supplemented_knowledge", {}),
            "mode": "PROMPT_ONLY",
            "notes": "已禁用自动视频生成，本包用于外部平台文生视频实验。"
        }
        return {"final_package": rich_prompt_package}

    # --- 4. 定义 LangGraph 的边 ---

    def should_proceed_after_audit(self, state: AgentState) -> str:
        """
        条件边: 检查审核结果。
        如果审核通过，则进入“组装包”节点。
        如果审核失败，则结束。
        """
        print("--- (条件边: 检查审核) ---")
        audit_result = state["audit_result"]
        if audit_result.get("status") == "Pass":
            print("审核通过，流向 -> [组装包]")
            return "assemble_package"
        else:
            print("审核失败，流向 -> [结束]")
            feedback = audit_result.get('feedback', '未知审核错误')
            # 将错误信息存入状态，以便
            state["error_message"] = "故事审核未通过"
            return END

    # --- 5. 构建图 ---
    def build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("parse_input", self.parse_user_input_node)
        workflow.add_node("retrieve_knowledge", self.retrieve_knowledge_node)
        workflow.add_node("supplement_knowledge", self.supplement_knowledge_node)
        workflow.add_node("draft_story", self.draft_story_node)
        workflow.add_node("assemble_package", self.assemble_package_node)
        workflow.set_entry_point("parse_input")
        workflow.add_edge("parse_input", "retrieve_knowledge")
        workflow.add_edge("retrieve_knowledge", "supplement_knowledge")
        workflow.add_edge("supplement_knowledge", "draft_story")
        workflow.add_edge("draft_story", "assemble_package")
        workflow.add_edge("assemble_package", END)
        return workflow.compile()


# --- 用于独立测试 Planner Agent ---
if __name__ == '__main__':
    # 注意：运行此测试前，请确保 config.py 中的 LANGCHAIN_API_KEY 已设置（如果启用了Tracing）
    # 并且 Neo4j 服务正在运行

    planner = PlannerAgent()
    test_input = "给我讲个《莲年有余》的故事，要喜庆一点的"

    result = planner.run(test_input)

    print("\n--- 最终生成的丰富提示词包 ---")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("------------------------------")

    # 清理连接
    planner.close_connections()
