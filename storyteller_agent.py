# storyteller_agent.py
# 原 storyteller_tool.py, 重构为 LangChain Tool

import json
import config
import re  # 新增: 视觉要素解析所需
from typing import Type, Dict, Any, List
try:
    from pydantic import BaseModel, Field
except Exception as e:
    print(f"[WARN] pydantic 未安装，使用占位: {e}")
    class BaseModel:  # 占位，忽略校验
        def __init__(self, **kwargs):
            pass
    def Field(*args, **kwargs):
        return None

# 尝试导入 LangChain 组件，失败则使用占位实现
try:
    from langchain_core.tools import BaseTool
    from langchain_openai import ChatOpenAI

    LC_AVAILABLE = True
except Exception as e:  # noqa
    LC_AVAILABLE = False
    print("[WARN] LangChain 相关依赖未成功导入，将使用最小占位实现（无真实LLM调用）:", e)


    class BaseTool:  # 占位工具基类
        def __init__(self, **kwargs):
            pass


    class ChatOpenAI:  # 占位 LLM
        def __init__(self, *args, **kwargs):
            pass

        def invoke(self, messages):
            class R:  # 简单响应对象
                content = "{}"

            return R()


class StoryDraftingInput(BaseModel):
    """用于故事起草工具的输入模型（重构版：需要知识包+原始动作）"""
    knowledge_packet: Dict[str, Any] = Field(description="从知识库检索并补充后的知识包")

    # --- ↓↓↓ 关键修改：增加这个新输入 ↓↓↓ ---
    original_action_request: str = Field(description="用户输入的原始动作描述，例如‘娃娃从窗户跳下来’")
    # --- ↑↑↑ 关键修改：增加这个新输入 ↑↑↑ ---


class StoryDraftingTool(BaseTool):
    """
    一个 LangChain 工具，负责根据知识包和叙事模板生成故事初稿。
    """
    name: str = "story_drafting_tool"
    description: str = "根据知识包和叙事模板生成包含多角色对话的故事分镜。"
    args_schema: Type[BaseModel] = StoryDraftingInput
    llm: Any = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("StoryDraftingTool: 已初始化。")
        if LC_AVAILABLE:
            try:
                self.llm = ChatOpenAI(
                    api_key=getattr(config, 'QWEN_API_KEY', ''),
                    base_url=getattr(config, 'QWEN_BASE_URL', ''),
                    model_name=getattr(config, 'STORYTELLER_LLM_MODEL',
                                       getattr(config, 'PLANNER_LLM_MODEL', 'qwen-turbo')),
                    temperature=0.7
                )
                print("StoryDraftingTool: LLM 客户端 (for Qwen) 已初始化。")
            except Exception as e:
                print("[WARN] 初始化真实 LLM 失败, 使用占位: ", e)
                self.llm = None
        else:
            self.llm = None

    def _generate_production_bible(self, knowledge: Dict[str, Any], template: Dict[str, Any]) -> Dict[str, Any]:
        """
        (V3 核心) 生成制作宝典。
        TODO: 将来这里应该调用LLM，根据知识和模板动态生成。
        """
        print("StoryDraftingTool (V3): 正在生成制作宝典 (当前使用占位符)...")

        # 基于 "提示词优化_1" 的示例结构
        return {
            "photography": {
                "lighting": "柔和、温暖的光线，模拟清晨的阳光，营造吉祥氛围。",
                "color_palette": "高饱和度的中国新年色板（中国红、帝王黄、莲花粉）。",
                "camera_movement": "运镜平稳、流畅，多用缓慢推近镜头。"
            },
            "art_direction": {
                "visual_style": "3D国风，角色有陶瓷娃娃的光滑质感，但背景保留年画纹理。",
                "costume_design": f"主角（{knowledge.get('figures', [{}])[0].get('figure_name', '胖娃娃')}）身穿传统红色锦缎上衣，有祥云刺绣。",
                "environment": "一个充满生机、魔幻的荷花池，水面波光粼粼。"
            },
            "casting": {
                "figure_look": f"主角'{knowledge.get('figures', [{}])[0].get('figure_name', '胖娃娃')}'约5岁，脸颊丰满红润，笑容可掬。"
            },
            "editing": {
                "pacing": "节奏舒缓、优雅。",
                "transitions": "使用'金色光斑'或'水彩晕染'效果转场。"
            },
            "audio": {
                "music": "轻松、欢快的中国民乐，以笛子和琵琶为主。",
                "sound_effects": "轻微的风铃声、水花声、清脆笑声。"
            }
        }

    def _scene_expansion_prompt(self, knowledge: Dict[str, Any], bible: Dict[str, Any], base_action: str,
                                scene_index: int) -> str:
        """构造用于丰富单个场景的提示。"""
        figure = knowledge.get('figures', [{}])[0].get('figure_name', '人物')
        meaning = knowledge.get('objects', [{}])[0].get('object_symbol', '')
        env = bible.get('art_direction', {}).get('environment', '')
        lighting = bible.get('photography', {}).get('lighting', '')
        movement = bible.get('photography', {}).get('camera_movement', '')
        style = bible.get('art_direction', {}).get('visual_style', '')
        return (
            f"你是非遗短视频导演。请在保持原意的基础上，将下面的场景描述扩展为 2-3 句具体、生动、具画面感的中文描述。"
            f"避免出现占位符，避免口语化的网络词。必须体现: 环境:{env}; 光线:{lighting}; 运镜:{movement}; 风格:{style}; 文化寓意:{meaning}。"
            f"\n原始场景描述: {base_action}\n角色: {figure}\n场序: {scene_index}\n输出只需扩展后的描述，不要添加额外说明。"
        )

    def _narrator_prompt(self, knowledge: Dict[str, Any], bible: Dict[str, Any], scene_index: int) -> str:
        figure = knowledge.get('figures', [{}])[0].get('figure_name', '我')
        meaning = knowledge.get('objects', [{}])[0].get('object_symbol', '')
        artwork = knowledge.get('artwork_name', '')
        return (
            f"你是短视频文案创作者。为第{scene_index}场生成一句至两句旁白，第一人称或第三人称均可，语气温暖、喜庆。"
            f"需自然融入作品《{artwork}》的寓意: {meaning}。最后一场可以加入一个互动问题。不要使用占位符。"
        )

    # --- 新增: 知识点汇总辅助函数 ---
    def _build_knowledge_points(self, knowledge: Dict[str, Any]) -> List[str]:
        points = []
        era = knowledge.get('era') or '清代'
        artwork = knowledge.get('artwork_name', '')
        creator = knowledge.get('creator_name') or '佚名画师'
        figures = knowledge.get('figures', [])
        objects = knowledge.get('objects', [])
        techniques = knowledge.get('techniques', [])

        # 典型元素与寓意（去重保留前若干）
        symbol_map = []
        for o in objects:
            if isinstance(o, dict):
                name = o.get('object_name')
                sym = o.get('object_symbol')
                if name and sym:
                    symbol_map.append((name, sym))
        # 去重
        uniq = []
        seen = set()
        for name, sym in symbol_map:
            key = f"{name}|{sym}"
            if key not in seen:
                uniq.append((name, sym))
                seen.add(key)
        # 选取前 6 个有代表性的寓意
        for name, sym in uniq[:6]:
            points.append(f"{name}象征{sym}")

        # 人物象征
        if figures:
            for f in figures[:2]:
                fn = f.get('figure_name')
                fs = f.get('figure_symbol')
                if fn and fs:
                    points.append(f"画中人物“{fn}”寓意{fs}")

        # 技法
        if techniques:
            points.append(f"采用技法：{'、'.join(techniques)}")

        # 时代与作者
        points.append(f"成画时代：{era}")
        if creator and creator not in ('""', '佚名'):
            points.append(f"相关画师：{creator}")

        # 去重保留顺序
        final = []
        seen2 = set()
        for p in points:
            if p not in seen2:
                final.append(p)
                seen2.add(p)
        return final

    def _call_llm_raw(self, system_content: str, user_content: str) -> str:
        if not self.llm:
            # 占位简单“伪创作”
            return ""
        try:
            return self.llm.invoke([
                ("system", system_content),
                ("user", user_content)
            ]).content.strip()
        except Exception as e:
            print(f"StoryDraftingTool: 原始 LLM 调用失败: {e}")
            return ""

    def _generate_dialogue_plan(self, knowledge: Dict[str, Any], plot: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        (重构) 使用 LLM 创作生动的多场景对话。

        该函数扮演一个“金牌剧本编剧”的角色，根据剧情大纲和知识点，
        调用大语言模型（LLM）来创作自然、符合角色身份的对话。
        它取代了之前基于循环和规则分配知识点的死板方法。

        Args:
            knowledge: 包含作品信息的知识包，如人物（figures）。
            plot: 包含故事大纲（opening, climax等）和知识点（knowledge_points）的字典。

        Returns:
            一个场景字典的列表，每个字典包含场景编号、标签和对话列表。
            如果 LLM 调用失败或返回格式无效，则返回空列表。
            e.g., [{'scene': 1, 'tag': '开端', 'dialogues': [{'speaker': '角色A', 'line': '...'}]}]
        """
        print("StoryDraftingTool (Refactored): 正在调用 LLM 创作对话...")

        # 1. 收集生成对话所需的上下文信息
        figures = [f for f in knowledge.get('figures', []) if isinstance(f, dict) and f.get('figure_name')]
        figure_names = [f.get('figure_name') for f in figures] or ['画中人']
        allowed_speakers = figure_names + ['旁白']
        knowledge_points = plot.get('knowledge_points', []) or []

        # 将剧情大纲字典转换为更易读的文本
        plot_outline = {
            '开端': plot.get('opening', ''),
            '发展': plot.get('development', ''),
            '知识讲述': plot.get('knowledge_exposition', ''),
            '高潮': plot.get('climax', ''),
            '结尾': plot.get('ending', '')
        }
        outline_text = '\n'.join([f"- {key}: {value}" for key, value in plot_outline.items() if value])

        # 2. 构建 Prompt
        system_prompt = (
            "你是一位国风非遗短视频的金牌剧本编剧。你的任务是基于给定的剧情大纲和一系列非遗知识点，创作出一段自然、生动、且符合角色身份的对话脚本。"
            "你的创作必须遵循以下规则：\n"
            "1. **角色限制**: 只能使用以下允许的角色列表进行对话: " + f" `{', '.join(allowed_speakers)}`。严禁创造任何新角色。\n"
            "2. **知识融合**: 必须将知识点巧妙地、有机地融入到角色的对话中，要听起来像是自然的交流，而不是生硬的知识宣讲或简单的复述“X象征Y”。可以通过角色的互动、情感表达或比喻来传递信息。\n"
            "3. **结构化输出**: 你的最终输出必须是一个严格的 JSON 格式的字符串，它是一个包含多个场景对象的列表。每个场景对象都应包含 'scene' (场景编号, int), 'tag' (场景标签, string), 和 'dialogues' (一个包含 {speaker, line} 对象的列表)。\n"
            "4. **格式示例**: `[{\"scene\": 1, \"tag\": \"开端\", \"dialogues\": [{\"speaker\": \"旁白\", \"line\": \"...\"}, {\"speaker\": \"角色A\", \"line\": \"...\"}]}, ...]`\n"
            "5. **简洁性**: 不要包含任何解释、注释或 ```json ``` 代码块标记，只返回纯粹的 JSON 字符串。"
        )

        user_prompt = (
            f"请根据以下信息创作剧本：\n\n"
            f"**剧情大纲:**\n{outline_text}\n\n"
            f"**需要巧妙融合的知识点 (请自然地使用，不必全部使用):**\n"
            f"- {json.dumps(knowledge_points, ensure_ascii=False)}\n\n"
            f"**允许的对话角色:**\n"
            f"- {json.dumps(allowed_speakers, ensure_ascii=False)}\n\n"
            "现在，请开始创作并返回符合要求的 JSON 字符串。"
        )

        # 3. 使用现有方法调用 LLM
        raw_response = self._call_llm_raw(system_prompt, user_prompt)

        if not raw_response:
            print('[ERROR] StoryDraftingTool: LLM 未返回任何内容，无法生成对话。')
            return []

        # 清理 LLM 可能返回的包裹标记
        if raw_response.startswith('```json'):
            raw_response = raw_response[7:].strip()
            if raw_response.endswith('```'):
                raw_response = raw_response[:-3].strip()

        # 确保只取最外层的 `[]`
        start_index = raw_response.find('[')
        end_index = raw_response.rfind(']')
        if start_index == -1 or end_index == -1:
            print(f'[ERROR] StoryDraftingTool: LLM 返回的内容中未找到有效的 JSON 列表结构。返回内容: {raw_response}')
            return []

        json_string = raw_response[start_index:end_index+1]

        # 4. 解析 JSON 并添加错误处理
        try:
            dialogue_plan = json.loads(json_string)
            # 基本的结构验证
            if not isinstance(dialogue_plan, list):
                print(f'[ERROR] StoryDraftingTool: 解析后的 JSON 不是一个列表。内容: {dialogue_plan}')
                return []

            # (可选) 进一步验证内部结构
            for scene in dialogue_plan:
                if not all(k in scene for k in ['scene', 'tag', 'dialogues']):
                     print(f'[WARN] StoryDraftingTool: 场景对象缺少必要的键: {scene}')

            print(f"StoryDraftingTool (Refactored): 成功创作并解析了 {len(dialogue_plan)} 个场景的对话。")
            return dialogue_plan
        except json.JSONDecodeError as e:
            print(f'[ERROR] StoryDraftingTool: 无法解析 LLM 返回的 JSON 字符串。错误: {e}')
            print(f'原始字符串: {json_string}')
            return []
        except Exception as e:
            print(f'[ERROR] StoryDraftingTool: 处理 LLM 响应时发生未知错误: {e}')
            return []

    # --- 修改: 生成剧情 (增加 knowledge_points) ---
    def _generate_story_plot(self, knowledge: Dict[str, Any], user_intent: List[str]) -> Dict[str, Any]:
        print("StoryDraftingTool (V4+): 正在生成核心故事剧本与知识点...")
        character = knowledge.get('figures', [{}])[0].get('figure_name', '画中人')
        artwork = knowledge.get('artwork_name', '这幅年画')
        knowledge_points = self._build_knowledge_points(knowledge)
        intent_style = "、".join(user_intent)
        system_inst = "你是非遗短视频剧情策划，输出结构化 JSON。"
        # 为避免模板变量解析，把示例 JSON 花括号双写转义
        user_prompt = f"""
    生成一个关于《{artwork}》的 5 段式剧情 (opening, development, knowledge_exposition, climax, ending)，融合风格倾向：{intent_style}。
    要求：
    - 每段 1-2 句。
    - climax 段给出一条角色核心口语化台词，写入 dialogue 字段。
    - knowledge_exposition 段聚合多元素寓意。
    - 提供 knowledge_points 数组（长度 5-10），元素为“X象征Y”或“成画时代：…”。
    - 绝对禁止引入知识包未给出的新角色或虚构元素；不要杜撰“仙子”“精灵”等。
    - 输出 JSON 键：opening, development, knowledge_exposition, climax, ending, dialogue, knowledge_points。
    示例 (结构示意)：
    {{"opening":"...","development":"...","knowledge_exposition":"...","climax":"...","ending":"...","dialogue":"一句高潮台词","knowledge_points":["鱼象征有余","荷花象征连年"]}}
    知识点候选：{'; '.join(knowledge_points[:12])}
    只输出 JSON，不要解释。
    """
        raw = self._call_llm_raw(system_inst, user_prompt)
        try:
            data = json.loads(raw)
        except Exception as e:
            print(f"StoryDraftingTool: 剧本 JSON 解析失败，使用回退模板。原因: {e}")
            data = {
                "opening": f"清晨，《{artwork}》微光闪动，{character}醒来。",
                "development": f"他循着水声靠近寓意丰富的画面。",
                "knowledge_exposition": f"他指着画中元素说出它们的寓意。",
                "climax": "吉祥的力量汇聚在光芒中。",
                "ending": "他邀请观众分享最喜欢的年画寓意。",
                "dialogue": "愿你我连年有余！",
                "knowledge_points": knowledge_points[:6]
            }
        if not data.get('knowledge_points'):
            data['knowledge_points'] = knowledge_points[:6]
        print("StoryDraftingTool (V4+): 剧本生成完成。")
        # 追加对话骨架
        data['dialogue_plan'] = self._generate_dialogue_plan(knowledge, data)
        return data

    # --- 修改: 分镜生成，增加到 5 场景并分配知识点 ---
    def _generate_final_storyboard_from_plot(self, knowledge: Dict[str, Any], template: Dict[str, Any],
                                             bible: Dict[str, Any], plot: Dict[str, Any]) -> List[Dict[str, Any]]:
        print("StoryDraftingTool (V4+): 正在生成知识融合分镜(含多角色对话)...")
        segments = [
            ("开端", plot.get('opening', '')),
            ("发展", plot.get('development', '')),
            ("知识讲述", plot.get('knowledge_exposition', '')),
            ("高潮", plot.get('climax', '')),
            ("结尾", plot.get('ending', ''))
        ]
        dialogue_plan = plot.get('dialogue_plan', [])
        kp_list = plot.get('knowledge_points', [])
        base_templates = template.get('storyboard_template', [])
        storyboard = []
        prev_action_summary = ""
        total = len(segments)
        for i, (seg_name, seg_text) in enumerate(segments):
            if i < len(base_templates):
                tpl = base_templates[i]
                camera = tpl.get('camera', '中景')
                duration = '10s'  # 统一 10s
            else:
                camera = '中景'
                duration = '10s'
            kp = kp_list[i] if i < len(kp_list) else ''
            action_prompt = f"""
    以沉浸式视觉语句（1-2句）描写该段：{seg_text}
    融入（若有）知识点：{kp}
    仅允许出现的元素：{', '.join([f.get('figure_name') for f in knowledge.get('figures', []) if isinstance(f, dict) and f.get('figure_name')] + [o.get('object_name') for o in knowledge.get('objects', []) if isinstance(o, dict) and o.get('object_name')]) or '（无）'}
    禁止添加新的角色/动物/道具/幻想元素。
    风格：{bible.get('art_direction', {}).get('visual_style')}; 环境：{bible.get('art_direction', {}).get('environment')}; 光线：{bible.get('photography', {}).get('lighting')}
    不要出现“镜头”或“画面”字样的开头；不要写对话。
    """
            action = self._call_llm_raw("你是国风分镜视觉导演。", action_prompt) or seg_text
            scene_dialogues = []
            scene_knowledge_point = ''
            if i < len(dialogue_plan):
                scene_dialogues = dialogue_plan[i].get('dialogues', [])
                scene_knowledge_point = dialogue_plan[i].get('scene_knowledge_point', '')
            # continuity
            continuity = ''
            if prev_action_summary:
                continuity = f"承接上一场景收尾：{prev_action_summary}"[:32]
            prev_action_summary = action[-20:] if action else prev_action_summary
            # 旁白兜底
            if not any(d['speaker'] == '旁白' for d in scene_dialogues):
                narrator_prompt = f"""写一句温暖旁白，含蓄衔接：{seg_name}；可暗示：{scene_knowledge_point or kp}。不超过18字。禁止引入新角色。"""
                narrator_line = self._call_llm_raw("你是诗意旁白写手。", narrator_prompt)
                if narrator_line:
                    scene_dialogues.append({"speaker": "旁白", "line": narrator_line})
            # 知识点说明兜底
            knowledge_core = (scene_knowledge_point or kp).split('象征')[0].split('寓意')[0].strip()
            if (scene_knowledge_point or kp) and knowledge_core:
                if not any(
                        (scene_knowledge_point or kp) in d.get('line', '') or knowledge_core in d.get('line', '') for d
                        in scene_dialogues):
                    scene_dialogues.append(
                        {"speaker": "旁白", "line": f"这里的元素体现——{scene_knowledge_point or kp}"})
            # 生成入/出衔接
            if i == 0:
                transition_in = "淡入，缓慢推进建立氛围"
            else:
                transition_in = "延续上镜头残留光影/动作自然接入"
            if i == total - 1:
                transition_out = "主题停留一瞬后渐隐"
            else:
                next_name = segments[i + 1][0]
                transition_out = f"为下段《{next_name}》预留运动或视线指向"
            storyboard.append({
                "scene": i + 1,
                "phase": seg_name,
                "camera": camera,
                "duration": duration,
                "action": action,
                "dialogues": scene_dialogues,
                "continuity_from_previous": continuity,
                "knowledge_point": scene_knowledge_point or kp,
                "transition_in": transition_in,
                "transition_out": transition_out
            })
        print(f"StoryDraftingTool (V4+): 分镜生成完成，共 {len(storyboard)} 场。")
        return storyboard

    def _extract_visual_tokens(self, textual_knowledge: str) -> List[str]:
        """粗提视觉相关词汇: 颜色/服饰/材质/纹理/器物等, 去重保序."""
        if not textual_knowledge:
            return []
        parts = re.split(r'[，。,、；;\n]|\s+', textual_knowledge)
        tokens = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if len(p) >= 2 and re.search(r'[\u4e00-\u9fa5]', p):  # 至少含中文
                # 排除常见功能词
                if p not in {"的", "和", "及", "与", "呈现"}:
                    tokens.append(p)
        dedup = []
        seen = set()
        for t in tokens:
            if t not in seen:
                seen.add(t)
                dedup.append(t)
        return dedup[:30]

    def _derive_action_phases(self, original_action_request: str) -> Dict[str, str]:
        """根据原始动作请求推导 4 个时间段的动作分解重点，强调微动作与物理细节。"""
        base = (original_action_request or "角色执行传统动作").strip()
        # 简单规则：尝试分解跳跃/出现/拿物/移动等关键词
        phases = {
            "0s-2s": f"起始定势/预备: {base} 的起势准备，聚力，目光方向，呼吸与衣料轻微晃动",
            "2s-5s": f"主动作加速阶段: {base} 的核心运动展开，四肢协同，服饰褶皱动态，金鱼/道具产生互动",
            "5s-8s": f"动作峰值与过渡: {base} 达到最高幅度或关键姿态，面部表情细化，缓冲过渡",
            "8s-10s": f"收势与余韵: {base} 动作落势、微调姿态、情绪停留与象征意味延续"
        }
        return phases

    def _run(self, knowledge_packet: Dict[str, Any], original_action_request: str) -> Dict[str, Any]:
        print(f"StoryDraftingTool (V10-ACTION): 收到知识包与动作请求 '{original_action_request}'，优先优化动作细节...")
        textual_knowledge = knowledge_packet.get("textual_knowledge", "一个杨柳青年画角色")
        ground_truth_url = knowledge_packet.get("ground_truth_url", "")
        # 提取视觉要素（次优先，用于风格保持）
        try:
            visual_tokens = self._extract_visual_tokens(textual_knowledge)
        except Exception:
            visual_tokens = []
        if not visual_tokens:
            visual_tokens = ["红肚兜", "圆润面庞", "细黑勾线", "饱和中国红", "金鱼纹理"]
        core_style_tokens = visual_tokens[:8]
        style_tokens_str = '、'.join(core_style_tokens)
        # 动作分解
        action_phases = self._derive_action_phases(original_action_request)

        # --- 修改后的 system_prompt：四段式结构 ---
        system_prompt = (
            "你是一位顶级的非遗视频动作提示词工程师，首要目标是：\n"
            "(1) 以电影化、精细的动态分解描写角色执行指定动作的全过程，强调身体部位协同、重心转移、速度变化、力的传导、服饰与随身物理反馈；\n"
            "(2) 次要目标：保持杨柳青年画核心视觉元素（颜色/服饰/纹理），但不得压倒动作细节。\n"
            "禁止：臆造现代/科幻/写实摄影/卡通化/霓虹/3D 渲染元素。\n"
            "你的输出必须严格遵循以下格式，不要添加任何解释：\n"
            "准备 (Preparation): [描述此阶段的动作细节]\n"
            "高潮 (Climax/Core Action): [描述核心动作高潮的细节]\n"
            "收尾 (Resolution/Follow-Through): [描述动作收尾和缓冲的细节]\n"
            "风格 (Overall Style): [描述整体视觉风格、服饰、色彩等]\n"
            "规范：每段 1-2 句，中文，避免英文、括号注释、以及以‘镜头’/‘画面’开头的句式。"
        )

        user_prompt = (
            f"【原始动作】：{original_action_request}\n"
            f"【知识事实（辅助风格，不强制逐字重复）】：{textual_knowledge}\n"
            f"【核心视觉元素（可分散，仅必要重复）】：{style_tokens_str}\n"
            f"【动作阶段参考】：\n"
            f"- 准备: {action_phases['0s-2s']}\n"
            f"- 核心推进: {action_phases['2s-5s']}\n"
            f"- 峰值与过渡: {action_phases['5s-8s']}\n"
            f"- 收尾余韵: {action_phases['8s-10s']}\n"
            "请基于这些阶段，将输出映射到‘准备/高潮/收尾/风格’四段式结构中。\n"
            "在‘准备’中强调起势与聚力与微动态；‘高潮’写主动作峰值与协调与物理反馈；‘收尾’写缓冲、余势与情绪残留；‘风格’收束整体视觉风格与核心元素。"
        )

        raw = self._call_llm_raw(system_prompt, user_prompt)
        negative_prompt = (
            "3D, 写实摄影, 真实照片, 渲染光效, 现代服饰, 西式卡通, 动漫, 科技元素, 霓虹, CG, 模糊, 低清晰度, 水印, 英文文本"
        )
        need_fallback = False
        if not raw:
            need_fallback = True
        else:
            required_tags = ["准备", "高潮", "收尾", "风格"]
            if not all(tag in raw for tag in required_tags):
                print("[WARN] 输出缺少四段标签，进入兜底补全。")
                need_fallback = True
        if need_fallback:
            act = original_action_request or "角色执行传统动作"
            raw = (
                f"准备 (Preparation): {act} 起势，重心下沉，手臂微调整，衣料与{style_tokens_str.split('、')[0]}轻幅预备晃动。\n"
                f"高潮 (Climax/Core Action): {act} 达到最大幅度，四肢协调展开，服饰延迟摆动，金鱼或随身物产生受力反馈与轨迹延伸。\n"
                f"收尾 (Resolution/Follow-Through): 动作缓冲与回收，重心回稳，衣料回摆趋平，表情逐渐收敛留寓意。\n"
                f"风格 (Overall Style): 杨柳青年画质感，平涂+勾线，高饱和国色（红/金/暖调），传统纹理与吉祥元素自然嵌入，杜绝现代失真。"
            )
        return {"master_prompt": raw, "negative_prompt": negative_prompt}
    async def _arun(self, knowledge_packet: Dict[str, Any]) -> Dict[str, Any]:
        return self._run(knowledge_packet)


# --- Test ---
if __name__ == '__main__':
    pass