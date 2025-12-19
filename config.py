import os
# config.py
# 用于存储敏感信息和配置

# Neo4j Database Configuration
NEO4J_URI = ""
NEO4J_USER = ""
NEO4J_PASSWORD = "" # <--- 请替换为你的密码
NEO4J_DATABASE = "" # 默认数据库名，如果不同请修改

# LLM API Configuration (以通义千问为例)
QWEN_API_KEY = ""  # 建议改为 os.getenv("QWEN_API_KEY", "")
# DashScope 视频生成：改为从环境变量读取，避免硬编码泄露
DASHSCOPE_API_KEY = QWEN_API_KEY # os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_VIDEO_MODEL = ""
DASHSCOPE_VIDEO_RESOLUTION = ""  # 官方示例使用大写
DASHSCOPE_VIDEO_DEFAULT_DURATION = "" # 秒，若场景未提供或无法解析时使用（已从5调整为10）
DASHSCOPE_VIDEO_NEGATIVE_PROMPT = ""  # 可按需填入统一 negative prompt
DASHSCOPE_AUDIO_URL = os.getenv("DASHSCOPE_AUDIO_URL", "")  # 可选：外部配音音频URL
# DASHSCOPE_FALLBACK_IMAGE_URL = os.getenv("DASHSCOPE_FALLBACK_IMAGE_URL", "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20250925/wpimhv/rap.png")  # 可选：备用图片URL
QWEN_BASE_URL = ""
PLANNER_LLM_MODEL = ""
GUARDIAN_LLM_MODEL = "" # (如果Guardian也使用LLM)
STORYTELLER_LLM_MODEL = "" # 为 Storyteller Agent 添加模型配置

# --- LangChain/LangSmith Tracing (推荐) ---
# 访问 https://smith.langchain.com/ 创建项目，获取API Key
# 这对于调试 LangGraph 非常有帮助

# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = "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" # <--- 替换
#os.environ["LANGCHAIN_PROJECT"] = "Non-legacy Story Director" # <--- 可选

# File Paths
# NARRATIVE_TEMPLATE_FILE = ''
#VISUAL_ANALYSIS_FILE = ''

# 本地图像兜底配置 (可选)。如果未设置环境变量则使用这些默认值。
# DEFAULT_IMAGE_DIR = os.getenv("DASHSCOPE_IMAGE_DIR", "")  # 请确保目录存在
# DEFAULT_IMAGE_FILE = os.getenv("DASHSCOPE_IMAGE_LOCAL_PATH", "")  # 若指定单文件优先

VIDEO_TOTAL_DURATION = 0  # 固定：目标总视频时长（秒）

# --- 新增：多镜头合一视频生成开关 ---
# True: 将所有分镜合并为一个 prompt，只生成一个总视频
# False: 每个分镜单独生成一个视频片段
COMBINE_SCENES_INTO_ONE_VIDEO = True

# 是否允许直接使用本地图片路径（固定开启）
DASHSCOPE_ALLOW_LOCAL_PATH = True
# 是否将本地图片自动转换为 data URI 传递（固定关闭）
DASHSCOPE_LOCAL_AS_DATA_URI = False
