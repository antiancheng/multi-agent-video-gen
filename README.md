# Multi-Agent Video Generation Framework (多智能体视频生成框架)

这是一个基于多智能体协同（Multi-Agent Collaboration）的视频生成框架代码实现。

该系统通过规划、知识检索、叙事构建和质量守护四个智能体的协作，旨在解决通用视频生成模型在特定领域（如非遗文化）生成中的痛点。

## 核心组件文件说明

* **`main.py`**: 整个多智能体系统的启动入口和主流程控制脚本。
* **`planner_agent.py` (规划智能体)**: 负责接收用户输入，进行意图识别和任务规划，锚定核心实体。
* **`knowledge_agent.py` (知识智能体)**: 负责对接外部知识库（如知识图谱），检索精准的领域知识。
* **`storyteller_agent.py` (叙事智能体)**: 将抽象知识转化为结构化的视频生成提示词（如四段式结构）。
* **`guardian_agent.py` (守护智能体)**: 负责对生成内容进行质量把关和反馈修正。
* **`video_generation_tool.py`**: 视频生成模型的调用接口封装。
* **`config.py`**: 项目的配置文件（API Key配置等）。

## 环境安装

1. 克隆本项目到本地：
```bash
git clone [https://github.com/你的用户名/你的仓库名.git](https://github.com/你的用户名/你的仓库名.git)
cd 你的仓库名
