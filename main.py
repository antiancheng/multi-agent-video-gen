# main.py
# (支持命令行参数传入用户请求；保持原交互方式)

# 新增: 禁用 LangSmith 追踪，避免网络连接错误干扰主流程
import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from planner_agent import PlannerAgent
# 移除未使用的导入 guardian_agent / json / sys
# 新增: 下载视频所需库
try:
    import requests
except Exception:
    requests = None

# 新增：切换工作目录到脚本所在目录，确保相对路径资源(如 narrative_template_library.json)可被加载
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 新增：视频下载辅助函数
def print_human_readable_package(package: dict):
    """以更适合人类阅读的格式打印最终的提示词包 (纯提示词模式)。"""
    if not package:
        print("错误：生成的提示词包为空。")
        return
    print("\n# === Master Prompt (动作优先) === #")
    print(package.get("master_prompt", "无"))
    print("\n# === Negative Prompt (过滤) === #")
    print(package.get("negative_prompt", "无"))
    # 简洁展示知识来源（若存在）
    src = package.get("_source_knowledge_packet", {}) or {}
    if src:
        print("\n# === 来源知识概要 === #")
        summary_items = []
        for k in ["artwork_name", "creator_name", "era", "creation_region"]:
            v = src.get(k)
            if v:
                summary_items.append(f"{k}: {v}")
        objects = src.get("objects") or []
        if objects:
            first_obj = next((o for o in objects if isinstance(o, dict)), None)
            if first_obj:
                on = first_obj.get("object_name"); osym = first_obj.get("object_symbol")
                if on or osym:
                    summary_items.append(f"object: {on or ''}/{osym or ''}".strip('/'))
        figures = src.get("figures") or []
        if figures:
            first_fig = next((f for f in figures if isinstance(f, dict)), None)
            if first_fig:
                fn = first_fig.get("figure_name"); fs = first_fig.get("figure_symbol")
                if fn or fs:
                    summary_items.append(f"figure: {fn or ''}/{fs or ''}".strip('/'))
        if summary_items:
            for line in summary_items:
                print("-", line)
        else:
            print("(无可用概要信息)")
    print("\n# === 模式 === #")
    print(package.get("mode", "UNKNOWN"))
    print("说明:", package.get("notes", "无"))



if __name__ == "__main__":
    print("欢迎使用非遗故事智能导演系统（全自动版）！")

    user_input = input("请输入你的需求（例如：生成一个‘抱鱼娃娃从窗户跳出来’的动作）：").strip()

    if user_input:
        planner = None
        try:
            planner = PlannerAgent()
            result_package = planner.run(user_input)

            print("\n\n--- 最终输出结果 ---")
            if not isinstance(result_package, dict):
                print("错误：工作流返回了非字典类型结果。")
            elif "error" in result_package:
                print(f"\n错误：{result_package.get('error')}\n详情：{result_package.get('details','无')}")
            else:
                print_human_readable_package(result_package)
            print("\n--------------------")

        except Exception as e:
            print(f"\n程序运行过程中发生意外错误: {e}")
        finally:
            if planner:
                planner.close_connections()
    else:
        print("输入为空，程序退出。")
