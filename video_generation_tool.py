# video_generation_tool.py
"""通义万相 图生视频 生成工具 (P4 计划)"""
import time
import json
import config
import os
from typing import Type, Dict, Any
import inspect

# 依赖：dashscope （需: pip install dashscope）
try:
    import dashscope
    from dashscope import VideoSynthesis
    DASHSCOPE_AVAILABLE = True
except Exception as e:
    DASHSCOPE_AVAILABLE = False
    _dashscope_import_error = e

# LangChain BaseTool
try:
    from langchain_core.tools import BaseTool
except Exception:
    # 简单占位，避免导入失败时整个系统不可用
    class BaseTool:
        def __init__(self, **kwargs):
            pass

from pydantic import BaseModel, Field


from typing import Type, Dict, Any, Optional # <--- 确保导入 Optional

class VideoGenerationInput(BaseModel):
    """用于视频生成工具的输入模型"""
    base_image_url: Optional[str] = Field(default=None, description="（可选）基础年画或参考图片URL。若为None，则启用T2V模式。")
    prompt: str = Field(description="融合制作宝典+场景描述的提示词")
    scene_id: int = Field(description="场景编号")
    duration: int = Field(description="目标视频秒数(1-30范围)")


class VideoGenerationTool(BaseTool):
    """一个 LangChain 工具，负责调用“通义万相”图生视频API并轮询结果。"""
    name: str = "video_generation_tool"
    description: str = "根据一张基础图片和一个详细提示词，生成一个短视频片段。"
    args_schema: Type[BaseModel] = VideoGenerationInput
    api_key: str = ""  # 声明为 pydantic 字段，避免动态赋值报错
    base_http_api_url: str = ""  # 可选：记录实际使用的 base url
    enabled: bool = False  # Pydantic 需要预先声明，避免运行时赋值出错

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 使用已声明字段
        self.api_key = config.DASHSCOPE_API_KEY
        self.enabled = False
        self._call_param_names = []  # 缓存 VideoSynthesis.call 的签名参数名
        if not self.api_key:
            print("[WARN] 未检测到 DASHSCOPE_API_KEY 环境变量，视频生成将使用占位。")
        elif not DASHSCOPE_AVAILABLE:
            print(f"[WARN] dashscope 库未安装或导入失败:{_dashscope_import_error}，使用占位。")
        else:
            try:
                dashscope.api_key = self.api_key
                self.base_http_api_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/api/v1")
                dashscope.base_http_api_url = self.base_http_api_url
                self.enabled = True
                print(f"VideoGenerationTool: 已启用真实视频生成 (base_http_api_url={self.base_http_api_url}).")
                # 反射签名
                try:
                    sig = inspect.signature(VideoSynthesis.call)
                    self._call_param_names = list(sig.parameters.keys())
                    print(f"[DEBUG] VideoSynthesis.call 签名参数: {self._call_param_names}")
                    ver = getattr(dashscope, '__version__', '未知版本')
                    print(f"[DEBUG] dashscope 版本: {ver}")
                except Exception as se:
                    print(f"[WARN] 反射 VideoSynthesis.call 签名失败: {se}")
            except Exception as e:
                print(f"[WARN] 初始化 DashScope 失败: {e}，使用占位模式。")
                self.enabled = False

    def _normalize_duration(self, duration: int) -> int:
        try:
            d = int(duration)
            if d < 1: d = 1
            if d > 30: d = 30
            return d
        except Exception:
            return getattr(config, 'DASHSCOPE_VIDEO_DEFAULT_DURATION', 5)

    def _prepare_image_url(self, url: Optional[str]) -> Optional[str]:
        if not url:
            # 不再返回备用 URL，而是返回 None
            return None
        return url

    def _compress_prompt(self, prompt: str) -> str:
        # 将 JSON 风格的大段宝典压缩成标签串，并显式保留多角色对话与知识要点。
        if not prompt:
            return ""
        style_tags = ["3D国风", "陶瓷质感角色", "柔和晨光", "国风年画纹理", "吉祥与富足", "缓慢推近镜头"]
        core = " | ".join(style_tags)
        # 标记集合
        knowledge_marker = "[场景知识要点]:"
        knowledge_list_marker = "[扩展知识要点列表]:"
        continuity_marker = "[跨场景衔接提示]:"
        action_marker_variants = ["[画面动作描写]:", "[画面动作描述]:"]
        dialogue_ref_marker = "[多角色对话参考]:"
        dialogue_sub_marker = "[字幕对白]:"
        # 提取知识要点
        knowledge_excerpt = ""
        if knowledge_marker in prompt:
            after = prompt.split(knowledge_marker, 1)[1]
            knowledge_excerpt = after.splitlines()[0].strip()[:60]
        # 扩展知识列表
        extended_excerpt = ""
        if knowledge_list_marker in prompt:
            after = prompt.split(knowledge_list_marker, 1)[1]
            extended_excerpt = after.splitlines()[0].strip()[:120]
        # 跨场景衔接
        continuity_excerpt = ""
        if continuity_marker in prompt:
            after = prompt.split(continuity_marker, 1)[1]
            continuity_excerpt = after.splitlines()[0].strip()[:60]
        # 画面动作
        action_excerpt = ""
        for mk in action_marker_variants:
            if mk in prompt:
                seg = prompt.split(mk, 1)[1]
                action_excerpt = seg.strip().splitlines()[0][:160]
                break
        if not action_excerpt:
            action_excerpt = prompt[:160].strip()
        # 对话：优先字幕对白，其次多角色参考
        dialogue_lines_kept = []
        def _extract_dialogue(after_block: str) -> list:
            raw_lines = [l.strip() for l in after_block.splitlines() if l.strip()]
            cand = [l for l in raw_lines if ':' in l and len(l) < 70]
            return cand
        if dialogue_sub_marker in prompt:
            after = prompt.split(dialogue_sub_marker, 1)[1]
            dialogue_lines_kept = _extract_dialogue(after)[:6]
        elif dialogue_ref_marker in prompt:
            after = prompt.split(dialogue_ref_marker, 1)[1]
            dialogue_lines_kept = _extract_dialogue(after)[:6]
        # 合并知识
        knowledge_total = knowledge_excerpt or ''
        if extended_excerpt:
            if knowledge_total:
                knowledge_total = f"{knowledge_total}; {extended_excerpt[:120]}"
            else:
                knowledge_total = extended_excerpt[:160]
        compressed = (
            f"风格标签:{core}\n"
            f"动作:{action_excerpt}\n"
            f"知识:{knowledge_total or '无'}\n"
            f"衔接:{continuity_excerpt or '平滑'}\n"
            f"对话:{' / '.join(dialogue_lines_kept) if dialogue_lines_kept else '无'}\n"
            f"基调:喜庆 温暖 吉祥 国风 写意 文化保真"
        )
        return compressed[:900]

    def _poll_task(self, task_id: str, scene_id: int, model: str, resolution: str, norm_duration: int) -> Dict[str, Any]:
        """轮询任务直到成功或超时。"""
        poll_interval = int(os.getenv("DASHSCOPE_POLL_INTERVAL", "5"))
        max_seconds = int(os.getenv("DASHSCOPE_MAX_POLL_SECONDS", "180"))
        deadline = time.time() + max_seconds
        print(f"VideoGenerationTool: 场景 {scene_id} 开始轮询任务 task_id={task_id} (interval={poll_interval}s, max={max_seconds}s)")
        last_status = None
        while time.time() < deadline:
            try:
                rsp = VideoSynthesis.fetch(task_id)
            except Exception as e:
                print(f"VideoGenerationTool: 场景 {scene_id} 轮询异常: {e}")
                time.sleep(poll_interval)
                continue
            status_code = getattr(rsp, 'status_code', None)
            if status_code != 200:
                msg = getattr(rsp, 'message', '未知错误')
                print(f"VideoGenerationTool: 场景 {scene_id} 轮询失败 status_code={status_code} msg={msg}")
                time.sleep(poll_interval)
                continue
            output = getattr(rsp, 'output', None)
            task_status = getattr(output, 'status', None) or getattr(output, 'task_status', None)
            video_url = getattr(output, 'video_url', None)
            if task_status and task_status != last_status:
                print(f"VideoGenerationTool: 场景 {scene_id} 任务状态 -> {task_status}")
                last_status = task_status
            if video_url:
                print(f"VideoGenerationTool: 场景 {scene_id} 轮询成功获得视频 URL -> {video_url}")
                return {"scene_id": scene_id, "video_url": video_url, "status": "SUCCEEDED", "model": model, "resolution": resolution, "duration": norm_duration}
            if task_status in {"FAILED", "ERROR"}:
                err = getattr(output, 'error_msg', None) or getattr(output, 'message', None) or '未知错误'
                print(f"VideoGenerationTool: 场景 {scene_id} 任务失败: {err}")
                return {"scene_id": scene_id, "status": "FAILED", "error": err, "model": model, "resolution": resolution, "duration": norm_duration}
            time.sleep(poll_interval)
        print(f"VideoGenerationTool: 场景 {scene_id} 轮询超时 (task_id={task_id})")
        return {"scene_id": scene_id, "status": "FAILED", "error": f"轮询超时({max_seconds}s)未获得视频", "model": model, "resolution": resolution, "duration": norm_duration}

    def _try_call_variants(self, scene_id: int, model: str, compressed_prompt: str, img_url: Optional[str], audio_url: str | None,
                           resolution: str, norm_duration: int, negative_prompt: str) -> tuple[Any, str]:
        """根据真实签名动态选择参数名，仅尝试存在的名称，减少无效异常。"""
        if not self._call_param_names:
            return None, "未知 VideoSynthesis.call 签名，无法调用"
        candidate_image_param_names = [n for n in ['image_url','img_url','input_image_urls','image_urls'] if n in self._call_param_names]
        if not candidate_image_param_names:
            return None, f"签名中未发现任何已知图片参数名: {self._call_param_names}"
        last_err = ''
        for i, pname in enumerate(candidate_image_param_names, 1):
            print(f"[DEBUG] 尝试图片参数 '{pname}' 变体 {i}/{len(candidate_image_param_names)}")
            payload = {
                'model': model,
                'prompt': compressed_prompt,
                'audio_url': audio_url,
                'resolution': resolution,
                'duration': norm_duration,
                'prompt_extend': True,
                'watermark': False,
                'negative_prompt': negative_prompt,
                'seed': scene_id * 12345 + 7
            }
            # 根据参数名填值
            # --- 新增：仅在 img_url 存在时才添加图像参数 ---
            if img_url:
                if pname in ('input_image_urls','image_urls'):
                    payload[pname] = [img_url]
                else:
                    payload[pname] = img_url
            # --- 修改结束 ---
            # 删除不存在的字段
            for k in list(payload.keys()):
                if k not in self._call_param_names:
                    payload.pop(k)
            try:
                rsp = VideoSynthesis.call(**payload)
                return rsp, ''
            except Exception as ce:
                last_err = f"调用异常({pname}): {ce}"
                print(f"[DEBUG] 变体 '{pname}' 调用失败: {last_err}")
                continue
        return None, last_err

    def _run(self, prompt: str, scene_id: int, duration: int, base_image_url: Optional[str] = None) -> Dict[str, Any]:
        print(f"VideoGenerationTool: 场景 {scene_id} 生成任务开始 (请求时长={duration})")
        model_primary = getattr(config, 'DASHSCOPE_VIDEO_MODEL', 'wan2.5-i2v-preview')
        model_fallbacks = []
        for m in [model_primary, 'wan2.5-i2v', 'wanx-i2v-preview', 'wanx-video-v1']:
            if m not in model_fallbacks:
                model_fallbacks.append(m)
        resolution_primary = getattr(config, 'DASHSCOPE_VIDEO_RESOLUTION', '480P')
        resolution_variants = []
        for r in {resolution_primary, resolution_primary.lower(), resolution_primary.upper()}:
            if r:
                resolution_variants.append(r)
        negative_prompt = getattr(config, 'DASHSCOPE_VIDEO_NEGATIVE_PROMPT', '')
        audio_url = getattr(config, 'DASHSCOPE_AUDIO_URL', '') or None
        norm_duration = self._normalize_duration(duration)
        if norm_duration != duration:
            print(f"[DEBUG] 归一化后时长={norm_duration}s (原始请求={duration}s) → 已限幅。")
        if norm_duration != 10:
            print(f"[WARN] 期望生成 10s 视频，但归一化后为 {norm_duration}s，请检查传入 duration 或默认配置。")
        # 运行时读取环境变量允许动态覆盖（env 优先）
        allow_local = os.getenv("DASHSCOPE_ALLOW_LOCAL_PATH", "1" if getattr(config, 'DASHSCOPE_ALLOW_LOCAL_PATH', False) else "0") == "1"
        local_as_data_uri = os.getenv("DASHSCOPE_LOCAL_AS_DATA_URI", "1" if getattr(config, 'DASHSCOPE_LOCAL_AS_DATA_URI', False) else "0") == "1"
        print(f"[DEBUG] 本地图片开关 allow_local={allow_local} (config={getattr(config,'DASHSCOPE_ALLOW_LOCAL_PATH',False)}) data_uri={local_as_data_uri} (config={getattr(config,'DASHSCOPE_LOCAL_AS_DATA_URI',False)})")
        img_url_raw = self._prepare_image_url(base_image_url)
        used_local_file = False
        # --- 判定是否进入纯文本 T2V 模式 ---
        is_t2v = img_url_raw is None
        if is_t2v:
            print("[INFO] 未提供基础图片，进入纯文本 T2V 模式。")
            img_url = None
        else:
            if img_url_raw and os.path.exists(img_url_raw):
                used_local_file = True
                if allow_local:
                    if local_as_data_uri:
                        try:
                            import mimetypes, base64
                            mime, _ = mimetypes.guess_type(img_url_raw)
                            if not mime: mime = 'image/png'
                            with open(img_url_raw, 'rb') as f:
                                b64 = base64.b64encode(f.read()).decode('utf-8')
                            img_url = f"data:{mime};base64,{b64}"
                            print(f"[DEBUG] 本地文件已转换为 data URI (长度={len(img_url)}) -> {img_url_raw}")
                        except Exception as ce:
                            print(f"[WARN] 本地文件转换 data URI 失败，回退为路径: {ce}")
                            img_url = img_url_raw
                    else:
                        img_url = img_url_raw
                        print(f"[DEBUG] 使用本地文件路径作为图片输入: {img_url}")
                else:
                    print(f"[DEBUG] 检测到本地文件路径: {img_url_raw} 但 allow_local=False，改用备用远程 URL")
                    img_url = getattr(config, 'DASHSCOPE_FALLBACK_IMAGE_URL', img_url_raw)
            else:
                img_url = img_url_raw
            if img_url and img_url.startswith('data:') and not allow_local:
                print(f"[WARN] data URI 在未启用本地开关条件下被替换为备用 URL")
                img_url = getattr(config, 'DASHSCOPE_FALLBACK_IMAGE_URL', img_url)
        # 安全日志输出图片输入
        safe_img_log = img_url if img_url else 'T2V-Mode(No Image)'
        # 避免对 None 做切片与 len 调用
        if isinstance(safe_img_log, str) and len(safe_img_log) > 180:
            safe_img_log_display = safe_img_log[:180] + '...'
        else:
            safe_img_log_display = safe_img_log
        print(f"[DEBUG] 最终用于调用的图片输入: {safe_img_log_display}")
        # 对白占位
        if '[字幕对白]:\n无' in prompt:
            prompt += "\n[字幕对白补充]:\n角色A: 祝福连连承古意。\n角色B: 年年有余映吉祥。"
            print("[DEBUG] 检测到无对白，已添加占位对白。")
        compressed_prompt = self._compress_prompt(prompt)
        print(f"[DEBUG] prompt 压缩后长度: {len(compressed_prompt)}")
        if not self.enabled:
            placeholder_url = f"https://placeholder.local/video_scene_{scene_id}.mp4"
            print(f"[FALLBACK] DashScope 不可用，返回占位: {placeholder_url}")
            return {"scene_id": scene_id, "video_url": placeholder_url, "status": "PLACEHOLDER", "model": model_primary, "resolution": resolution_primary, "duration": norm_duration}
        last_error = None
        # --- 纯文本 T2V 直接调用路径 ---
        if is_t2v:
            print("[DEBUG] 进入 T2V 直接调用路径，不遍历图片参数变体。")
            try:
                # 构造最小化 payload (过滤掉签名中不存在的键)
                base_payload = {
                    'model': model_primary,
                    'prompt': compressed_prompt,
                    'resolution': resolution_primary,
                    'duration': norm_duration,
                    'negative_prompt': negative_prompt,
                    'prompt_extend': True,
                    'watermark': False,
                    'seed': scene_id * 12345 + 7,
                    'audio_url': audio_url
                }
                if not self._call_param_names:
                    print("[WARN] 未能获取签名参数列表，直接尝试全部键。")
                    rsp = VideoSynthesis.call(**base_payload)
                else:
                    filtered_payload = {k: v for k, v in base_payload.items() if k in self._call_param_names}
                    rsp = VideoSynthesis.call(**filtered_payload)
                status_code = getattr(rsp, 'status_code', None)
                if status_code != 200:
                    msg = getattr(rsp, 'message', None) or '非200响应'
                    print(f"[ERROR] T2V 初始调用失败: {msg}")
                else:
                    output = getattr(rsp, 'output', None)
                    video_url = getattr(output, 'video_url', None)
                    task_id = getattr(output, 'task_id', None) or getattr(output, 'id', None)
                    if video_url:
                        print(f"VideoGenerationTool: 场景 {scene_id} T2V 成功 -> {video_url}")
                        return {"scene_id": scene_id, "video_url": video_url, "status": "SUCCEEDED", "model": model_primary, "resolution": resolution_primary, "duration": norm_duration}
                    if task_id:
                        print(f"VideoGenerationTool: 场景 {scene_id} 捕获 T2V task_id={task_id} 进入轮询。")
                        return self._poll_task(task_id, scene_id, model_primary, resolution_primary, norm_duration)
                    last_error = "T2V 模式下 API 未返回 video_url 或 task_id"
            except Exception as e:
                last_error = f"T2V 调用异常: {e}"
                print(f"[ERROR] {last_error}")
        else:
            # --- I2V 图片循环尝试路径 ---
            for mi, model in enumerate(model_fallbacks, 1):
                for ri, res in enumerate(resolution_variants, 1):
                    print(f"[DEBUG] 尝试模型 {mi}/{len(model_fallbacks)} -> {model} 分辨率变体 {ri}/{len(resolution_variants)} -> {res}")
                    rsp, variant_err = self._try_call_variants(scene_id, model, compressed_prompt, img_url, audio_url, res, norm_duration, negative_prompt)
                    if rsp is None:
                        last_error = variant_err
                        print(f"[DEBUG] 图片参数变体调用失败(模型={model}, 分辨率={res}): {variant_err}")
                        continue
                    status_code = getattr(rsp, 'status_code', None)
                    if status_code != 200:
                        msg = getattr(rsp, 'message', None)
                        output = getattr(rsp, 'output', None)
                        extra_msg = getattr(output, 'message', None) if output else None
                        last_error = msg or extra_msg or f"非200响应(代码={status_code})"
                        print(f"VideoGenerationTool: 场景 {scene_id} 调用失败(model={model}, res={res}): {last_error}")
                        continue
                    output = getattr(rsp, 'output', None)
                    if not output:
                        last_error = "响应无 output 字段"
                        print(f"[DEBUG] 无 output (model={model}, res={res})，继续下一分辨率/模型。")
                        continue
                    video_url = getattr(output, 'video_url', None)
                    task_id = getattr(output, 'task_id', None) or getattr(output, 'id', None)
                    if video_url:
                        print(f"VideoGenerationTool: 场景 {scene_id} 成功 -> {video_url} (model={model}, res={res}, duration={norm_duration}s)")
                        return {"scene_id": scene_id, "video_url": video_url, "status": "SUCCEEDED", "model": model, "resolution": res, "duration": norm_duration}
                    if task_id:
                        print(f"VideoGenerationTool: 场景 {scene_id} 捕获 task_id={task_id} 进入轮询。(model={model}, res={res})")
                        return self._poll_task(task_id, scene_id, model, res, norm_duration)
                    last_error = "API未返回video_url且无task_id"
                    print(f"[DEBUG] 未获得 video_url/task_id (model={model}, res={res})，继续。")
        # 简化提示词兜底（T2V 或 I2V 均可复用）
        simple_prompt = "动作:" + prompt[:160].replace('\n', ' ') + "\n对话:文化寓意,吉祥祝福,年成圆满"
        simple_compressed = self._compress_prompt(simple_prompt)
        print(f"[INFO] 主调用失败，使用简化提示词再次尝试 (长度={len(simple_compressed)})")
        try:
            base_payload2 = {
                'model': model_primary,
                'prompt': simple_compressed,
                'resolution': resolution_primary,
                'duration': norm_duration,
                'negative_prompt': negative_prompt,
                'prompt_extend': True,
                'watermark': False,
                'seed': scene_id * 9876 + 3,
                'audio_url': audio_url
            }
            if self._call_param_names:
                base_payload2 = {k: v for k, v in base_payload2.items() if k in self._call_param_names}
            rsp2 = VideoSynthesis.call(**base_payload2)
            status_code2 = getattr(rsp2, 'status_code', None)
            if status_code2 == 200:
                output2 = getattr(rsp2, 'output', None)
                video_url2 = getattr(output2, 'video_url', None)
                task_id2 = getattr(output2, 'task_id', None) or getattr(output2, 'id', None)
                if video_url2:
                    print(f"VideoGenerationTool: 场景 {scene_id} 简化提示成功 -> {video_url2}")
                    return {"scene_id": scene_id, "video_url": video_url2, "status": "SUCCEEDED", "model": model_primary, "resolution": resolution_primary, "duration": norm_duration}
                if task_id2:
                    print(f"VideoGenerationTool: 场景 {scene_id} 简化提示捕获 task_id={task_id2}")
                    return self._poll_task(task_id2, scene_id, model_primary, resolution_primary, norm_duration)
                last_error = "简化提示：API未返回 video_url 或 task_id"
            else:
                last_error = getattr(rsp2, 'message', None) or f"简化提示非200响应({status_code2})"
        except Exception as e:
            last_error = f"简化提示调用异常: {e}"
            print(f"[ERROR] {last_error}")
        print(f"VideoGenerationTool: 场景 {scene_id} 最终失败: {last_error}")
        return {"scene_id": scene_id, "status": "FAILED", "error": last_error or '未知错误', "model": model_primary, "resolution": resolution_primary, "duration": norm_duration}

    async def _arun(self, base_image_url: str, prompt: str, scene_id: int, duration: int) -> Dict[str, Any]:
        return self._run(base_image_url, prompt, scene_id, duration)


if __name__ == '__main__':
    # 简单自测（需安装 dashscope 且具备有效 Key）
    print("[VideoGenerationTool] 自测占位：请在实际环境安装 dashscope 并提供真实 image_url。")
