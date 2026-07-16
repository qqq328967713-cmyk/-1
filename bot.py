#!/usr/bin/env python3
import os
import sys
import io
import re
import time
import asyncio
import base64
import logging
import json
import httpx
from PIL import Image
from dotenv import load_dotenv
from openai import AsyncOpenAI
from telegram import Update, InputFile
from telegram.constants import ParseMode, ChatType
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

load_dotenv()

# ============================================
# 基础配置
# ============================================
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
API_KEY = os.getenv("API_KEY", "")
BASE_URL = os.getenv("BASE_URL", "https://yunwu.ai/v1")
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "20"))
ALLOWED_USERS = os.getenv("ALLOWED_USERS", "")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a helpful assistant.")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")

# ============================================
# 多模型配置
# ============================================
MODEL_MAP = {
    "NB": {"id": "gpt-image-2", "type": "image"},
    "A": {"id": "gemini-3.1-pro-preview", "type": "chat"},
    "ZZ": {"id": "claude-sonnet-5", "type": "chat"},
    "SP": {"id": "kling-3.0-turbo", "type": "video"},
}

log = logging.getLogger(__name__)
client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

histories = {}
user_models = {}
_bot_username = None


def allowed(uid):
    if not ALLOWED_USERS:
        return True
    ids = [int(x.strip()) for x in ALLOWED_USERS.split(",") if x.strip()]
    return uid in ids


def get_hist(uid):
    if uid not in histories:
        histories[uid] = []
    return histories[uid]


def should_reply(update: Update) -> bool:
    msg = update.message
    if msg.chat.type == ChatType.PRIVATE:
        return True
    if msg.reply_to_message and msg.reply_to_message.from_user:
        if msg.reply_to_message.from_user.id == int(BOT_TOKEN.split(":")[0]):
            return True
    text = msg.text or msg.caption or ""
    if _bot_username and f"@{_bot_username}" in text:
        return True
    return False


def strip_mention(text: str) -> str:
    if _bot_username:
        text = text.replace(f"@{_bot_username}", "").strip()
    return text


def detect_model_from_text(text: str) -> tuple:
    text = text.strip()
    if not text:
        return None, text

    for prefix, info in MODEL_MAP.items():
        model_id = info["id"]
        pattern = rf"^{re.escape(prefix)}[\s:：\n]+"
        match = re.match(pattern, text, re.IGNORECASE)
        if match:
            clean_text = text[match.end():].strip()
            return model_id, clean_text

    for prefix, info in MODEL_MAP.items():
        model_id = info["id"]
        if text.lower().startswith(prefix.lower()):
            remaining = text[len(prefix):]
            if not remaining or not remaining[0].isalpha():
                clean_text = remaining.strip()
                return model_id, clean_text

    return None, text


def escape_md(text: str) -> str:
    out = text
    blocks = []
    def save_block(m):
        blocks.append(m.group(0))
        return f"\x00BLOCK{len(blocks)-1}\x00"
    out = re.sub(r"```[\s\S]*?```", save_block, out)
    inlines = []
    def save_inline(m):
        inlines.append(m.group(0))
        return f"\x00INLINE{len(inlines)-1}\x00"
    out = re.sub(r"`[^`]+`", save_inline, out)
    for ch in r"\_*[]()~>#+-=|{}.!":
        out = out.replace(ch, f"\\{ch}")
    out = re.sub(r"\\\*\\\*(.*?)\\\*\\\*", r"*\1*", out)
    out = re.sub(r"\\_([^\\_]+)\\_", r"_\1_", out)
    for i, block in enumerate(blocks):
        out = out.replace(f"\x00BLOCK{i}\x00", block)
    for i, inline in enumerate(inlines):
        out = out.replace(f"\x00INLINE{i}\x00", inline)
    return out


async def send_long(msg, text):
    chunks = []
    while text:
        if len(text) <= 4096:
            chunks.append(text)
            break
        cut = text[:4096].rfind("\n")
        if cut < 100:
            cut = 4096
        chunks.append(text[:cut])
        text = text[cut:].lstrip("\n")
    first = chunks[0]
    try:
        await msg.edit_text(first, parse_mode=ParseMode.MARKDOWN_V2)
    except Exception:
        try:
            await msg.edit_text(first)
        except Exception:
            pass
    for chunk in chunks[1:]:
        try:
            await msg.reply_text(chunk, parse_mode=ParseMode.MARKDOWN_V2)
        except Exception:
            await msg.reply_text(chunk)


async def generate_image(prompt: str) -> str:
    """文生图：只输入文字，生成新图片"""
    gen_url = f"{BASE_URL}/images/generations"
    payload = {
        "model": "gpt-image-2",
        "prompt": prompt,
        "n": 1
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        async with httpx.AsyncClient(timeout=60.0) as http:
            resp = await http.post(gen_url, json=payload, headers=headers)
            data = resp.json()
        if resp.status_code != 200:
            return f"Image generation error: {data.get('error', {}).get('message', resp.text)}"
        if "data" in data and len(data["data"]) > 0:
            item = data["data"][0]
            if "url" in item and item["url"]:
                return item["url"]
            if "b64_json" in item and item["b64_json"]:
                return f"data:image/png;base64,{item['b64_json']}"
        if "url" in data:
            return data["url"]
        if "output" in data:
            output = data["output"][0] if isinstance(data["output"], list) else data["output"]
            if isinstance(output, str) and (output.startswith("http") or output.startswith("data:")):
                return output
        return f"Unexpected response: {json.dumps(data)[:200]}"
    except Exception as e:
        return f"Image generation failed: {str(e)}"


async def generate_image_edit(prompt: str, image_base64: str) -> str:
    """图生图：输入图片+文字，修改/修复图片"""
    gen_url = f"{BASE_URL}/images/edits"

    image_bytes = base64.b64decode(image_base64)

    files = {
        "image": ("image.png", io.BytesIO(image_bytes), "image/png")
    }
    data = {
        "prompt": prompt,
        "model": "gpt-image-2",
        "n": "1"
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as http:
            resp = await http.post(gen_url, files=files, data=data, headers=headers)
            data = resp.json()
        if resp.status_code != 200:
            return f"Image edit error: {data.get('error', {}).get('message', resp.text)}"
        if "data" in data and len(data["data"]) > 0:
            item = data["data"][0]
            if "url" in item and item["url"]:
                return item["url"]
            if "b64_json" in item and item["b64_json"]:
                return f"data:image/png;base64,{item['b64_json']}"
        if "url" in data:
            return data["url"]
        if "output" in data:
            output = data["output"][0] if isinstance(data["output"], list) else data["output"]
            if isinstance(output, str) and (output.startswith("http") or output.startswith("data:")):
                return output
        return f"Unexpected response: {json.dumps(data)[:200]}"
    except Exception as e:
        return f"Image edit failed: {str(e)}"


# ============================================
# Kling 视频：异步任务轮询 (修复版)
# ============================================
async def poll_kling_task(task_id: str, task_type: str = "text-to-video", max_wait: int = 300, interval: int = 10) -> dict:
    """
    轮询 Kling 任务状态，直到成功、失败或超时。
    
    yunwu.ai 的查询接口格式：
    GET https://yunwu.ai/kling/{task_type}/kling-3.0-turbo/{task_id}
    """
    status_url = f"https://yunwu.ai/kling/{task_type}/kling-3.0-turbo/{task_id}"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    elapsed = 0
    
    async with httpx.AsyncClient(timeout=30.0) as http:
        while elapsed < max_wait:
            try:
                resp = await http.get(status_url, headers=headers)
                data = resp.json()
            except Exception as e:
                log.error("poll: 请求失败: %s", e)
                await asyncio.sleep(interval)
                elapsed += interval
                continue

            log.info("kling task %s response: %s", task_id, json.dumps(data)[:500])

            # 检查 API 返回码
            if data.get("code") != 0:
                return {"error": data.get("message", "API error")}

            task_data = data.get("data", {})
            
            # yunwu.ai 返回的状态字段是 "status"，不是 "task_status"
            status = task_data.get("status", "").lower()
            
            # 可能的状态: submitted, processing, succeed/success, failed
            if status in ("succeed", "success", "completed"):
                return task_data
            if status == "failed":
                return {"error": task_data.get("message", "Task failed")}
            
            # 还在处理中 (submitted, processing, pending 等)
            log.info("kling task %s status: %s, waiting...", task_id, status)
            await asyncio.sleep(interval)
            elapsed += interval
            
    return {"error": f"Timed out after {max_wait}s waiting for video generation"}


def extract_video_url(result: dict) -> str:
    """从 Kling 任务成功结果中提取视频 URL，兼容多种返回结构"""
    
    # 尝试各种可能的字段名
    # 1. works 数组
    works = result.get("works") or result.get("videos") or []
    if works:
        work = works[0] if isinstance(works, list) else works
        if isinstance(work, dict):
            video = work.get("video", work)
            url = (video.get("resource") or 
                   video.get("url") or 
                   video.get("resource_without_watermark") or
                   video.get("video_url"))
            if url:
                return url
    
    # 2. 直接的 URL 字段
    for field in ["video_url", "url", "output_url", "result_url", "file_url"]:
        if result.get(field):
            return result[field]
    
    # 3. output 字段
    output = result.get("output")
    if output:
        if isinstance(output, str) and output.startswith("http"):
            return output
        if isinstance(output, list) and output:
            return output[0] if isinstance(output[0], str) else output[0].get("url", "")
        if isinstance(output, dict):
            return output.get("url", "")
    
    return ""


async def generate_video_with_image(prompt: str, image_base64: str) -> str:
    """图生视频：输入图片+文字，生成视频"""
    gen_url = "https://yunwu.ai/kling/image-to-video/kling-3.0-turbo"
    payload = {
        "model": "kling-3.0-turbo",
        "prompt": prompt,
        "image": f"data:image/jpeg;base64,{image_base64}"
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as http:
            resp = await http.post(gen_url, json=payload, headers=headers)
            data = resp.json()

        log.info("kling image-to-video submit response: %s", json.dumps(data)[:500])

        if data.get("code") != 0:
            return f"Video generation error: {data.get('message', resp.text)}"

        # 获取任务 ID - 可能是 task_id 或 id
        task_id = data.get("data", {}).get("task_id") or data.get("data", {}).get("id")
        if not task_id:
            return f"No task_id in response: {json.dumps(data)[:200]}"

        log.info("kling task submitted: %s, polling...", task_id)

        # 轮询等待结果
        result = await poll_kling_task(task_id, task_type="image-to-video")
        if "error" in result:
            return f"Video generation error: {result['error']}"

        video_url = extract_video_url(result)
        if video_url:
            return video_url
        return f"Could not find video URL in result: {json.dumps(result)[:200]}"
        
    except Exception as e:
        log.exception("generate_video_with_image failed")
        return f"Video generation failed: {str(e)}"


async def generate_video_text_only(prompt: str) -> str:
    """文生视频：只输入文字，生成视频"""
    gen_url = "https://yunwu.ai/kling/text-to-video/kling-3.0-turbo"
    payload = {
        "model": "kling-3.0-turbo",
        "prompt": prompt
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as http:
            resp = await http.post(gen_url, json=payload, headers=headers)
            data = resp.json()

        log.info("kling text-to-video submit response: %s", json.dumps(data)[:500])

        if data.get("code") != 0:
            return f"Video generation error: {data.get('message', resp.text)}"

        # 获取任务 ID - 可能是 task_id 或 id
        task_id = data.get("data", {}).get("task_id") or data.get("data", {}).get("id")
        if not task_id:
            return f"No task_id in response: {json.dumps(data)[:200]}"

        log.info("kling task submitted: %s, polling...", task_id)

        # 轮询等待结果
        result = await poll_kling_task(task_id, task_type="text-to-video")
        if "error" in result:
            return f"Video generation error: {result['error']}"

        video_url = extract_video_url(result)
        if video_url:
            return video_url
        return f"Could not find video URL in result: {json.dumps(result)[:200]}"
        
    except Exception as e:
        log.exception("generate_video_text_only failed")
        return f"Video generation failed: {str(e)}"


async def send_image_result(msg, result: str, original_prompt: str = ""):
    if result.startswith("http") or result.startswith("data:image"):
        try:
            await msg.reply_photo(result, caption=original_prompt[:200] if original_prompt else None)
            await msg.edit_text("✅ 图片已完成")
            return True
        except Exception as e:
            log.error("发送图片失败: %s", e)
            if result.startswith("data:image"):
                try:
                    base64_data = result.split(",")[1] if "," in result else result
                    image_bytes = base64.b64decode(base64_data)
                    await msg.reply_photo(InputFile(io.BytesIO(image_bytes), filename="image.png"))
                    await msg.edit_text("✅ 图片已完成")
                    return True
                except Exception as e2:
                    log.error("base64 发送失败: %s", e2)
    await msg.edit_text(f"❌ {result}")
    return False


def get_model_type(model_id: str) -> str:
    for prefix, info in MODEL_MAP.items():
        if info["id"] == model_id:
            return info["type"]
    return "chat"


async def stream_reply(msg, messages, model, image_base64: str = None):
    full = ""
    last_len = 0
    last_t = time.time()

    user_content = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            raw = m.get("content", "")
            if isinstance(raw, list):
                for item in raw:
                    if item.get("type") == "text":
                        user_content = item.get("text", "")
                        break
            else:
                user_content = raw
            break

    model_type = get_model_type(model)

    # ===== 视频生成 =====
    if model_type == "video":
        prompt = user_content
        for prefix in ["视频", "生成", "拍", "录", "SP"]:
            prompt = prompt.replace(prefix, "").strip()
        if not prompt:
            prompt = user_content

        await msg.edit_text("🎬 视频生成中，请稍候（约2-5分钟）...")

        if image_base64:
            result = await generate_video_with_image(prompt, image_base64)
        else:
            result = await generate_video_text_only(prompt)

        if result.startswith("http"):
            try:
                await msg.reply_video(result, caption=prompt[:200] if prompt else None)
                await msg.edit_text("🎬 视频已生成！")
            except Exception as e:
                log.error("发送视频失败: %s", e)
                await msg.edit_text(f"🎬 视频生成完成！\n链接：{result}")
        else:
            await msg.edit_text(f"❌ {result}")
        return "video_generated"

    # ===== 图片生成/编辑 =====
    if model_type == "image":
        prompt = user_content
        for prefix in ["画", "生成", "图片", "照片", "图", "绘", "NB"]:
            prompt = prompt.replace(prefix, "").strip()
        if not prompt:
            prompt = user_content

        if image_base64:
            result = await generate_image_edit(prompt, image_base64)
        else:
            result = await generate_image(prompt)

        await send_image_result(msg, result, prompt)
        return "image_generated"

    # ===== 文本聊天 =====
    try:
        stream = await client.chat.completions.create(
            model=model, messages=messages, stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                full += chunk.choices[0].delta.content
            now = time.time()
            if full and (now - last_t > 1.2 or len(full) - last_len > 120):
                try:
                    preview = full[:4000] + "..." if len(full) > 4000 else full + " ▌"
                    await msg.edit_text(preview)
                except Exception:
                    pass
                last_t = now
                last_len = len(full)
    except Exception as e:
        err = str(e)
        if "401" in err or "403" in err:
            await msg.edit_text("Error: Authentication failed — check API_KEY")
        elif "429" in err:
            await msg.edit_text("Rate limited — wait a moment and try again.")
        else:
            await msg.edit_text(f"API error: {err[:300]}")
        return None

    if not full:
        await msg.edit_text("(empty response)")
        return None

    escaped = escape_md(full)
    await send_long(msg, escaped)
    return full


# ============================================
# 命令处理器
# ============================================
async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    lines = [
        "**🤖 多模型 Bot**",
        "",
        "**可用模型：**",
    ]
    for prefix, info in MODEL_MAP.items():
        emoji = "🖼️" if info["type"] == "image" else "🎬" if info["type"] == "video" else "💬"
        lines.append(f"  • `{prefix}` → {emoji} {info['id']}")
    lines.append(f"  • (无前缀) → {DEFAULT_MODEL}")
    lines.append("")
    lines.append("**📷 图片操作：**")
    lines.append("  • 发图片 + `NB 把猫变成白色` → 修改/修复图片")
    lines.append("  • `NB 画一只猫` → 生成新图片")
    lines.append("")
    lines.append("**🎬 视频操作：**")
    lines.append("  • 发图片 + `SP 让它挥手` → 图生视频")
    lines.append("  • `SP 生成3秒视频` → 文生视频")
    lines.append("")
    lines.append("**💬 聊天：**")
    lines.append("  • `A 解释量子计算` → AI 回答")
    lines.append("  • `ZZ 写一段代码` → AI 编程")
    lines.append("")
    lines.append("**其他：**")
    lines.append("  • `/clear` — 清空对话历史")

    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN_V2)


async def cmd_clear(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    histories[update.effective_user.id] = []
    await update.message.reply_text("✅ 对话已重置")


# ============================================
# 消息处理器
# ============================================
async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not allowed(uid) or not should_reply(update):
        return

    raw_text = strip_mention(update.message.text)
    if not raw_text:
        return

    image_base64 = None
    if update.message.photo:
        photo_file = update.message.photo[-1]
        try:
            file = await ctx.bot.get_file(photo_file.file_id)
            raw = await file.download_as_bytearray()
            img = Image.open(io.BytesIO(raw))
            img.thumbnail((1024, 1024))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            image_base64 = base64.b64encode(buf.getvalue()).decode()
        except Exception as e:
            log.error("图片处理失败: %s", e)

    selected_model, clean_text = detect_model_from_text(raw_text)
    if selected_model is None:
        selected_model = DEFAULT_MODEL

    user_models[uid] = selected_model

    if not clean_text:
        await update.message.reply_text("请告诉我你想做什么，比如：`NB 画一只猫`", parse_mode=ParseMode.MARKDOWN_V2)
        return

    hist = get_hist(uid)
    hist.append({"role": "user", "content": clean_text})

    while len(hist) > MAX_HISTORY:
        hist.pop(0)

    msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + hist

    placeholder = await update.message.reply_text(f"⏳ 使用模型: {selected_model}")
    reply = await stream_reply(placeholder, msgs, selected_model, image_base64)
    if reply and reply not in ["image_generated", "video_generated"]:
        hist.append({"role": "assistant", "content": reply})


async def on_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not allowed(uid) or not should_reply(update):
        return

    raw_caption = strip_mention(update.message.caption or "") or "What's in this image?"
    selected_model, clean_caption = detect_model_from_text(raw_caption)
    if selected_model is None:
        selected_model = DEFAULT_MODEL

    try:
        if update.message.photo:
            file = await ctx.bot.get_file(update.message.photo[-1].file_id)
        elif update.message.document and update.message.document.mime_type and update.message.document.mime_type.startswith("image/"):
            file = await ctx.bot.get_file(update.message.document.file_id)
        else:
            return
        raw = await file.download_as_bytearray()
        img = Image.open(io.BytesIO(raw))
        img.thumbnail((1024, 1024))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        log.error("failed to process photo: %s", e)
        await update.message.reply_text("Couldn't process the image. Try again?")
        return

    selected_model_type = get_model_type(selected_model)

    if selected_model_type in ("video", "image"):
        placeholder = await update.message.reply_text(f"⏳ 使用模型: {selected_model}")
        reply = await stream_reply(placeholder, [{"role": "user", "content": clean_caption}], selected_model, b64)
        return

    vision_msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": clean_caption},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ],
    }

    hist = get_hist(uid)
    msgs
