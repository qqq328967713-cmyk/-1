#!/usr/bin/env python3
import os
import sys
import io
import re
import time
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
BASE_URL = os.getenv("BASE_URL", "https://api.tokenmix.ai/v1")
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
    """
    检测文本中的指令前缀，返回 (模型ID, 去除前缀后的文本)
    """
    text = text.strip()
    if not text:
        return None, text

    # 先检查完整匹配（前缀+分隔符）
    for prefix, info in MODEL_MAP.items():
        model_id = info["id"]
        pattern = rf"^{re.escape(prefix)}[\s:：\n]+"
        match = re.match(pattern, text, re.IGNORECASE)
        if match:
            clean_text = text[match.end():].strip()
            return model_id, clean_text

    # 再检查无分隔符的情况（如 "NB画一只猫"）
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
    """调用 /images/generations 接口生成图片"""
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


async def generate_video(prompt: str) -> str:
    """调用 /kling/image-to-video/kling-3.0-turbo 接口生成视频"""
    gen_url = f"{BASE_URL}/kling/image-to-video/kling-3.0-turbo" 
    payload = {
        "model": "kling-3.0-turbo",
        "prompt": prompt,
        "duration": 3
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        async with httpx.AsyncClient(timeout=120.0) as http:
            resp = await http.post(gen_url, json=payload, headers=headers)
            data = resp.json()
        if resp.status_code != 200:
            return f"Video generation error: {data.get('error', {}).get('message', resp.text)}"
        if "data" in data and len(data["data"]) > 0:
            item = data["data"][0]
            if "url" in item and item["url"]:
                return item["url"]
            if "video_url" in item:
                return item["video_url"]
        if "url" in data:
            return data["url"]
        if "video_url" in data:
            return data["video_url"]
        return f"Unexpected response: {json.dumps(data)[:200]}"
    except Exception as e:
        return f"Video generation failed: {str(e)}"


async def send_image_result(msg, result: str, original_prompt: str = ""):
    """发送图片结果"""
    if result.startswith("http") or result.startswith("data:image"):
        try:
            await msg.reply_photo(result, caption=original_prompt[:200] if original_prompt else None)
            await msg.edit_text("✅ 绘图已完成")
            return True
        except Exception as e:
            log.error("发送图片失败: %s", e)
            if result.startswith("data:image"):
                try:
                    base64_data = result.split(",")[1] if "," in result else result
                    image_bytes = base64.b64decode(base64_data)
                    await msg.reply_photo(InputFile(io.BytesIO(image_bytes), filename="image.png"))
                    await msg.edit_text("✅ 绘图已完成")
                    return True
                except Exception as e2:
                    log.error("base64 发送失败: %s", e2)
    await msg.edit_text(f"❌ {result}")
    return False


def get_model_type(model_id: str) -> str:
    """根据模型ID获取类型"""
    for prefix, info in MODEL_MAP.items():
        if info["id"] == model_id:
            return info["type"]
    return "chat"


async def stream_reply(msg, messages, model):
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
        for prefix in ["视频", "生成", "拍", "录"]:
            prompt = prompt.replace(prefix, "").strip()
        if not prompt:
            prompt = user_content
        result = await generate_video(prompt)
        if result.startswith("http"):
            try:
                await msg.reply_video(result, caption=prompt[:200] if prompt else None)
                await msg.edit_text("🎬 视频已生成！")
            except Exception:
                await msg.edit_text(f"生成的视频：{result}")
        else:
            await msg.edit_text(f"❌ {result}")
        return "video_generated"

    # ===== 图片生成 =====
    if model_type == "image":
        prompt = user_content
        for prefix in ["画", "生成", "图片", "照片", "图", "绘"]:
            prompt = prompt.replace(prefix, "").strip()
        if not prompt:
            prompt = user_content
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
    lines = ["**可用模型：**"]
    for prefix, info in MODEL_MAP.items():
        emoji = "🖼️" if info["type"] == "image" else "🎬" if info["type"] == "video" else "💬"
        lines.append(f"  • `{prefix}` → {emoji} {info['id']}")
    lines.append(f"  • (无前缀) → {DEFAULT_MODEL}")
    lines.append("\n示例：")
    lines.append("  `NB 画一只猫` → 生成图片")
    lines.append("  `SP 生成3秒视频` → 生成视频")
    lines.append("  `A 解释量子计算` → AI 回答")
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
    reply = await stream_reply(placeholder, msgs, selected_model)
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

    vision_msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": clean_caption},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ],
    }

    hist = get_hist(uid)
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + hist + [vision_msg]

    placeholder = await update.message.reply_text(f"⏳ 分析图片中... (模型: {selected_model})")
    reply = await stream_reply(placeholder, msgs, selected_model)
    if reply and reply not in ["image_generated", "video_generated"]:
        hist.append({"role": "user", "content": f"[photo] {clean_caption}"})
        hist.append({"role": "assistant", "content": reply})


async def on_voice(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not allowed(uid):
        return

    voice = update.message.voice or update.message.audio
    if not voice:
        return

    try:
        file = await ctx.bot.get_file(voice.file_id)
        raw = await file.download_as_bytearray()
        audio_file = io.BytesIO(raw)
        audio_file.name = "voice.ogg"
        transcript = await client.audio.transcriptions.create(
            model="whisper-1", file=audio_file,
        )
        text = transcript.text.strip()
    except Exception as e:
        log.warning("transcription failed: %s", e)
        await update.message.reply_text("Voice transcription isn't supported by your current API provider.")
        return

    if not text:
        await update.message.reply_text("(couldn't make out the audio)")
        return

    hist = get_hist(uid)
    hist.append({"role": "user", "content": text})

    while len(hist) > MAX_HISTORY:
        hist.pop(0)

    msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + hist
    model = user_models.get(uid, DEFAULT_MODEL)

    placeholder = await update.message.reply_text(f"🎤 \"{text}\"\n\n⏳ 处理中...")
    reply = await stream_reply(placeholder, msgs, model)
    if reply and reply not in ["image_generated", "video_generated"]:
        hist.append({"role": "assistant", "content": reply})


def main():
    global _bot_username

    if not BOT_TOKEN:
        print("Error: BOT_TOKEN not set.")
        sys.exit(1)

    if not API_KEY:
        print("Error: API_KEY not set.")
        sys.exit(1)

    logging.basicConfig(
        format="%(asctime)s [%(name)s] %(message)s",
        level=logging.INFO,
    )

    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_start))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, on_photo))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, on_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    async def on_error(update, ctx):
        log.error("unhandled error: %s", ctx.error, exc_info=ctx.error)

    app.add_error_handler(on_error)

    async def post_init(application):
        global _bot_username
        bot = await application.bot.get_me()
        _bot_username = bot.username
        log.info("bot @%s started — default model: %s", _bot_username, DEFAULT_MODEL)

    app.post_init = post_init
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
