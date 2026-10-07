import os
import asyncio
import threading
import time

import streamlit as st

st.set_page_config(
    page_title="Cub AI Solver",
    page_icon="🧠",
    layout="centered",
)

required_secrets = ["TELEGRAM_BOT_TOKEN", "GEMINI_API_KEY"]
missing = [key for key in required_secrets if key not in st.secrets]

if missing:
    st.error("Missing Streamlit secrets:\n\n" + "\n".join(missing))
    st.stop()

os.environ["TELEGRAM_BOT_TOKEN"] = st.secrets["TELEGRAM_BOT_TOKEN"]
os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
os.environ["GEMINI_LIVE_MODEL"] = st.secrets.get("GEMINI_LIVE_MODEL", "gemini-3.8-live")
os.environ["GEMINI_LATEX_MODEL"] = st.secrets.get("GEMINI_LATEX_MODEL", "gemini-3.1-flash-lite")
os.environ["CHROMIUM_PATH"] = st.secrets.get("CHROMIUM_PATH", "/usr/bin/chromium")


@st.cache_resource
def start_telegram_bot():
    state = {"started": False, "error": None}

    def runner():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            from bot import run_bot

            state["started"] = True
            run_bot()
        except Exception as error:
            state["error"] = f"{type(error).__name__}: {error}"
            state["started"] = False

    thread = threading.Thread(
        target=runner,
        name="CubTelegramBot",
        daemon=True,
    )
    thread.start()
    time.sleep(2)
    return thread, state


bot_thread, bot_state = start_telegram_bot()

st.title("🧠 Cub")
st.subheader("AI Question Solver")
st.write(
    """
Send a question image to the Cub Telegram bot.

Cub reads the question, solves it step by step, reconstructs proper LaTeX mathematics,
renders it with MathJax, and returns a polished solution image.
"""
)

if bot_state["error"]:
    st.error("Telegram bot failed to start.")
    st.code(bot_state["error"])
elif bot_thread.is_alive():
    st.success("✅ Telegram bot is running")
else:
    st.warning("⚠️ Telegram bot thread is not running")

st.divider()
st.subheader("AI Pipeline")
st.code(
    f"""
Question Image
      ↓
{os.environ['GEMINI_LIVE_MODEL']}
      ↓
Step-by-Step Draft
      ↓
{os.environ['GEMINI_LATEX_MODEL']}
      ↓
Verified LaTeX
      ↓
MathJax + Chromium
      ↓
High-resolution PNG
      ↓
Telegram
""".strip()
)

st.divider()
st.info(
    "Streamlit Community Cloud may put inactive apps to sleep. "
    "For a true 24/7 Telegram bot, use a VPS or an always-on service."
)
st.caption("Cub Telegram Bot • Streamlit deployment dashboard")
