import os
import re
import io
import html
import asyncio
import logging

import fitz  # PyMuPDF
from dotenv import load_dotenv
from playwright.async_api import async_playwright

from telegram import Update, BotCommand
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from google import genai
from google.genai import types


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LIVE_MODEL = os.getenv("GEMINI_LIVE_MODEL", "gemini-3.8-live")
LATEX_MODEL = os.getenv("GEMINI_LATEX_MODEL", "gemini-3.1-flash-lite")
CHROMIUM_PATH = os.getenv("CHROMIUM_PATH", "").strip()

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is missing.")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing.")


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("Cub")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


# ============================================================
# CLIENTS / GLOBALS
# ============================================================

gemini = genai.Client(api_key=GEMINI_API_KEY)
PLAYWRIGHT = None
BROWSER = None


# ============================================================
# SETTINGS
# ============================================================

MAX_IMAGE_SIZE = 20 * 1024 * 1024
SUPPORTED_IMAGE_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}
DEFAULT_OUTPUT_MODE = "image"
VALID_OUTPUT_MODES = {"image", "text", "both"}


# ============================================================
# PROMPTS
# ============================================================

LIVE_SOLVER_PROMPT = """
You are Cub, an expert academic problem-solving assistant.

A picture containing a question is attached to this request.
You can already see the image. Never ask the user to upload or resend it.

Carefully inspect the question and solve it.

ABSOLUTE REQUIREMENT:
Always solve the problem step by step. Do not give only the final answer.

For mathematics:
- identify the exact expression
- identify what must be found
- show formulas
- show substitutions
- show algebra
- show calculations
- explain important transformations
- verify the result
- clearly state the final answer

For physics:
- identify given quantities
- identify what must be found
- use the correct formula
- substitute values
- calculate step by step
- preserve units

For chemistry:
- preserve formulas and reactions
- explain important transformations
- balance equations where required
- show calculations

For MCQs:
- reason through the question
- show important steps
- give the correct option at the end

For coding:
- explain the logic
- show the approach
- provide correct code when required

For theory:
- give a clear, structured, exam-ready answer.

If multiple questions are visible, solve them separately.
Read numbers, symbols, powers, fractions and signs very carefully.
Do not invent unreadable information.
Before finishing, verify the final answer.
"""

LATEX_FORMATTER_PROMPT = r"""
You are the final mathematical editor for an academic question-solving system.

You receive:
1. The ORIGINAL QUESTION IMAGE.
2. A draft step-by-step solution produced by another model.

You must inspect the original image yourself, verify the exact question,
verify every calculation, correct any mistake in the draft, produce a polished
step-by-step solution, and typeset ALL mathematics using valid LaTeX.

The final result will be rendered by MathJax.

STRICT OUTPUT FORMAT

Use ONLY these section headings:

QUESTION:
GIVEN:
REQUIRED:
FORMULA:
STEP 1:
STEP 2:
STEP 3:
STEP 4:
...
FINAL ANSWER:

For MCQ questions you may additionally use:
CORRECT OPTION:

Do not write an introduction before QUESTION:.
Do not write anything after the final section.
Do not use Markdown headings, Markdown code fences, or HTML.

LATEX RULES

All mathematical expressions must use LaTeX.
Use inline mathematics as \( ... \).
Use important equations or multi-line calculations as \[ ... \].

Examples:

\[
\sum_{n=1}^{\infty}
\frac{(4n+5)5^{-n}}{n(5n+5)}
\]

\[
\frac{4n+5}{n(n+1)}
=
\frac{5}{n}
-
\frac{1}{n+1}
\]

Use \frac{a}{b} for fractions, x^{2} for powers, x_{n} for subscripts,
\sqrt{x} for roots, \sum for summation, \lim for limits, \int for integrals,
and proper matrix environments for matrices.

Use \begin{aligned} ... \end{aligned} for aligned calculations where useful.
The final result should preferably use \boxed{...}.

Always solve step by step. Do not create unnecessary tiny steps.
Each step should represent a meaningful action.
Preserve the exact question from the image.
Do not trust the draft blindly; correct it if necessary.
Keep explanatory prose concise but sufficient for learning.
"""


def build_live_prompt(extra_instruction: str = "") -> str:
    prompt = LIVE_SOLVER_PROMPT.strip()
    if extra_instruction:
        prompt += (
            "\n\nAdditional instruction from the user:\n"
            f"{extra_instruction.strip()}\n\n"
            "Still solve the problem step by step."
        )
    return prompt


def parse_output_mode(arguments):
    if not arguments:
        return DEFAULT_OUTPUT_MODE, ""

    parts = list(arguments)
    first = parts[0].lower().strip()

    if first in VALID_OUTPUT_MODES:
        return first, " ".join(parts[1:]).strip()

    return DEFAULT_OUTPUT_MODE, " ".join(parts).strip()


# ============================================================
# GEMINI LIVE
# ============================================================

async def receive_live_answer(session) -> str:
    transcript = []
    fallback = []

    async for response in session.receive():
        server_content = getattr(response, "server_content", None)
        if not server_content:
            continue

        transcription = getattr(server_content, "output_transcription", None)
        if transcription:
            value = getattr(transcription, "text", None)
            if value:
                transcript.append(value)

        model_turn = getattr(server_content, "model_turn", None)
        if model_turn:
            parts = getattr(model_turn, "parts", None)
            if parts:
                for part in parts:
                    value = getattr(part, "text", None)
                    if value:
                        fallback.append(value)

        if getattr(server_content, "turn_complete", False):
            break

    answer = "".join(transcript).strip()
    if not answer:
        answer = "".join(fallback).strip()
    return answer


async def solve_with_live(
    image_bytes: bytes,
    mime_type: str,
    extra_instruction: str = "",
) -> str:
    prompt = build_live_prompt(extra_instruction)
    config = {
        "response_modalities": ["AUDIO"],
        "output_audio_transcription": {},
    }

    last_error = None

    for attempt in range(1, 3):
        try:
            logger.info("Gemini Live connection | attempt=%s", attempt)

            async with gemini.aio.live.connect(
                model=LIVE_MODEL,
                config=config,
            ) as session:
                content = types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(
                            data=image_bytes,
                            mime_type=mime_type,
                        ),
                        types.Part(text=prompt),
                    ],
                )

                await session.send_client_content(
                    turns=content,
                    turn_complete=True,
                )

                answer = await asyncio.wait_for(
                    receive_live_answer(session),
                    timeout=120,
                )

                if not answer:
                    raise RuntimeError("Gemini Live returned an empty answer.")

                return answer

        except asyncio.CancelledError:
            raise
        except Exception as error:
            last_error = error
            logger.warning("Gemini Live attempt %s failed: %s", attempt, error)
            if attempt < 2:
                await asyncio.sleep(2)

    raise RuntimeError(f"Gemini Live failed: {last_error}")


# ============================================================
# LATEX VERIFIER
# ============================================================

def clean_model_output(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(
            r"^```(?:latex|tex|text)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


async def convert_solution_to_latex(
    image_bytes: bytes,
    mime_type: str,
    draft_solution: str,
    extra_instruction: str = "",
) -> str:
    prompt = LATEX_FORMATTER_PROMPT
    prompt += (
        "\n\n==================================================\n"
        "DRAFT SOLUTION FROM GEMINI LIVE\n"
        "==================================================\n\n"
        f"{draft_solution}\n"
    )

    if extra_instruction:
        prompt += (
            "\n\n==================================================\n"
            "USER'S ADDITIONAL INSTRUCTION\n"
            "==================================================\n\n"
            f"{extra_instruction}\n"
        )

    contents = types.Content(
        role="user",
        parts=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type,
            ),
            types.Part(text=prompt),
        ],
    )

    response = await asyncio.wait_for(
        gemini.aio.models.generate_content(
            model=LATEX_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(temperature=0.1),
        ),
        timeout=120,
    )

    output = getattr(response, "text", None)
    if not output:
        raise RuntimeError("LaTeX formatter returned an empty response.")

    return clean_model_output(output)


# ============================================================
# TELEGRAM IMAGE INPUT
# ============================================================

def message_has_image(message) -> bool:
    if message.photo:
        return True
    if message.document:
        mime = (message.document.mime_type or "").lower()
        return mime.startswith("image/")
    return False


async def get_image_bytes(message):
    if message.photo:
        photo = message.photo[-1]
        telegram_file = await photo.get_file()
        data = await telegram_file.download_as_bytearray()
        image_bytes = bytes(data)

        if len(image_bytes) > MAX_IMAGE_SIZE:
            raise ValueError("Image exceeds the 20 MB limit.")

        return image_bytes, "image/jpeg"

    if message.document:
        mime_type = (message.document.mime_type or "").lower()

        if mime_type not in SUPPORTED_IMAGE_TYPES:
            raise ValueError(
                "Unsupported image format. Please use JPG, PNG or WEBP."
            )

        if (
            message.document.file_size
            and message.document.file_size > MAX_IMAGE_SIZE
        ):
            raise ValueError("Image exceeds the 20 MB limit.")

        telegram_file = await message.document.get_file()
        data = await telegram_file.download_as_bytearray()
        return bytes(data), mime_type

    raise ValueError("No question image was found.")


# ============================================================
# STRUCTURED ANSWER PARSER
# ============================================================

SECTION_PATTERN = re.compile(
    r"^(QUESTION(?:\s+\d+)?|"
    r"GIVEN|"
    r"REQUIRED|"
    r"FORMULA|"
    r"STEP\s+\d+|"
    r"FINAL\s+ANSWER|"
    r"CORRECT\s+OPTION)"
    r"\s*:\s*(.*)$",
    re.IGNORECASE,
)


def parse_sections(text: str):
    sections = []
    current_title = None
    current_lines = []

    def flush():
        nonlocal current_title, current_lines
        if current_title is not None:
            sections.append((current_title, "\n".join(current_lines).strip()))
        current_title = None
        current_lines = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        match = SECTION_PATTERN.match(line.strip())

        if match:
            flush()
            current_title = match.group(1).upper().strip()
            first_content = match.group(2).strip()
            if first_content:
                current_lines.append(first_content)
        else:
            if current_title is None:
                if line.strip():
                    current_title = "SOLUTION"
                    current_lines.append(line)
            else:
                current_lines.append(line)

    flush()

    if not sections:
        sections = [("SOLUTION", text.strip())]

    return sections


def section_class(title: str) -> str:
    upper = title.upper()
    if upper.startswith("FINAL ANSWER") or upper.startswith("CORRECT OPTION"):
        return "final"
    if upper.startswith("QUESTION"):
        return "question"
    if upper.startswith("STEP"):
        return "step"
    return "normal"


def pretty_section_title(title: str) -> str:
    upper = title.upper()
    if upper.startswith("STEP") or upper.startswith("QUESTION"):
        return upper.title()
    if upper == "FINAL ANSWER":
        return "Final Answer"
    if upper == "CORRECT OPTION":
        return "Correct Option"
    return upper.title()


# ============================================================
# HTML + MATHJAX
# ============================================================

def create_solution_html(latex_answer: str) -> str:
    rendered_sections = []

    for title, body in parse_sections(latex_answer):
        css_class = section_class(title)
        safe_title = html.escape(pretty_section_title(title), quote=False)
        safe_body = html.escape(body, quote=False)

        if css_class == "final":
            rendered_sections.append(
                f"""
                <section class="solution-section final-section">
                    <div class="final-title">{safe_title}</div>
                    <div class="section-content final-content">{safe_body}</div>
                </section>
                """
            )
        else:
            rendered_sections.append(
                f"""
                <section class="solution-section {css_class}">
                    <div class="section-title">{safe_title}</div>
                    <div class="section-content">{safe_body}</div>
                </section>
                """
            )

    sections_html = "\n".join(rendered_sections)

    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">

<script>
window.MathJax = {{
    tex: {{
        inlineMath: [['\\\\(', '\\\\)']],
        displayMath: [['\\\\[', '\\\\]']],
        processEscapes: true,
        packages: {{ '[+]': ['ams'] }}
    }},
    svg: {{ fontCache: 'local' }}
}};
</script>
<script id="MathJax-script" async
    src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>

<style>
@page {{ size: A4; margin: 14mm 15mm 16mm 15mm; }}
* {{ box-sizing: border-box; }}
html, body {{ background: #ffffff; }}
body {{
    margin: 0;
    padding: 0;
    color: #20242c;
    font-family: "Segoe UI", "Arial", sans-serif;
    font-size: 16px;
    line-height: 1.58;
}}
.header {{
    background: #202e49;
    color: white;
    border-radius: 13px;
    padding: 22px 26px;
    margin-bottom: 27px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}}
.brand {{ font-size: 27px; font-weight: 800; letter-spacing: 0.5px; }}
.header-subtitle {{
    color: #d6deed;
    font-size: 15px;
    letter-spacing: 1.3px;
    text-transform: uppercase;
}}
.solution-section {{ margin: 0 0 22px 0; break-inside: auto; }}
.section-title {{
    background: #edf2fb;
    color: #245391;
    border-radius: 8px;
    padding: 8px 13px;
    margin-bottom: 11px;
    font-weight: 750;
    font-size: 17px;
    break-after: avoid;
}}
.section-content {{
    padding: 1px 9px;
    white-space: pre-wrap;
    overflow-wrap: break-word;
}}
.question .section-title {{ background: #e8eef9; color: #1f4f8b; }}
mjx-container {{ color: #131923; }}
mjx-container[display="true"] {{
    margin: 18px 0 !important;
    font-size: 116% !important;
    max-width: 100%;
}}
.final-section {{
    margin-top: 28px;
    margin-bottom: 25px;
    padding: 20px 22px;
    border: 2px solid #6bb783;
    border-radius: 13px;
    background: #eef9f1;
    break-inside: avoid;
}}
.final-title {{
    color: #237540;
    font-size: 19px;
    font-weight: 800;
    text-transform: uppercase;
    text-align: center;
    margin-bottom: 11px;
}}
.final-content {{
    color: #175b31;
    text-align: center;
    font-size: 18px;
    font-weight: 600;
}}
.final-content mjx-container[display="true"] {{ font-size: 145% !important; }}
.footer {{
    margin-top: 32px;
    padding-top: 12px;
    border-top: 1px solid #dce1e8;
    color: #89909c;
    font-size: 11px;
    text-align: center;
}}
</style>
</head>
<body>
<header class="header">
    <div class="brand">CUB</div>
    <div class="header-subtitle">Step-by-Step Solution</div>
</header>
<main>{sections_html}</main>
<footer class="footer">Generated by Cub • AI Question Solver</footer>
</body>
</html>
"""


# ============================================================
# CHROMIUM RENDERER
# ============================================================

async def render_html_to_pdf(document_html: str) -> bytes:
    global BROWSER

    if BROWSER is None:
        raise RuntimeError("Chromium renderer is not running.")

    page = await BROWSER.new_page(
        viewport={"width": 1200, "height": 1600}
    )

    try:
        await page.set_content(
            document_html,
            wait_until="domcontentloaded",
            timeout=60000,
        )

        await page.wait_for_function(
            """
            () => (
                window.MathJax &&
                MathJax.startup &&
                MathJax.startup.promise
            )
            """,
            timeout=30000,
        )

        await page.evaluate(
            """
            async () => {
                await MathJax.startup.promise;
                if (MathJax.typesetPromise) {
                    await MathJax.typesetPromise();
                }
                if (document.fonts) {
                    await document.fonts.ready;
                }
            }
            """
        )

        await page.wait_for_timeout(300)

        return await page.pdf(
            format="A4",
            print_background=True,
            prefer_css_page_size=True,
            margin={
                "top": "0mm",
                "right": "0mm",
                "bottom": "0mm",
                "left": "0mm",
            },
        )
    finally:
        await page.close()


def pdf_to_png_pages(pdf_bytes: bytes):
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []

    try:
        matrix = fitz.Matrix(2.3, 2.3)

        for page_number in range(document.page_count):
            page = document.load_page(page_number)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            output = io.BytesIO(pixmap.tobytes("png"))
            output.name = f"cub_solution_{page_number + 1}.png"
            output.seek(0)
            images.append(output)
    finally:
        document.close()

    return images


async def create_solution_images(latex_answer: str):
    pdf_bytes = await render_html_to_pdf(
        create_solution_html(latex_answer)
    )
    return await asyncio.to_thread(pdf_to_png_pages, pdf_bytes)


async def send_solution_images(message, latex_answer: str):
    await message.chat.send_action(action=ChatAction.UPLOAD_PHOTO)
    pages = await create_solution_images(latex_answer)
    total = len(pages)

    try:
        for index, page in enumerate(pages, start=1):
            caption = (
                "✅ Step-by-step solution"
                if total == 1
                else f"✅ Step-by-step solution\nPage {index}/{total}"
            )

            size = page.getbuffer().nbytes
            page.seek(0)

            if size < 9 * 1024 * 1024:
                await message.reply_photo(photo=page, caption=caption)
            else:
                await message.reply_document(document=page, caption=caption)
    finally:
        for page in pages:
            try:
                page.close()
            except Exception:
                pass


# ============================================================
# TEXT OUTPUT
# ============================================================

async def send_text_solution(message, draft_solution: str):
    text = draft_solution.strip()
    max_length = 3900

    while text:
        if len(text) <= max_length:
            chunk = text
            text = ""
        else:
            cut = text.rfind("\n", 0, max_length)
            if cut < 2500:
                cut = text.rfind(" ", 0, max_length)
            if cut < 2500:
                cut = max_length
            chunk = text[:cut]
            text = text[cut:].lstrip()

        await message.reply_text(chunk.strip())


# ============================================================
# SOLVE PIPELINE
# ============================================================

async def solve_message(
    update: Update,
    source_message,
    output_mode: str = DEFAULT_OUTPUT_MODE,
    extra_instruction: str = "",
):
    message = update.effective_message
    if not message:
        return

    status = await message.reply_text(
        "🔍 <b>Reading question...</b>\n🧠 Solving step by step...",
        parse_mode=ParseMode.HTML,
    )

    try:
        image_bytes, mime_type = await get_image_bytes(source_message)

        logger.info(
            "Image received | size=%s | type=%s",
            len(image_bytes),
            mime_type,
        )

        await status.edit_text(
            "🔍 <b>Question detected</b>\n🧠 Solving with Gemini Live...",
            parse_mode=ParseMode.HTML,
        )

        draft_solution = await solve_with_live(
            image_bytes=image_bytes,
            mime_type=mime_type,
            extra_instruction=extra_instruction,
        )

        if output_mode == "text":
            try:
                await status.delete()
            except Exception:
                pass
            await send_text_solution(message, draft_solution)
            return

        await status.edit_text(
            "✅ <b>Solution completed</b>\n📐 Typesetting mathematics...",
            parse_mode=ParseMode.HTML,
        )

        latex_solution = await convert_solution_to_latex(
            image_bytes=image_bytes,
            mime_type=mime_type,
            draft_solution=draft_solution,
            extra_instruction=extra_instruction,
        )

        await status.edit_text(
            "📐 <b>Mathematics typeset</b>\n🖼 Creating solution image...",
            parse_mode=ParseMode.HTML,
        )

        try:
            await status.delete()
        except Exception:
            pass

        await send_solution_images(message, latex_solution)

        if output_mode == "both":
            await send_text_solution(message, draft_solution)

    except asyncio.TimeoutError:
        try:
            await status.edit_text(
                "❌ The request took too long.\n\nPlease try again."
            )
        except Exception:
            pass
    except ValueError as error:
        try:
            await status.edit_text(f"❌ {error}")
        except Exception:
            pass
    except Exception as error:
        logger.exception("Solve failed: %s", error)
        try:
            await status.edit_text(
                "❌ <b>Failed to generate the solution.</b>\n\n"
                f"<code>{html.escape(str(error))}</code>",
                parse_mode=ParseMode.HTML,
            )
        except Exception:
            pass


# ============================================================
# COMMANDS
# ============================================================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = """
🤖 <b>Cub • AI Question Solver</b>

Send me a picture of a question.

Cub will:
🔍 Read the question
🧠 Solve it step by step
📐 Typeset proper mathematics
🖼 Send a clean solution sheet

<b>Default output:</b> Image

Reply to a question image with:
<code>/solve</code>

For text only:
<code>/solve text</code>

For image + text:
<code>/solve both</code>

You can also add instructions:
<code>/solve image solve question 3 only</code>
<code>/solve image explain in simple language</code>

/help - Full instructions
/model - Model information
/about - About Cub
/ping - Bot status
"""
    await update.effective_message.reply_text(
        text.strip(),
        parse_mode=ParseMode.HTML,
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = """
📚 <b>Cub Help</b>

Simply send a clear picture of a question.
The default result is a professionally typeset solution image.

<code>/solve</code> - Image solution
<code>/solve image</code> - Image solution
<code>/solve text</code> - Text solution
<code>/solve both</code> - Image + text

Examples:
<code>/solve image solve only question 5</code>
<code>/solve image explain every calculation</code>
<code>/solve both use an easy method</code>

Cub always solves the question step by step.
"""
    await update.effective_message.reply_text(
        text.strip(),
        parse_mode=ParseMode.HTML,
    )


async def solve_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.effective_message
    if not message:
        return

    mode, instruction = parse_output_mode(context.args)
    replied = message.reply_to_message

    if replied and message_has_image(replied):
        await solve_message(
            update=update,
            source_message=replied,
            output_mode=mode,
            extra_instruction=instruction,
        )
        return

    await message.reply_text(
        "📸 <b>Reply to a question image.</b>\n\n"
        "Example:\n\n<code>/solve</code>\n\nor\n\n<code>/solve both</code>",
        parse_mode=ParseMode.HTML,
    )


async def image_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.effective_message
    if not message:
        return

    caption = (message.caption or "").strip()
    mode = DEFAULT_OUTPUT_MODE
    instruction = ""

    if caption:
        if caption.lower().startswith("/solve"):
            mode, instruction = parse_output_mode(caption.split()[1:])
        else:
            instruction = caption

    await solve_message(
        update=update,
        source_message=message,
        output_mode=mode,
        extra_instruction=instruction,
    )


async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = f"""
🤖 <b>Cub Model Pipeline</b>

<b>Solver</b>
<code>{html.escape(LIVE_MODEL)}</code>

<b>Math Formatter / Verifier</b>
<code>{html.escape(LATEX_MODEL)}</code>

<b>Renderer</b>
MathJax + Chromium

<b>Output</b>
High-resolution paginated PNG solution sheets.
"""
    await update.effective_message.reply_text(
        text.strip(),
        parse_mode=ParseMode.HTML,
    )


async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = """
🤖 <b>Cub</b>

A question-solving Telegram bot with professional mathematical typesetting.

Question Image → Gemini reasoning → Step-by-step verification → LaTeX → MathJax → High-resolution solution image
"""
    await update.effective_message.reply_text(
        text.strip(),
        parse_mode=ParseMode.HTML,
    )


async def ping_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    browser_status = "Ready" if BROWSER is not None else "Unavailable"
    await update.effective_message.reply_text(
        "🏓 <b>Pong!</b>\n\n"
        "✅ Bot: Online\n"
        f"✅ Renderer: {browser_status}",
        parse_mode=ParseMode.HTML,
    )


async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        "❓ Unknown command.\n\nUse /help for instructions."
    )


# ============================================================
# STARTUP / SHUTDOWN
# ============================================================

async def post_init(application: Application):
    global PLAYWRIGHT, BROWSER

    logger.info("Starting Chromium math renderer...")
    PLAYWRIGHT = await async_playwright().start()

    launch_options = {
        "headless": True,
        "args": [
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu",
        ],
    }

    if CHROMIUM_PATH and os.path.exists(CHROMIUM_PATH):
        launch_options["executable_path"] = CHROMIUM_PATH

    BROWSER = await PLAYWRIGHT.chromium.launch(**launch_options)

    await application.bot.set_my_commands(
        [
            BotCommand("start", "Start Cub"),
            BotCommand("solve", "Solve a question image"),
            BotCommand("help", "How to use Cub"),
            BotCommand("model", "Show model pipeline"),
            BotCommand("about", "About Cub"),
            BotCommand("ping", "Check bot status"),
        ]
    )

    logger.info("Chromium renderer ready.")


async def post_shutdown(application: Application):
    global PLAYWRIGHT, BROWSER

    if BROWSER is not None:
        try:
            await BROWSER.close()
        except Exception:
            pass
        BROWSER = None

    if PLAYWRIGHT is not None:
        try:
            await PLAYWRIGHT.stop()
        except Exception:
            pass
        PLAYWRIGHT = None

    try:
        await gemini.aio.aclose()
    except Exception:
        pass


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    error = context.error
    if not error:
        return

    logger.error(
        "Telegram error: %s",
        error,
        exc_info=(type(error), error, error.__traceback__),
    )


# ============================================================
# BOT RUNNER
# ============================================================

def run_bot():
    application = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("solve", solve_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("model", model_command))
    application.add_handler(CommandHandler("about", about_command))
    application.add_handler(CommandHandler("ping", ping_command))
    application.add_handler(
        MessageHandler(filters.PHOTO | filters.Document.IMAGE, image_handler)
    )
    application.add_handler(MessageHandler(filters.COMMAND, unknown_command))
    application.add_error_handler(error_handler)

    print()
    print("================================================")
    print(" CUB • AI QUESTION SOLVER")
    print("================================================")
    print(f" Solver      : {LIVE_MODEL}")
    print(f" Math Editor : {LATEX_MODEL}")
    print(" Renderer    : MathJax + Chromium")
    print(" Output      : Professional PNG")
    print(" Steps       : Always enabled")
    print(" Status      : ONLINE")
    print("================================================")
    print(" Press Ctrl+C to stop")
    print("================================================")
    print()

    application.run_polling(
        allowed_updates=Update.ALL_TYPES,
        stop_signals=None,
        close_loop=False,
    )


if __name__ == "__main__":
    run_bot()
