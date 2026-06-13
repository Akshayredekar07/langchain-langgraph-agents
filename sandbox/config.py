import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

# Load .env once here too, in case config.py is imported before agent.py does it
env_file = os.getenv("ENV_FILE")
load_dotenv(env_file or find_dotenv(".env", usecwd=True))
load_dotenv(override=False)


class Config:
    # ── Anthropic / model ──────────────────────────────────────────────
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    ANTHROPIC_MODEL = os.getenv("CLAUDE_MODEL") or os.getenv("ANTHROPIC_MODEL") or "claude-sonnet-4-6"

    # ── Sandbox / file generation (PDF, DOCX, PPTX, SVG, etc) ───────────
    BASE_DIR = Path(__file__).resolve().parent
    SANDBOX_DIR = os.getenv("SANDBOX_DIR", str(BASE_DIR / "output"))

    # Ensure the sandbox directory exists at import time
    Path(SANDBOX_DIR).mkdir(parents=True, exist_ok=True)

    # ── Node executor settings (for GSAPSVGCreator) ─────────────────────
    NODE_TIMEOUT_SECONDS = int(os.getenv("NODE_TIMEOUT_SECONDS", "20"))