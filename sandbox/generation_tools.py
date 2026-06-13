# ── GSAP SVG Diagram / Animation Generator ────────────────────────────────
#
# Requires (one-time setup in Config.SANDBOX_DIR):
#   cd <SANDBOX_DIR> && npm install jsdom gsap
#
# Produces:
#   - .svg  -> raw static SVG file (architecture diagrams, flowcharts, icons, charts)
#   - .html -> SVG wrapped in HTML + GSAP timeline (with plugins) for animated diagrams
#
# Node is used to (a) validate the SVG via jsdom (catch malformed markup before
# writing the file) and (b) sanity-check any GSAP animation code by running it
# against the jsdom DOM so syntax/reference errors surface immediately.
#
# NOTE: jsdom has no real layout engine (no getBoundingClientRect/getComputedStyle
# results), so plugins like ScrollTrigger/MotionPath/SplitText/Flip may throw
# layout-related errors during the pre-check even though they work fine in a
# real browser. Such errors are treated as WARNINGS (file is still written),
# while syntax errors / unknown identifiers are treated as hard failures.

import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from typing import Optional

from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from config import Config

logger = logging.getLogger("generation_tools")

NODE_TIMEOUT = int(getattr(Config, "NODE_TIMEOUT_SECONDS", 20))

# Plugins loaded into every animated HTML output. GSAP (incl. these plugins)
# is free as of the 2025 license change.
GSAP_PLUGINS_CDN = [
    "https://cdn.jsdelivr.net/npm/gsap@3.12.5/dist/gsap.min.js",
    "https://cdn.jsdelivr.net/npm/gsap@3.12.5/dist/ScrollTrigger.min.js",
    "https://cdn.jsdelivr.net/npm/gsap@3.12.5/dist/MotionPathPlugin.min.js",
    "https://cdn.jsdelivr.net/npm/gsap@3.12.5/dist/SplitText.min.js",
    "https://cdn.jsdelivr.net/npm/gsap@3.12.5/dist/Draggable.min.js",
    "https://cdn.jsdelivr.net/npm/gsap@3.12.5/dist/Flip.min.js",
    "https://cdn.jsdelivr.net/npm/gsap@3.12.5/dist/CustomEase.min.js",
]

# Errors from these plugins/features are common & expected in jsdom (no real
# layout engine) — treat as warnings rather than failures.
LENIENT_ERROR_MARKERS = (
    "ScrollTrigger",
    "MotionPathPlugin",
    "SplitText",
    "Draggable",
    "Flip",
    "getBoundingClientRect",
    "getComputedStyle",
    "MotionPath",
    "CustomEase",
)


def node_available() -> bool:
    return shutil.which("node") is not None


def run_node_script(js_code: str, timeout: int = NODE_TIMEOUT) -> tuple[str, str, int]:
    """Write js_code to a temp file in the sandbox and run it with node."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".js", delete=False, dir=Config.SANDBOX_DIR, encoding="utf-8"
    ) as f:
        f.write(js_code)
        script_path = f.name
    try:
        result = subprocess.run(
            ["node", script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Config.SANDBOX_DIR,
            encoding="utf-8",
        )
        return result.stdout, result.stderr, result.returncode
    finally:
        os.unlink(script_path)


# ── Pydantic schema ───────────────────────────────────────────────────────

class GSAPSVGInput(BaseModel):
    svg_content: str = Field(
        description=(
            "Full raw SVG markup for the diagram, starting with <svg ...> and ending with "
            "</svg>. Use this for ANY diagram type: architecture/system diagrams, flowcharts, "
            "network topologies, sequence diagrams, org charts, icons, data charts, card "
            "decks, dashboards, etc. Give elements id or class attributes (e.g. "
            "<rect id='box1' .../>, <path class='arrow' .../>) so they can be targeted by "
            "GSAP animations. For motion-path effects, include a hidden guide <path> with "
            "its own id. For text reveals, wrap text in <text id='headline'>."
        )
    )
    gsap_animation_js: Optional[str] = Field(
        default=None,
        description=(
            "Optional GSAP animation code (JS) that animates the SVG elements by id/class. "
            "The following plugins are pre-loaded and registered: ScrollTrigger, "
            "MotionPathPlugin, SplitText, Draggable, Flip, CustomEase — use them freely "
            "for richer effects (motion paths, text-character reveals, draggable nodes, "
            "Flip transitions, custom eases). If omitted, a static SVG file is produced. "
            "If provided, the output is an HTML file embedding the SVG with GSAP + plugins "
            "loaded from CDN and this code run on load."
        ),
    )
    filename: str = Field(
        default="diagram.svg",
        description=(
            "Output filename. Use .svg for static diagrams (no gsap_animation_js) and "
            ".html for animated ones (gsap_animation_js provided)."
        ),
    )
    title: str = Field(default="SVG Diagram", description="Title used in the generated HTML page (animated output only).")


# ── tool implementation ──────────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>{title}</title>
{plugin_scripts}
<style>
  html, body {{ margin:0; padding:0; background:#0f1117; display:flex;
                justify-content:center; align-items:center; min-height:100vh;
                font-family: 'Segoe UI', Arial, sans-serif; overflow:hidden; }}
  svg {{ max-width:95vw; max-height:95vh; }}
</style>
</head>
<body>
{svg_content}
<script>
gsap.registerPlugin(ScrollTrigger, MotionPathPlugin, SplitText, Draggable, Flip, CustomEase);
{gsap_js}
</script>
</body>
</html>
"""


def gsap_svg_creator_fn(
    svg_content: str,
    gsap_animation_js: Optional[str] = None,
    filename: str = "diagram.svg",
    title: str = "SVG Diagram",
) -> str:
    svg_content = svg_content.strip()

    if not svg_content.startswith("<svg"):
        return "Invalid input: svg_content must start with an <svg ...> tag."

    if not node_available():
        return (
            "GSAPSVGCreator failed: Node.js is not installed or not on PATH. "
            "Install Node.js and run `npm install jsdom gsap` in "
            f"{Config.SANDBOX_DIR} to enable this tool."
        )

    out_path = os.path.join(Config.SANDBOX_DIR, f"{uuid.uuid4().hex}_{filename}")

    # --- Static SVG: validate with jsdom, then write file as-is ---
    if not gsap_animation_js:
        validation_js = f"""
const {{ JSDOM }} = require("jsdom");
const svg = {svg_content!r};
try {{
  const dom = new JSDOM(`<!DOCTYPE html><body>${{svg}}</body>`);
  const el = dom.window.document.querySelector("svg");
  if (!el) {{ console.error("NO_SVG_ELEMENT"); process.exit(1); }}
  console.log("OK");
}} catch (e) {{
  console.error("PARSE_ERROR: " + e.message);
  process.exit(1);
}}
"""
        stdout, stderr, code = run_node_script(validation_js)
        if code != 0:
            logger.warning("GSAPSVGCreator validation failed: %s", stderr.strip())
            return f"GSAPSVGCreator failed: invalid SVG markup. Details: {stderr.strip()[:300]}"

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
        return f"Static SVG file generated: {out_path}"

    # --- Animated SVG: validate svg + gsap code (with plugins) via jsdom ---
    check_js = f"""
const {{ JSDOM }} = require("jsdom");
const gsapCore = require("gsap");
const {{ MotionPathPlugin }} = require("gsap/MotionPathPlugin");
const {{ Draggable }} = require("gsap/Draggable");
const {{ Flip }} = require("gsap/Flip");
const {{ CustomEase }} = require("gsap/CustomEase");
let SplitText, ScrollTrigger;
try {{ SplitText = require("gsap/SplitText").SplitText; }} catch (e) {{ SplitText = function(){{}}; }}
try {{ ScrollTrigger = require("gsap/ScrollTrigger").ScrollTrigger; }} catch (e) {{ ScrollTrigger = {{}}; }}

const svg = {svg_content!r};
const animCode = {gsap_animation_js!r};

try {{
  const dom = new JSDOM(`<!DOCTYPE html><body>${{svg}}</body>`, {{ runScripts: "outside-only" }});
  global.window = dom.window;
  global.document = dom.window.document;

  const gsap = gsapCore.gsap || gsapCore;
  global.gsap = gsap;
  global.MotionPathPlugin = MotionPathPlugin;
  global.Draggable = Draggable;
  global.Flip = Flip;
  global.CustomEase = CustomEase;
  global.SplitText = SplitText;
  global.ScrollTrigger = ScrollTrigger;

  try {{ gsap.registerPlugin(MotionPathPlugin, Draggable, Flip, CustomEase); }} catch (e) {{}}

  const el = document.querySelector("svg");
  if (!el) {{ console.error("NO_SVG_ELEMENT"); process.exit(1); }}

  // Run the user's GSAP code against the jsdom-backed SVG to catch syntax/reference errors
  eval(animCode);
  console.log("OK");
}} catch (e) {{
  console.error("ANIM_ERROR: " + e.message);
  process.exit(1);
}}
"""
    stdout, stderr, code = run_node_script(check_js)
    if code != 0:
        err = stderr.strip()
        if any(marker in err for marker in LENIENT_ERROR_MARKERS):
            logger.warning(
                "GSAPSVGCreator: pre-check raised a layout/plugin-related error "
                "(expected in jsdom, likely fine in a real browser): %s", err[:300]
            )
        else:
            logger.warning("GSAPSVGCreator animation check failed: %s", err)
            return f"GSAPSVGCreator failed: GSAP animation error. Details: {err[:300]}"

    plugin_scripts = "\n".join(f'<script src="{url}"></script>' for url in GSAP_PLUGINS_CDN)

    html = HTML_TEMPLATE.format(
        title=title,
        plugin_scripts=plugin_scripts,
        svg_content=svg_content,
        gsap_js=gsap_animation_js,
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return f"Animated SVG (HTML) file generated: {out_path}\nOpen in a browser to view the GSAP animation."


gsap_svg_creator = StructuredTool.from_function(
    func=gsap_svg_creator_fn,
    name="GSAPSVGCreator",
    description=(
        "Create SVG diagrams and animated visuals of any kind — architecture/system "
        "diagrams, flowcharts, network topologies, sequence/UML diagrams, org charts, "
        "icons, infographics, data dashboards, card decks, animated backgrounds — from "
        "raw SVG markup. Optionally animate elements using GSAP plus the ScrollTrigger, "
        "MotionPathPlugin, SplitText, Draggable, Flip, and CustomEase plugins (all "
        "pre-loaded and registered) by also providing gsap_animation_js. Static output is "
        ".svg; animated output is a self-contained .html file. Validates SVG/GSAP code via "
        "a Node.js sandbox before writing the file."
    ),
    args_schema=GSAPSVGInput,
    handle_tool_error=True,
)