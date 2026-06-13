import os
import sys
from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent
from langgraph.types import Overwrite

stdout_reconfigure = getattr(sys.stdout, "reconfigure", None)
if callable(stdout_reconfigure):
    stdout_reconfigure(encoding="utf-8", errors="replace")

env_file = os.getenv("ENV_FILE")
load_dotenv(env_file or find_dotenv(".env", usecwd=True))
load_dotenv(override=False)

from generation_tools import gsap_svg_creator, node_available

assert os.getenv("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY not set"

raw_anthropic_model = os.getenv("CLAUDE_MODEL") or os.getenv("ANTHROPIC_MODEL")
ANTHROPIC_MODEL = (
    raw_anthropic_model
    if raw_anthropic_model and raw_anthropic_model.startswith("claude-")
    else "claude-sonnet-4-6"
)

MODEL_TIMEOUT_SECONDS = int(os.getenv("MODEL_TIMEOUT_SECONDS", "600"))

model = init_chat_model(
    f"anthropic:{ANTHROPIC_MODEL}",
    timeout=MODEL_TIMEOUT_SECONDS,
    max_retries=2,
)

if not node_available():
    print(
        "[SVGAGENT] WARNING: Node.js not found on PATH. "
        "GSAPSVGCreator will fail until Node.js is installed and "
        "`npm install jsdom gsap` is run in Config.SANDBOX_DIR.",
        flush=True,
    )

GSAP_PATTERNS = """
GSAP TECHNIQUES AVAILABLE (use these for richer visuals, not just static box+arrow diagrams):

1. CARD DECK / STACK REVEAL
   - Stack multiple <g id="card-N"> elements with slight rotation/offset.
   - gsap.timeline().to("#card-1", {y: -40, rotation: -8, duration: 0.6, ease: "back.out(1.7)"})
     .to("#card-2", {y: -20, rotation: 4, duration: 0.6}, "<0.1")
     ... stagger with relative positions ("<0.1") for cascading reveals.

2. MOTION PATH (object follows a curved SVG path)
   - Define a hidden <path id="motion-path" d="..." fill="none"/>
   - gsap.to("#mover", {duration: 3, repeat: -1, ease: "none",
       motionPath: {path: "#motion-path", align: "#motion-path", autoRotate: true}})

3. TEXT REVEAL (SplitText)
   - Wrap title text in <text id="headline">...</text>
   - let split = new SplitText("#headline", {type: "chars"});
     gsap.from(split.chars, {opacity: 0, y: 20, stagger: 0.03, duration: 0.5})

4. ANIMATED GRADIENT BACKGROUND
   - Define <linearGradient id="bgGrad"> with 2-3 <stop> elements with ids.
   - gsap.to("#stop1", {attr: {offset: 0.8}, duration: 4, repeat: -1, yoyo: true, ease: "sine.inOut"})

5. LINE-DRAWING CONNECTORS (architecture diagrams)
   - stroke-dasharray = path length, stroke-dashoffset = same value initially.
   - gsap.to("#line", {strokeDashoffset: 0, duration: 1, ease: "power2.inOut"})

6. ENTRANCE STAGGER FOR MULTIPLE ELEMENTS
   - gsap.from(".node", {opacity: 0, scale: 0.6, stagger: 0.15, ease: "back.out(1.6)"})

7. INFINITE AMBIENT MOTION (subtle "alive" feel)
   - gsap.to("#glow", {filter: "brightness(1.15)", repeat: -1, yoyo: true, duration: 2, ease: "sine.inOut"})

8. DRAGGABLE ELEMENTS (interactive diagrams)
   - Draggable.create("#node1", {bounds: "svg", inertia: false})

9. FLIP TRANSITIONS (state-change animations, e.g. card expand/collapse)
   - const state = Flip.getState("#card1");
   - // change layout/attributes, then:
   - Flip.from(state, {duration: 0.6, ease: "power1.inOut"})

10. CUSTOM EASES (signature, branded motion feel)
   - CustomEase.create("custom", "M0,0 C0.25,0.46 0.45,0.94 1,1")
   - gsap.to("#el", {x: 100, ease: "custom", duration: 1})

Pre-registered plugins (use directly, no need to register): gsap core, ScrollTrigger,
MotionPathPlugin, SplitText, Draggable, Flip, CustomEase.
"""

agent = create_deep_agent(
    model=model,
    tools=[gsap_svg_creator],
    system_prompt=(
        "You are an SVG diagram and animation generation assistant.\n\n"
        "When the user asks for a diagram, illustration, chart, icon, card deck, dashboard, "
        "or any visual (architecture/system diagrams, flowcharts, network topologies, "
        "sequence/UML diagrams, org charts, infographics, data visualizations, animated "
        "backgrounds, etc.), call the GSAPSVGCreator tool with valid SVG markup.\n\n"
        "Guidelines:\n"
        "- Always produce complete, valid SVG markup starting with <svg ...> and ending "
        "with </svg>, including a viewBox attribute sized appropriately for the content.\n"
        "- Give elements meaningful id or class attributes so they can be targeted by "
        "animations or referenced later.\n"
        "- If the user does NOT ask for animation or motion, omit gsap_animation_js and "
        "produce a static .svg file.\n"
        "- If the user DOES ask for animation (e.g. 'animate this', 'show the flow step by "
        "step', 'entrance animation', 'looping motion', 'card deck', 'dashboard', "
        "'interactive'), provide gsap_animation_js and use a filename ending in .html.\n"
        "- Aim for visually impressive, modern output (gradients, glow filters, smooth "
        "easing, staggered reveals, ambient motion) — not plain flat shapes.\n"
        "- Choose a sensible default filename based on the diagram's subject.\n"
        "- After generating a file, tell the user the output path and, for .html files, "
        "remind them to open it in a browser to view the animation.\n\n"
        + GSAP_PATTERNS
    ),
)


def get_messages(node_data):
    if isinstance(node_data, Overwrite):
        node_data = node_data.value
    if isinstance(node_data, dict):
        msgs = node_data.get("messages", [])
    elif hasattr(node_data, "messages"):
        msgs = node_data.messages
    else:
        msgs = node_data
    if isinstance(msgs, Overwrite):
        msgs = msgs.value
    if isinstance(msgs, list):
        return msgs
    return []


def normalize_stream_chunk(chunk):
    if isinstance(chunk, dict):
        return chunk.get("type"), chunk.get("data")
    if isinstance(chunk, tuple):
        if len(chunk) == 3:
            _, chunk_type, data = chunk
            return chunk_type, data
        if len(chunk) == 2:
            chunk_type, data = chunk
            return chunk_type, data
    return None, None


def stream_agent(user_input: str):
    if raw_anthropic_model and raw_anthropic_model != ANTHROPIC_MODEL:
        print(f"[SVGAGENT] ignoring non-Claude ANTHROPIC_MODEL={raw_anthropic_model!r}", flush=True)
    print(f"[SVGAGENT] model=anthropic:{ANTHROPIC_MODEL}", flush=True)
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        stream_mode=["updates", "messages"],
        subgraphs=True,
        version="v2",
    ):
        chunk_type, data = normalize_stream_chunk(chunk)

        if chunk_type == "updates":
            for node_name, node_data in data.items():
                msgs = get_messages(node_data)
                if not msgs:
                    continue
                for msg in msgs:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            print(f"\n[TOOL CALL] {tc['name']}")
                            print(f"  args: {tc['args']}")
                    elif getattr(msg, "type", None) == "tool":
                        print(f"\n[TOOL RESULT] {msg.name}")
                        print(f"  {str(msg.content)[:300]}")

        elif chunk_type == "messages":
            token, _ = data
            if getattr(token, "content", None) and getattr(token, "type", None) == "ai":
                print(token.content, end="", flush=True)


# ── Example queries — uncomment the one you want to run ───────────────────

# query = (
#     "Create an animated SVG diagram showing a simple 3-tier web architecture: "
#     "Client -> API Gateway -> Backend Service -> Database. Animate each box and "
#     "arrow appearing in sequence with a glowing dark theme."
# )

# query = (
#     "Create an animated SVG card deck with 4 feature highlight cards (like a "
#     "feature showcase). Each card should slide and fan out from a stack with "
#     "a staggered, springy entrance animation. Dark background with glowing "
#     "accent borders."
# )

# query = (
#     "Create an animated SVG dashboard mockup with 3 stat cards and a line "
#     "chart. Animate the numbers counting up, the chart line drawing in, and "
#     "the cards fading in with a staggered entrance."
# )

# query = (
#     "Create an animated SVG icon of a satellite orbiting a planet along a "
#     "curved motion path, with autoRotate enabled, looping infinitely. Use a "
#     "dark space background with glowing stars."
# )

# query = (
#     "Create an animated SVG hero banner with a large title that reveals "
#     "character-by-character using a text split animation, an animated "
#     "gradient background that slowly shifts colors, and a subtle floating "
#     "icon with ambient pulsing glow."
# )


query = (
    "Create an animated SVG/HTML hero section with a heading that says "
    "'Build Accessible Animations' and a subheading 'Motion that respects everyone'. "
    "Animate the heading using a SplitText character reveal (staggered fade + slide up), "
    "but follow GSAP accessibility best practices: keep the real heading text in the DOM "
    "for screen readers (do not let SplitText destroy/hide the actual content from "
    "assistive tech), mark any cloned/duplicated animation elements with aria-hidden='true', "
    "and wrap the GSAP animation code so it checks "
    "window.matchMedia('(prefers-reduced-motion: reduce)') and skips the stagger/motion "
    "animation (just fades in instantly) if the user has reduced motion enabled. "
    "Use a dark background with a soft glowing accent color."
)

if __name__ == "__main__":
    stream_agent(query)
    print()