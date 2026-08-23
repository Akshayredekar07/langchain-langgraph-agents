import os
import sys
import time
from typing import cast

from dotenv import load_dotenv
load_dotenv()

import litellm
from litellm import ModelResponse
from litellm.router import Router

litellm.suppress_debug_info = True

API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not API_KEY:
    print("OPENROUTER_API_KEY not set")
    sys.exit(1)

router = Router(
    model_list=[{
        "model_name": "smart",
        "litellm_params": {
            "model": "openrouter/stealth/ox-alpha",
            "api_key": API_KEY,
        },
    }],
    num_retries=1,
    timeout=60,
)


def ask(prompt: str) -> str:
    stream = router.completion(
        model="smart",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        max_tokens=800,
    )

    chunks = []
    for raw_chunk in stream:
        chunk = cast(ModelResponse, raw_chunk)
        choices = getattr(chunk, "choices", None)
        if not choices:
            continue
        delta = getattr(choices[0], "delta", None)
        content = getattr(delta, "content", None) if delta else None
        if content:
            print(content, end="", flush=True)
            chunks.append(content)

    answer = "".join(chunks).strip()

    if not answer:
        resp = cast(ModelResponse, router.completion(
            model="smart",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
        ))
        msg = resp.choices[0].message
        answer = (getattr(msg, "content", None) or "").strip()
        print(answer)

    return answer


if __name__ == "__main__":
    prompt = input("> ").strip()
    if not prompt:
        print("Empty question. Exiting.")
        sys.exit(0)

    start = time.time()
    ask(prompt)
    print(f"\n\n[{time.time() - start:.1f}s]")