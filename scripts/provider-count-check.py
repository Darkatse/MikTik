#!/usr/bin/env python3
"""Compares a provider's own token counts with a MikTik row on the public-domain texts.

Usage: scripts/provider-count-check.py <row> <provider> <model>...

Providers and the environment variable each one's API key is read from:

    gemini     GEMINI_API_KEY       countTokens
    anthropic  ANTHROPIC_API_KEY    messages/count_tokens
    openai     OPENAI_API_KEY       responses/input_tokens
    xai        XAI_API_KEY          tokenize-text
    moonshot   MOONSHOT_API_KEY     tokenizers/estimate-token-count (api.moonshot.ai)
    dashscope  DASHSCOPE_API_KEY    chat completion, thinking off
    zhipu      ZHIPU_API_KEY        chat completion (open.bigmodel.cn)
    minimax    MINIMAX_API_KEY      chat completion (api.minimaxi.com)
    deepseek   DEEPSEEK_API_KEY     chat completion

Run scripts/fetch-test-assets.sh first. Each text is cut into 20 pieces of 3,000 characters, stripped
of whitespace at their edges since some chat templates trim it, and the provider and MikTik (exact
and estimate) count every piece. A provider adds a fixed number of
tokens to every request, for its chat template or nothing; counting "a" measures them, and they are
subtracted. A row serves a model exactly when its exact count equals the provider's on every
piece. Providers without a counting endpoint are asked for one output token: about 40 requests and
60,000 input tokens per model. Only the public-domain texts are sent, and the key only as a request
header; it is never printed.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPORA = ("en", "zh")
PIECES, PIECE_CHARS = 20, 3000


def request(url, headers, body):
    data = json.dumps(body).encode()
    for attempt in range(7):
        req = urllib.request.Request(url, data=data, headers={"content-type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=300) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code in (429, 500, 502, 503) and attempt < 6:
                time.sleep(2**attempt)
                continue
            raise SystemExit(f"{url}: HTTP {error.code}: {error.read()[:300].decode(errors='replace')}")
    raise AssertionError("unreachable")


def bearer(key):
    return {"authorization": f"Bearer {key}"}


def chat(url, extra=None):
    def count(key, model, text):
        body = {"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": 1, **(extra or {})}
        return request(url, bearer(key), body)["usage"]["prompt_tokens"]

    return count


def gemini(key, model, text):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:countTokens"
    body = {"contents": [{"role": "user", "parts": [{"text": text}]}]}
    return request(url, {"x-goog-api-key": key}, body)["totalTokens"]


def anthropic(key, model, text):
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
    body = {"model": model, "messages": [{"role": "user", "content": text}]}
    return request("https://api.anthropic.com/v1/messages/count_tokens", headers, body)["input_tokens"]


def openai(key, model, text):
    body = {"model": model, "input": text}
    return request("https://api.openai.com/v1/responses/input_tokens", bearer(key), body)["input_tokens"]


def xai(key, model, text):
    body = {"model": model, "text": text}
    return len(request("https://api.x.ai/v1/tokenize-text", bearer(key), body)["token_ids"])


def moonshot(key, model, text):
    time.sleep(21)  # Three requests a minute.
    body = {"model": model, "messages": [{"role": "user", "content": text}]}
    url = "https://api.moonshot.ai/v1/tokenizers/estimate-token-count"
    return request(url, bearer(key), body)["data"]["total_tokens"]


PROVIDERS = {
    "gemini": ("GEMINI_API_KEY", gemini),
    "anthropic": ("ANTHROPIC_API_KEY", anthropic),
    "openai": ("OPENAI_API_KEY", openai),
    "xai": ("XAI_API_KEY", xai),
    "moonshot": ("MOONSHOT_API_KEY", moonshot),
    "dashscope": (
        "DASHSCOPE_API_KEY",
        chat("https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions", {"enable_thinking": False}),
    ),
    "zhipu": ("ZHIPU_API_KEY", chat("https://open.bigmodel.cn/api/paas/v4/chat/completions")),
    "minimax": ("MINIMAX_API_KEY", chat("https://api.minimaxi.com/v1/chat/completions")),
    "deepseek": ("DEEPSEEK_API_KEY", chat("https://api.deepseek.com/chat/completions")),
}


def pieces(text):
    step = len(text) // PIECES
    return [text[i * step : i * step + PIECE_CHARS].strip() for i in range(PIECES)]


def source_args(row):
    """`--source` and the row's file in .local/models, unless the row is built in."""
    directory = os.path.join(ROOT, ".local/models")
    files = [name for name in os.listdir(directory) if os.path.splitext(name)[0] == row]
    return ["--source", os.path.join(directory, files[0])] if files else []


def miktik_counts(row, texts):
    """Exact and estimated counts of each text with the row."""
    with tempfile.TemporaryDirectory() as directory:
        paths = []
        for index, text in enumerate(texts):
            path = os.path.join(directory, f"{index}.txt")
            with open(path, "w", encoding="utf-8") as file:
                file.write(text)
            paths.append(path)
        output = subprocess.run(
            ["cargo", "run", "--quiet", "--release", "--all-features", "--example", "count", "--", row,
             *source_args(row), *paths],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout
    return [tuple(map(int, line.split()[:2])) for line in output.splitlines()]


def percent(value, reference):
    return (value - reference) / reference * 100


def main():
    if len(sys.argv) < 4 or sys.argv[2] not in PROVIDERS:
        raise SystemExit(__doc__)
    row, provider, models = sys.argv[1], sys.argv[2], sys.argv[3:]
    variable, count = PROVIDERS[provider]
    key = os.environ.get(variable)
    if not key:
        raise SystemExit(f"{variable} is not set")
    texts = {}
    for corpus in CORPORA:
        with open(os.path.join(ROOT, ".local/corpora", f"{corpus}.txt"), encoding="utf-8") as file:
            texts[corpus] = pieces(file.read().replace("\r\n", "\n"))
    local = {corpus: miktik_counts(row, parts) for corpus, parts in texts.items()}

    for model in models:
        overhead = count(key, model, "a") - 1
        print(f"{model}: {overhead} tokens added to every request")
        for corpus, parts in texts.items():
            remote = [count(key, model, part) - overhead for part in parts]
            exact = [e for e, _ in local[corpus]]
            estimate = [e for _, e in local[corpus]]
            differences = sorted(e - r for e, r in zip(exact, remote) if e != r)
            worst = max(abs(percent(e, r)) for e, r in zip(estimate, remote))
            print(
                f"  {corpus}: {provider} {sum(remote)}, exact {sum(exact)} ({percent(sum(exact), sum(remote)):+.2f}%,"
                f" differs on {len(differences)} of {len(parts)} pieces{': ' if differences else ''}"
                f"{' '.join(f'{d:+d}' for d in differences)}), estimate {sum(estimate)}"
                f" ({percent(sum(estimate), sum(remote)):+.2f}%, largest piece difference {worst:.2f}%)"
            )


if __name__ == "__main__":
    main()
