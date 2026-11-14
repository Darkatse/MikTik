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
and estimate) count every piece as a user message. A provider adds a fixed number of tokens to every
request, for its chat template or nothing; counting "a" measures them, and they are subtracted. A row
serves a model exactly when its exact count equals the provider's on every piece.

Then the framing: three pieces as a user, an assistant and a user message, against the first alone.
What the provider adds for the assistant and the user message is compared with the row's
count_messages; they differ by at most one, where the row rounds an odd pair up. The difference of
the one-message requests is what the provider adds to every request beyond the row, such as a
default system prompt. Both include any difference in the pieces' own counts, so they measure
framing only for a row that counts the pieces exactly. xAI counts text only, so its models get no
framing check.

Providers without a counting endpoint are asked for one output token: about 45 requests and 70,000
input tokens per model. Only the public-domain texts are sent, and the key only as a request header;
it is never printed.
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
        except urllib.error.URLError as error:
            if attempt < 6:
                time.sleep(2**attempt)
                continue
            raise SystemExit(f"{url}: {error.reason}")
    raise AssertionError("unreachable")


def bearer(key):
    return {"authorization": f"Bearer {key}"}


# Every count function takes a list of messages: {"role": "user" or "assistant", "content": text}.


def chat(url, extra=None):
    def count(key, model, messages):
        body = {"model": model, "messages": messages, "max_tokens": 1, **(extra or {})}
        return request(url, bearer(key), body)["usage"]["prompt_tokens"]

    return count


def gemini(key, model, messages):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:countTokens"
    roles = {"user": "user", "assistant": "model"}
    contents = [{"role": roles[message["role"]], "parts": [{"text": message["content"]}]} for message in messages]
    return request(url, {"x-goog-api-key": key}, {"contents": contents})["totalTokens"]


def anthropic(key, model, messages):
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
    body = {"model": model, "messages": messages}
    return request("https://api.anthropic.com/v1/messages/count_tokens", headers, body)["input_tokens"]


def openai(key, model, messages):
    body = {"model": model, "input": messages}
    return request("https://api.openai.com/v1/responses/input_tokens", bearer(key), body)["input_tokens"]


def xai(key, model, messages):
    [message] = messages  # Counts text only.
    body = {"model": model, "text": message["content"]}
    return len(request("https://api.x.ai/v1/tokenize-text", bearer(key), body)["token_ids"])


def moonshot(key, model, messages):
    body = {"model": model, "messages": messages}
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
# Providers that count text, not messages.
TEXT_ONLY = {"xai"}


def user(text):
    return {"role": "user", "content": text}


def assistant(text):
    return {"role": "assistant", "content": text}


def pieces(text):
    step = len(text) // PIECES
    return [text[i * step : i * step + PIECE_CHARS].strip() for i in range(PIECES)]


def source_args(row):
    """`--source` and the row's file in .local/models, unless the row is built in."""
    directory = os.path.join(ROOT, ".local/models")
    files = [name for name in os.listdir(directory) if os.path.splitext(name)[0] == row]
    return ["--source", os.path.join(directory, files[0])] if files else []


def miktik_counts(row, inputs):
    """Exact and estimated counts with the row of each input: a text, or a list of messages."""
    with tempfile.TemporaryDirectory() as directory:
        paths = []
        for index, value in enumerate(inputs):
            path = os.path.join(directory, f"{index}.txt" if isinstance(value, str) else f"{index}.json")
            with open(path, "w", encoding="utf-8") as file:
                file.write(value if isinstance(value, str) else json.dumps(value))
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
    first, second, third = texts["en"][:3]
    one, three = [user(first)], [user(first), assistant(second), user(third)]
    (local_one, _), (local_three, _) = miktik_counts(row, [one, three])

    for model in models:
        overhead = count(key, model, [user("a")]) - 1
        print(f"{model}: {overhead} tokens added to every request")
        for corpus, parts in texts.items():
            remote = [count(key, model, [user(part)]) - overhead for part in parts]
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
        if provider in TEXT_ONLY:
            continue
        remote_one, remote_three = count(key, model, one), count(key, model, three)
        pair, local_pair = remote_three - remote_one, local_three - local_one
        print(
            f"  framing: {provider} adds {pair} tokens for an assistant and a user message, the row"
            f" {local_pair} ({local_pair - pair:+d}); {remote_one - local_one:+d} to every request beyond the row"
        )


if __name__ == "__main__":
    main()
