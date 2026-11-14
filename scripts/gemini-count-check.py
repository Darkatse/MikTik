#!/usr/bin/env python3
"""Compares Gemini's countTokens with MikTik's `gemma` row on the public-domain texts.

Usage: GEMINI_API_KEY=... scripts/gemini-count-check.py <Gemini model>...

Run scripts/fetch-test-assets.sh first. For each text the script counts the whole text and 20
pieces of 3,000 characters, with Gemini and with MikTik (exact and estimate). Gemini adds a fixed
number of tokens to every request, measured here by counting "a"; it is subtracted before the
comparison. The key is sent only as a request header and never printed.
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


def gemini_count(key, model, text):
    body = json.dumps({"contents": [{"role": "user", "parts": [{"text": text}]}]}).encode()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:countTokens"
    for attempt in range(6):
        request = urllib.request.Request(
            url, data=body, headers={"x-goog-api-key": key, "content-type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return json.load(response)["totalTokens"]
        except urllib.error.HTTPError as error:
            if error.code in (429, 500, 503):
                time.sleep(2**attempt)
                continue
            raise SystemExit(f"{model}: HTTP {error.code}: {error.read()[:200].decode(errors='replace')}")
    raise SystemExit(f"{model}: gave up after repeated rate limits")


def pieces(text):
    step = len(text) // PIECES
    return [text[i * step : i * step + PIECE_CHARS] for i in range(PIECES)]


def miktik_counts(texts):
    """Exact and estimated counts of each text with the `gemma` row."""
    with tempfile.TemporaryDirectory() as directory:
        files = []
        for index, text in enumerate(texts):
            path = os.path.join(directory, f"{index}.txt")
            with open(path, "w", encoding="utf-8") as file:
                file.write(text)
            files.append(path)
        output = subprocess.run(
            ["cargo", "run", "--quiet", "--release", "--features", "sentencepiece", "--example", "count",
             "--", "gemma", "--source", os.path.join(ROOT, ".local/models/gemma.model"), *files],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout
    return [tuple(map(int, line.split()[:2])) for line in output.splitlines()]


def percent(value, reference):
    return (value - reference) / reference * 100


def main():
    models = sys.argv[1:]
    key = os.environ.get("GEMINI_API_KEY")
    if not models or not key:
        raise SystemExit(__doc__)
    texts = {}
    for corpus in CORPORA:
        with open(os.path.join(ROOT, ".local/corpora", f"{corpus}.txt"), encoding="utf-8") as file:
            texts[corpus] = file.read()
    local = {corpus: miktik_counts([text, *pieces(text)]) for corpus, text in texts.items()}

    for model in models:
        overhead = gemini_count(key, model, "a") - 1
        print(f"{model}: {overhead} tokens added to every request")
        for corpus, text in texts.items():
            remote = [gemini_count(key, model, part) - overhead for part in [text, *pieces(text)]]
            (exact, estimate), *piece_counts = local[corpus]
            worst = max(abs(percent(e, r)) for (e, _), r in zip(piece_counts, remote[1:]))
            print(
                f"  {corpus}: Gemini {remote[0]}, exact {exact} ({percent(exact, remote[0]):+.2f}%),"
                f" estimate {estimate} ({percent(estimate, remote[0]):+.2f}%),"
                f" largest exact difference on a piece {worst:.2f}%"
            )


if __name__ == "__main__":
    main()
