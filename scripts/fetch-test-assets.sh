#!/usr/bin/env bash
# Fetches what tests/corpus.rs and examples/throughput.rs read into .local/: a source for every model
# that needs one, named after the model id, and two public-domain books. Every file is pinned by its
# SHA-256; files already present and intact are kept.
set -euo pipefail
cd "$(dirname "$0")/.."

sillytavern=https://raw.githubusercontent.com/SillyTavern/SillyTavern/7e8663cd9c184a550b37238218bdd32c6efc68e9/src/tokenizers
web_tokenizers=https://raw.githubusercontent.com/SillyTavern/SillyTavern-Tokenizers/798bcd823dfd04227a79b487e1cf29e6de9aaece
kimi=https://huggingface.co/moonshotai/Kimi-K2-Instruct/resolve/fd1984e2b7a3350dbf7305fe73a4ede25c14de50
gemma3=https://huggingface.co/unsloth/gemma-3-1b-it/resolve/5b11413a10db4e486ef16a20101fd028f8f2499c
qwen3_5=https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/2fc06364715b967f1860aea9cf38778875588b17
glm=https://huggingface.co/zai-org/GLM-5.2/resolve/cf457fa734ab149ffef225f80893eb38c6ff5cdc
minimax=https://huggingface.co/MiniMaxAI/MiniMax-M3/resolve/d95e30cb912fe3021bb9c2ba66a8ce60b6c937a9
gutenberg=https://www.gutenberg.org/cache/epub

sha256() {
    if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1
}

# fetch <path> <url> <sha256>: a `.gz` URL is decompressed, and the hash is of the result.
fetch() {
    local path=$1 url=$2 expected=$3
    if [ -f "$path" ] && [ "$(sha256 "$path")" = "$expected" ]; then
        return
    fi
    mkdir -p "$(dirname "$path")"
    echo "fetching $path"
    if [[ $url == *.gz ]]; then
        curl -fsSL "$url" | gunzip >"$path.part"
    else
        curl -fsSL "$url" -o "$path.part"
    fi
    local actual
    actual=$(sha256 "$path.part")
    if [ "$actual" != "$expected" ]; then
        rm "$path.part"
        echo "$path: expected SHA-256 $expected, got $actual" >&2
        exit 1
    fi
    mv "$path.part" "$path"
}

models=.local/models
fetch $models/claude.json $sillytavern/claude.json c241737df24b4e7f7c9af4fdcee29a0ca903dcb288a8b753bc346a3092911767
fetch $models/llama3.json $sillytavern/llama3.json 7a413fb6f10a3e285a400d7cc9b8959c32116caf81ba8a60a4ec26b9aca2bd25
fetch $models/command-r.json $web_tokenizers/command-r.json.gz 23cf885a64b0f790f8da6ee36a47eef6005c9231287ee1ad22463a34d7b092d2
fetch $models/command-a.json $web_tokenizers/command-a.json.gz c6a9a15bf2175a05fcbe58f1242678bba7d2afcef1138a54983bcbd1745e5d12
fetch $models/qwen2.json $web_tokenizers/qwen2.json.gz fb366a22efb7db2dcca2c619402431cda342a97721395a2da2879ceb36262782
fetch $models/nemo.json $web_tokenizers/nemo.json.gz d27c2739d2d9ad785baa78f62ed23e79984246cb3e29c25d6cb5f43d9d00a6a9
fetch $models/deepseek.json $web_tokenizers/deepseek.json.gz a8d67ac5efa0def0716d813ccd9b288eaae51654221eb30112fb603cf241b247
fetch $models/qwen3.5.json $qwen3_5/tokenizer.json 5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42
fetch $models/glm.json $glm/tokenizer.json 19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d
fetch $models/minimax.json $minimax/tokenizer.json bb1f1626cf01448f1e3b6036d0a061ffc66c91d9046aada14ea23a5441b5ad6e
# Kimi K2 and K3 share a tokenizer and frame messages differently.
fetch $models/kimi.tiktoken $kimi/tiktoken.model b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103
fetch $models/kimi-k3.tiktoken $kimi/tiktoken.model b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103
fetch $models/llama.model $sillytavern/llama.model 9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347
fetch $models/mistral.model $sillytavern/mistral.model dadfd56d766715c61d2ef780a525ab43b8e6da4de6865bda3d95fdef5e134055
fetch $models/yi.model $sillytavern/yi.model 386c49cf943d71aa110361135338c50e38beeff0a66593480421f37b319e1a39
# Gemma 3's tokenizer, which Gemini 3 shares; each row reads a file named after it.
fetch $models/gemma.model $gemma3/tokenizer.model 1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c
fetch $models/gemini.model $gemma3/tokenizer.model 1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c
fetch $models/jamba.model $sillytavern/jamba.model 8b0df4fb43262c452ef37061951a06df4c63ca191d02a60ea08f14428af24376
fetch $models/nerdstash.model $sillytavern/nerdstash.model 578fa0ed4d6dbee435f21d7f7a741506d09cdd93cce241008abf725407cbdb41
fetch $models/nerdstash_v2.model $sillytavern/nerdstash_v2.model 005ad680b10f1abd406bdb0ca9c6a5d83fc1f6e0a855bdd1942c1ceab1fb47ab

corpora=.local/corpora
# Pride and Prejudice, and Dream of the Red Chamber, from Project Gutenberg.
fetch $corpora/en.txt $gutenberg/1342/pg1342.txt 3f6bb9d6f78e0293b56acd4714dd68cb7d6d1d293402031ce9d5a216bcaf9d75
fetch $corpora/zh.txt $gutenberg/24264/pg24264.txt ff1526996bf4b81807651921a85e5c1c0f1d1d123c9fa4553057ba6a3ec72011

echo "MIKTIK_MODEL_DIR=$PWD/$models MIKTIK_CORPUS_DIR=$PWD/$corpora"
