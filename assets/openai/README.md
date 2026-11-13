# OpenAI encodings

The rank files of OpenAI's `o200k_base`, `cl100k_base` and `r50k_base` encodings, as published for [tiktoken](https://github.com/openai/tiktoken) (MIT license) at `https://openaipublic.blob.core.windows.net/encodings/<name>.tiktoken`. The SHA-256 of each file matches the hash tiktoken pins:

| File | SHA-256 |
| --- | --- |
| `o200k_base.tiktoken` | `446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d` |
| `cl100k_base.tiktoken` | `223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7` |
| `r50k_base.tiktoken` | `306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930` |

The split pattern and special tokens of each encoding are in `src/source.rs`.
