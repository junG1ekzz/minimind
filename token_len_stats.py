"""统计 MiniMind 数据集的 token 长度分布，看 max_seq_len 会截断多少样本。

用法（在仓库根目录 ~/minimind 下、激活环境后运行）：
    python token_length_stats.py --data dataset/pretrain_t2t_mini.jsonl --kind pretrain
    python token_length_stats.py --data dataset/sft_t2t_mini.jsonl --kind sft
"""
import argparse
import json
import random

import matplotlib
matplotlib.use("Agg")  # WSL 没有图形界面，直接把图存成文件
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer


def reservoir_sample(path, k, seed):
    """蓄水池抽样：只把文件从头到尾读一遍，就能均匀随机地抽出 k 行。
    比直接取前 k 行更有代表性（文件开头的数据可能来自同一个来源）。"""
    rng = random.Random(seed)
    sample = []
    total = 0
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            total += 1
            if i < k:
                sample.append(line)
            else:
                j = rng.randint(0, i)
                if j < k:
                    sample[j] = line
    return sample, total


def count_tokens(tok, record, kind):
    if kind == "pretrain":
        return len(tok.encode(record["text"]))
    # SFT：先用 chat template 拼成训练时的完整文本，再数 token
    msgs = record["conversations"]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False)
    except Exception:
        # 个别带工具调用的样本模板可能渲染失败，退化为直接拼接内容（会略微低估长度）
        text = "".join(str(m.get("content", "") or "") for m in msgs)
    return len(tok.encode(text))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="jsonl 数据文件路径")
    p.add_argument("--kind", choices=["pretrain", "sft"], required=True)
    p.add_argument("--tokenizer", default="model", help="tokenizer 所在目录")
    p.add_argument("--n", type=int, default=10000, help="抽样条数")
    p.add_argument("--max_len", type=int, default=768, help="训练用的 max_seq_len")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    lines, total = reservoir_sample(args.data, args.n, args.seed)
    lengths = np.array([count_tokens(tok, json.loads(l), args.kind) for l in lines])

    L = args.max_len
    truncated = lengths > L
    kept = np.minimum(lengths, L)
    stats = {
        "文件总条数": total,
        "抽样条数": len(lengths),
        "平均长度": lengths.mean(),
        "中位数 P50": np.percentile(lengths, 50),
        "P90": np.percentile(lengths, 90),
        "P95": np.percentile(lengths, 95),
        "P99": np.percentile(lengths, 99),
        "最长": lengths.max(),
        f"超过 {L} 被截断的样本比例": truncated.mean(),
        "被截掉的 token 占总 token 比例": (lengths - kept).sum() / lengths.sum(),
        f"补齐到 {L} 时 padding 占比（算力浪费）": 1 - kept.sum() / (len(lengths) * L),
        "估算全量 token 数（百万）": lengths.mean() * total / 1e6,
    }
    for k, v in stats.items():
        if isinstance(v, (float, np.floating)) and v < 1:
            print(f"{k:<32}{v:.2%}")
        elif isinstance(v, (float, np.floating)):
            print(f"{k:<32}{v:,.1f}")
        else:
            print(f"{k:<32}{v:,}")

    # 画直方图，横轴截到 P99.5，避免少数超长样本把图压扁
    xmax = max(np.percentile(lengths, 99.5), L * 1.2)
    plt.figure(figsize=(9, 4.5))
    plt.hist(np.clip(lengths, 0, xmax), bins=80, color="#4C78A8", alpha=0.85)
    plt.axvline(L, color="#E45756", linestyle="--", linewidth=2,
                label=f"max_seq_len = {L}  (truncated {truncated.mean():.1%})")
    plt.xlabel("tokens per sample")
    plt.ylabel("count")
    plt.title(f"{args.kind} token length distribution (n={len(lengths)})")
    plt.legend()
    plt.tight_layout()
    out = f"{args.kind}_token_lengths.png"
    plt.savefig(out, dpi=150)
    print(f"\n直方图已保存到 {out}")


if __name__ == "__main__":
    main()