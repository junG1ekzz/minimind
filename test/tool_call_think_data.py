import json

found = {}
with open("dataset/sft_t2t_mini.jsonl") as f:
    for line in f:
        conv = json.loads(line)["conversations"]
        if "tool" not in found and any(m.get("tools") or m.get("tool_calls") for m in conv):
            found["tool"] = conv
        if "think" not in found and any(m.get("reasoning_content") for m in conv):
            found["think"] = conv
        if len(found) == 2:
            break

for k, v in found.items():
    print(f"===== {k} 样本 =====")
    print(json.dumps(v, ensure_ascii=False, indent=2))