import json, itertools

with open("dataset/dpo.jsonl") as f:
    for line in itertools.islice(f, 3):
        print(json.dumps(json.loads(line), ensure_ascii=False, indent=2))


# DATASETS SIZE
# pretrain: 1270238
# sft:      905718
# dpo:      17166
# rlaif:    19502