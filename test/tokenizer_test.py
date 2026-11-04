from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("model")

print(type(tok).__name__)                # tokenizer 的类名
print(len(tok))                          # 词表大小，应该在 6400 左右
print(tok.bos_token, tok.eos_token, tok.pad_token)


ids = tok.encode("What is the weather today?")
print(ids)                               # token id 列表
print([tok.decode([i]) for i in ids])    # 每个 id 对应的文字
print(tok.decode(ids))                   # 整体还原，应该和原文一样