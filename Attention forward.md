# MiniMind 学习笔记：Attention 的 forward

2026-11-05 · junG1ek

Attention 的输入输出都是 [B, T, 768]；中间把向量拆成 8 个 Q 头和 4 个 KV 头，经 QK-Norm、RoPE、KV cache、掩码和 softmax，再合并回 768 维，加回残差主干。

## 流程总览

一次 forward 走 11 步；形状里 T 是本次输入的 token 数（`seq_len`），T_总 是加上缓存后的 key 总数，训练时两者相等。

| # | 步骤 | 代码 | 形状 |
| --- | --- | --- | --- |
| 1 | 输入 | `bsz, seq_len, _ = x.shape` | [B, T, 768]，`_` 是 `hidden_size` |
| 2 | 投影 | `q_proj(x)`、`k_proj(x)`、`v_proj(x)` | Q [B, T, 768]；K、V [B, T, 384] |
| 3 | 拆成多头 | `.view(bsz, seq_len, 头数, head_dim)` | Q [B, T, 8, 96]；K、V [B, T, 4, 96] |
| 4 | QK-Norm | `q_norm(xq)`、`k_norm(xk)` | 不变 |
| 5 | RoPE | `apply_rotary_pos_emb(xq, xk, cos, sin)` | 不变 |
| 6 | 拼接缓存 | `torch.cat([past_key_value[0], xk], dim=1)` | K、V [B, T_总, 4, 96] |
| 7 | 保存缓存 | `past_kv = (xk, xv) if use_cache else None` | 存 4 个头的版本 |
| 8 | 复制 KV 头并换维度 | `repeat_kv(...).transpose(1, 2)` | Q、K、V [B, 8, T 或 T_总, 96] |
| 9 | 算注意力 | flash 或手写路径 | 分数 [B, 8, T, T_总]；输出 [B, 8, T, 96] |
| 10 | 合并多头 | `.transpose(1, 2).reshape(bsz, seq_len, -1)` | [B, T, 768] |
| 11 | 输出投影 | `resid_dropout(o_proj(output))` | [B, T, 768]，与输入相同 |

输入的 x 是残差主干上的向量（第 1 层来自嵌入层，之后来自上一层），先经过 Block 开头的 RMSNorm。输出形状和输入完全一致，回到 Block 里直接加回：`x = x + output`。

## 投影、拆多头与 QK-Norm

K、V 比 Q 窄一半（GQA），拆多头用 `view` 不复制数据，Q 和 K 各有一个独立的 RMSNorm。

### GQA：几个 Q 头共用一组 K、V

| 方式 | KV 头数 | K/V 维度 | 特点 |
| --- | --- | --- | --- |
| MHA | 8 | 768 | 效果最好，KV cache 最大 |
| GQA（MiniMind） | 4 | 384 = 4 × 96 | 每 2 个 Q 头共用 1 个 KV 头，KV cache 和 K/V 参数都减半 |
| MQA | 1 | 96 | 最省显存，效果损失较明显 |

### view 只改形状，不复制数据

`view` 只改形状和步长等元信息，新旧张量共用同一块内存（改一个另一个也变）。它要求内存连续，所以经过 `transpose`、`permute`、`expand` 之后再用 `view` 会报错。

| 操作 | 是否复制数据 |
| --- | --- |
| `view` | 否，只能用于连续张量 |
| `transpose`、`permute`、`expand`、`unsqueeze`、切片 | 否，但前三个通常让张量不连续 |
| `reshape` | 连续时等同 `view`，不连续时自动复制 |
| `contiguous()` | 不连续时复制成连续 |

### QK-Norm：q、k 为什么分开归一化

Qwen3 引入的设计，在头的维度（96）上各做一次 RMSNorm。

- 分开的原因：Q 和 K 来自不同的投影矩阵、角色不同（q 是「我在找什么」，k 是「我能提供什么」），数值分布不同；分开后两个 γ 能独立学习缩放。
- 作用：q、k 长度可能在训练中越变越大，点积失控会让 softmax 压到单个 token 上、梯度消失、出现 loss 尖峰。归一化后长度稳定，分数的整体大小改由 γ 学习，相当于学一个温度。
- 细节：Q 的 8 个头共用 `q_norm` 的同一个 γ，K 的 4 个头共用 `k_norm` 的 γ，每个长 96；每层 192 个参数。位置在 RoPE 之前，RoPE 是旋转、不改变长度，所以长度依然受控。

$$
\text{score} = \frac{q \cdot k}{\sqrt{d}}
$$

## RoPE 与 KV cache

推理时传进来的 x 只有新 token，所以投影只算新 token；旧 token 的 K、V 从缓存里拼到前面，新的 q 才能和全部历史计算注意力。

**RoPE**：只旋转 Q、K，不旋转 V；cos、sin 表按当前 token 的实际位置切片。细节见《MiniMind 学习笔记：RoPE 与 YaRN》。

### KV cache 怎么工作

| 阶段 | 送进来的 x | 投影算了什么 | 拼接后的 K |
| --- | --- | --- | --- |
| 预填充（prompt） | [B, 100, 768] | 100 个 token | 无缓存，就是这 100 个 |
| 解码第 1 步 | [B, 1, 768]，只有新 token | 1 个 token | 100 + 1 = 101 个 |
| 解码第 2 步 | [B, 1, 768] | 1 个 token | 101 + 1 = 102 个 |

- `past_key_value[0]` 是 K，`[1]` 是 V；沿 `dim=1`（序列长度维）拼接。
- 没有缓存：每生成一个 token 都要重算所有历史 K、V，总计算量随 n² 增长。
- 生成循环每步只把最后一个 token 送进模型（`generate` 里对 `input_ids` 切片）；位置编码从「已缓存长度」开始切，在 `MiniMindModel.forward` 里计算起始位置。

### 缓存的位置

- 在 RoPE 之后：token 位置确定后不再变，旋转结果直接复用。
- 在 `repeat_kv` 之前：只存 4 个 KV 头，用时再复制成 8 个，GQA 省的就是这里的显存。

### 缓存有多大

$$
\text{每个 token 的缓存} = 2 \times n_\text{layers} \times n_\text{kv\_heads} \times d_\text{head} \times \text{字节数}
$$

$$
2 \times 8 \times 4 \times 96 \times 2 = 12{,}288 \ \text{字节} \approx 12\ \text{KB}
$$

开头的 2 是 K 和 V 各一份；bf16 每个数 2 字节。一条 32768 长的序列约 400MB；大模型的 KV cache 常常比权重还占显存。

### 训练时为什么不用缓存

训练时整段文本已知，一次前向并行算出所有位置的 q、k、v，靠因果掩码防止看到未来，效果等价于逐个生成 T 次。训练时 `past_key_value=None`、`use_cache=False`；反向传播本来就保存了 K、V，再存缓存是浪费。例外：GRPO、PPO 的 rollout 是在做生成，会用缓存；之后计算 loss 的前向又回到不用缓存。

## repeat_kv 与 transpose

K、V 各自被复制到 8 个头（K 只和 K 拼，V 只和 V 拼），再把头的维度换到前面，让矩阵乘法在每个头内部对 token 维度计算。

### repeat_kv 的三步

```python
x[:, :, :, None, :].expand(bs, slen, n_kv, n_rep, d).reshape(bs, slen, n_kv * n_rep, d)
```

| 步骤 | 形状 | 说明 |
| --- | --- | --- |
| 原始 | [bs, slen, 4, 96] | 4 个 KV 头 |
| `None` 插入维度 | [bs, slen, 4, 1, 96] | 等价于 `unsqueeze(3)` |
| `expand` | [bs, slen, 4, 2, 96] | 不复制数据，步长设为 0 |
| `reshape` | [bs, slen, 8, 96] | 内存不连续，这一步才真正复制 |

头的顺序是「每个重复相邻」：K 为 `[k0, k0, k1, k1, k2, k2, k3, k3]`，V 同理。第 j 个 Q 头用第 j // 2 个原始 KV 头。顺序若是 `[k0, k1, k2, k3, k0, ...]` 不会报错，但 Q、KV 配对就错了。等价写法：`torch.repeat_interleave(x, n_rep, dim=2)`；`n_rep == 1` 时直接返回。

### transpose(1, 2)：把头换到前面

```
之前：[B, T, 头数, 96]  →  之后：[B, 头数, T, 96]
```

`@` 只对最后两维做矩阵乘，前面的维度都当批次。换成 [B, 8, T, 96] 后，B 和 8 个头都是批次，每个头独立计算；最后两维是 [T, 96]。不换的话最后两维是 [8, 96]，乘出来变成「头与头之间」的分数，含义完全不对。

$$
\text{scores} = \frac{Q K^\top}{\sqrt{d}}
$$

## 两条计算路径

条件都满足时用 PyTorch 融合算子（更快、更省显存），否则走手写路径（能处理缓存和 padding 的掩码）；两者在数学上结果一致。

### scores 的形状

Q [B, 8, T, 96] 乘以 Kᵀ [B, 8, 96, T_总]，得到 [B, 8, T, T_总]：96 这一维被求和掉（m × n 乘 n × p 得 m × p）。每个元素是两个 token 的 96 维向量点积，表示 token 之间的关系，所以是 T × T_总，不是 96 × 96。

$$
\text{scores}[m, n] = \frac{1}{\sqrt{96}} \sum_{j=1}^{96} q_{m,j}\, k_{n,j}
$$

- 有缓存时 query 只有新 token（T 个），key 包括缓存（T_总 个），所以是长方形；无缓存（训练、预填充）时才是方阵。
- 训练时每条样本都被 padding 到 768，所以 T 永远是 768，每个头的 scores 是 768 × 768。注意力的显存和计算随序列长度平方增长，与 head_dim 无关。

### 融合路径的条件

`self.flash = hasattr(F, 'scaled_dot_product_attention') and config.flash_attn`：PyTorch 2.0 以上有这个函数，且配置开启。

| 条件 | 不满足时为什么退回手写路径 |
| --- | --- |
| `self.flash` | 没开开关 |
| `seq_len > 1` | 解码时只有 1 个 query |
| `not is_causal or past_key_value is None` | 因果掩码加缓存 |
| `attention_mask` 为 None 或全为 1 | 有 padding；`is_causal=True` 不能和自定义 `attn_mask` 同时用 |

第 2、3 条的根本原因：PyTorch 的 `is_causal=True` 生成的是左上角对齐的下三角（`torch.ones(L, S).tril()`），只在 query 和 key 一样长时正确；有缓存时新 token 会被错误地挡住大部分历史。

### FlashAttention 为什么快

`scaled_dot_product_attention` 会自动选 FlashAttention、Memory-Efficient 或 Math 后端。手写路径要把 [B, 8, T, T] 的分数矩阵完整写进显存：batch 32、长度 768 时，一层 fp32 就约 604MB，且反复读写。FlashAttention 把 Q、K、V 切块，在芯片高速缓存里一次完成「乘 K、softmax、乘 V」，完整矩阵从不写入显存；用在线 softmax 边算边更新最大值和总和。显存从随长度平方变成线性，速度通常快几倍，结果是精确的而非近似。

$$
32 \times 8 \times 768 \times 768 \times 4\ \text{字节} \approx 604\ \text{MB}
$$

训练时 MiniMind 不传 `attention_mask`，走的就是融合路径。检查 5070 能否用 FlashAttention 后端：在 `sdpa_kernel(SDPBackend.FLASH_ATTENTION)` 下调用一次，不报错即可用。

## 掩码

因果掩码挡住「未来」的 key，padding 掩码挡住「无效」的 key；两者直接相加，被任意一个挡住就看不到。

### 因果掩码：只加在最后 seq_len 列

```python
scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf")).triu(1)
```

- 切片 [B, 8, T, T]，掩码 [T, T]；广播时从右对齐，掩码视为 [1, 1, T, T]，复制到所有样本和头上。切片是视图，`+=` 原地写回 scores。
- key 的顺序是 `cat([缓存, 新 token])`：缓存在左，对每个新 query 都是过去，全部可见；新 token 在右，彼此有先后，套下三角。这就是右下角对齐。
- 缓存为 0（训练）：切片覆盖全部列，就是标准下三角。新输入为 1（解码）：1 × 1 的 `triu(1)` 是 [[0]]，什么都不掩。

例：缓存 3 个、新输入 2 个，scores 为 [2, 5]：

$$
\begin{array}{c|ccc|cc}
 & k_0 & k_1 & k_2 & k_3 & k_4 \\
\hline
q_3 & 0 & 0 & 0 & 0 & -\infty \\
q_4 & 0 & 0 & 0 & 0 & 0
\end{array}
$$

### padding 掩码：掩的是列（key）

```python
scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
```

`attention_mask` [B, T_总]，真实 token 为 1、padding 为 0；两次 `unsqueeze` 变成 [B, 1, 1, T_总]，广播到所有头和 query。每个 query 行的 padding 列都被加上 -1e9。它的长度必须覆盖缓存加新 token。

### 为什么因果用 -inf，padding 用 -1e9

两种被掩的位置 softmax 后权重都是 0，区别在于整行是否可能全被掩掉：

- 因果掩码不会掩掉整行（对角线永远可见），用 -inf 安全，exp(-inf) 精确为 0。
- 左边 padding 时，padding 自己作为 query 的行能看到的全是 padding；若用 -inf，整行全是 -inf，softmax 计算 -inf − (-inf) 得到 NaN，并会扩散。-1e9 是有限值，不会出 NaN。
- 正常行里 exp(-1e9 − 最大值) 下溢为 0，和 -inf 效果一样。
- 坑：fp16 最大值只有 65504，-1e9 会溢出成 -inf。MiniMind 先 `.float()` 转 fp32；Hugging Face 用 `torch.finfo(dtype).min`。

$$
\frac{\exp(-\infty - (-\infty))}{\sum \exp(-\infty - (-\infty))} = \text{NaN}
$$

**padding 行的权重怎么分**：softmax 对整行加同一常数不敏感，数学上 -1e9 被约掉，权重按原始分数分配。但 fp32 在 1e9 附近的间距是 64，原始分数（个位数）加上 -1e9 后被舍入成同一个值，所以真实 PyTorch 里结果是均匀分布。这几行的输出最后都会被丢掉，怎么分不影响结果。

$$
\text{softmax}(s_1 + c,\ s_2 + c) = \text{softmax}(s_1,\ s_2)
$$

### 什么时候 padding 在左、什么时候在右

| 场景 | padding 位置 | 原因 |
| --- | --- | --- |
| 训练（预训练、SFT、DPO 前向） | 右边 | 整条一次算完；因果掩码让真实 token 天然看不到右边 padding，MiniMind 训练时不传 `attention_mask`；padding 位置的 label 为 -100 |
| 批量生成（部署服务、RL rollout） | 左边 | 每步取最后一个位置预测并在右边接新 token，左补让所有 prompt 结尾对齐；需传 `attention_mask`（`tokenizer.padding_side = "left"`） |
| 单条推理 | 不需要 | 只有一条输入 |

左边 padding 让真实 token 整体后移，但 RoPE 只看相对距离，分数不受影响。训练时固定补到 768 浪费算力（200 个真实 token 只用到 768² 里约 7%），改进方法有动态 padding 和 packing。

## softmax、dropout 与输出

softmax 按行把分数变成每个 query 的注意力分布，对 V 加权求和，合并 8 个头后经 `o_proj` 混合，输出加回残差。

### softmax 按行归一化

`F.softmax(scores, dim=-1)` 沿 key 维度，每一行（每个 query）单独归一化，行与行、样本与样本、头与头之间互不影响。按列做就变成「每个 key 把自己分给所有 query」，含义不对。

$$
w_{mn} = \frac{\exp(s_{mn})}{\sum_{n'} \exp(s_{mn'})},\qquad \text{out}_m = \sum_{n} w_{mn}\, v_n
$$

### 为什么对角线不一定是每行最大

1. Q 和 K 是不同的投影，分数是双线性形式，中间矩阵既不对称也不正定，衡量的是「我要找的」和「你能提供的」的匹配，而不是相似度。例：「打开」的 q 在找动作的主语，和「小明」的 k 匹配度最高。
2. 即使 Wq = Wk，点积也未归一化：另一个向量更长、方向足够接近时可以超过 ‖a‖²；只有余弦相似度才保证自己最大。
3. token 自身信息已通过残差连接保留，注意力的价值在于从别的 token 收集信息。

$$
s_{mn} = \frac{x_m^\top W_q^\top W_k\, x_n}{\sqrt{d}}, \qquad a \cdot b \le \|a\|\,\|b\|
$$

真实模型里常见的头：前一个 token 头（看 m − 1）、归纳头（找上次出现相同内容后的 token）、注意力汇聚（大量注意力落在第一个 token 上）。待做实验：预训练后把某层某头的权重画成热力图。

### 两种 dropout

| | attention dropout | 残差 dropout |
| --- | --- | --- |
| 代码 | `attn_dropout(softmax(...))` | `resid_dropout(o_proj(output))` |
| 作用对象 | softmax 后的权重 [B, 8, T, T_总] | 子层输出 [B, T, 768] |
| 置 0 的含义 | 这次不看某个 token | 这次不加入某些特征维度 |

- 要作用在 softmax 之后：在分数上置 0，exp(0) = 1 反而可能让权重变大。
- inverted dropout：以概率 p 置 0，其余除以 (1 − p)，期望不变，推理时直接关闭（`model.eval()`）；dropout 后每行之和不再严格为 1。
- 残差 dropout 只丢支路、不丢主干，之前各层的信息总能沿残差完整传下去；来自原始 Transformer（p = 0.1）。
- MiniMind 里 `self.dropout = kwargs.get("dropout", 0.0)`，训练脚本不传，所以两种 dropout 都不起作用。大模型预训练数据多、只训一个 epoch，问题是学不够而非过拟合。小数据 SFT/LoRA 过拟合时可在 `MiniMindConfig(dropout=0.1)` 打开做对比。

$$
\tilde{w}_{mn} = \begin{cases} 0 & \text{概率 } p \\ \dfrac{w_{mn}}{1-p} & \text{概率 } 1-p \end{cases}
$$

### 合并多头、输出投影与返回

1. `weights @ xv`：[B, 8, T, T_总] × [B, 8, T_总, 96] → [B, 8, T, 96]。
2. `transpose(1, 2).reshape(bsz, seq_len, -1)`：→ [B, T, 768]，第 0–95 维是第 0 个头，以此类推；`transpose` 后不连续，必须 `reshape`。
3. `o_proj`（768 → 768，即 W_O）：拼接只是把各头并排放，`o_proj` 让每个输出维度都能用上所有头的信息。
4. `return output, past_kv`：output 与输入同形，Block 里 `x = x + output`；模型收集 8 层各自的 `past_kv` 供下一步生成使用。

$$
\text{MultiHead}(x) = \text{Concat}(\text{head}_1, \dots, \text{head}_8)\, W_O, \qquad h = x + \text{Dropout}\big(\text{MultiHead}(\text{RMSNorm}(x))\big)
$$

