# MiniMind 学习笔记：RoPE 与 YaRN

Nov 4, 2026 · @junG1ek

RoPE 把每个头的 96 维 Q、K 两两配成 48 对，位置 m 的第 i 对旋转 m·θᵢ，使注意力分数只依赖两个 token 的相对距离；不增加任何可学习参数。YaRN 在此基础上只压缩低频维度，让训练长度之外的位置也落回模型见过的角度范围。

## 核心思想

把向量维度两两配对，每对看成二维平面上的一个点；位置 m 的 token，第 i 对旋转 m·θᵢ。

```latex
\begin{pmatrix} x'_a \\ x'_b \end{pmatrix}
=
\begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix}
\begin{pmatrix} x_a \\ x_b \end{pmatrix}
```

**为什么旋转能编码位置**：位置 m 的 q 和位置 n 的 k 都旋转后再做点积，结果只取决于相对距离 n − m。注意力分数因此天然感知「两个 token 隔了多远」。

```latex
\langle R_m q,\ R_n k \rangle = \langle q,\ R_{n-m}\, k \rangle
```

**频率**：每对维度的旋转速度不同。

```latex
\theta_i = \text{base}^{-2i/d}, \qquad i = 0, 1, \dots, \tfrac{d}{2}-1
```

**波长与圈数**：转满一圈需要 λᵢ 个位置；在训练长度 L 内一共转 rᵢ 圈。

```latex
\lambda_i = \frac{2\pi}{\theta_i}, \qquad r_i = \frac{L}{\lambda_i} = \frac{L\,\theta_i}{2\pi}
```

| 维度对 | θᵢ（弧度/位置） | 波长 | L = 2048 内转了 | 负责 |
| --- | --- | --- | --- | --- |
| i = 0（最高频） | 1 | 约 6.3 个位置 | 约 326 圈 | 相邻 token 之间的精细距离 |
| i = 47（最低频） | 约 1.3 × 10⁻⁶ | 约 470 万个位置 | 约 0.0004 圈 | 很远距离的粗粒度信息 |

**MiniMind 的参数**：`head_dim` d = 96，所以有 48 对、48 个频率；`rope_base` = 1e6，比原论文的 1 万大得多，低频转得更慢，能区分更远的距离，对长上下文更友好（Qwen 也用这么大的值）。

**不是可学习参数**：cos、sin 表完全由公式算出，通常注册成 buffer，随模型移到 GPU，但不参与训练。RoPE 由苏剑林等人在 2021 年的 RoFormer 中提出。

## precompute\_freqs\_cis：预先算好 cos、sin 表

模型初始化时运行一次，产出两张 \[32768, 96\] 的表，之后每次 forward 只按当前位置切片。

| 步骤 | 代码 | 结果形状 | 含义 |
| --- | --- | --- | --- |
| 1 计算频率 | `1.0 / (rope_base ** (torch.arange(0, dim, 2)[:dim // 2].float() / dim))` | \[48\] | 48 个 θᵢ，一维张量 |
| 2 YaRN（可选） | `freqs * (1 - ramp + ramp / factor)` | \[48\] | `ramp` 也是 \[48\]，逐元素相乘 |
| 3 位置序列 | `t = torch.arange(end)` | \[32768\] | 位置 0 到 32767 |
| 4 外积 | `torch.outer(t, freqs).float()` | \[32768, 48\] | 第 m 行第 i 列 = m·θᵢ |
| 5 取 cos / sin 并拼两份 | `torch.cat([cos, cos], dim=-1) * attn_factor` | \[32768, 96\] | 第 i 列和第 i+48 列相同 |

**第 4 步外积**：每个位置乘以每个频率，得到所有位置、所有频率的旋转角度。每一行是一个位置（第 0 行全为 0，不旋转），每一列是一个频率（从上往下线性增长，第 0 列每行加 1 弧度，第 47 列几乎不动）。变量名 `freqs` 在这一步之后从「频率」变成了「角度」。

```latex
\text{freqs}[m, i] = m \cdot \theta_i
```

**`.float()` 很重要**：第 0 列最后一行的角度约 32767 弧度。fp32 能较精确地表示；bf16 只有两三位有效数字，在这么大的数值附近相邻可表示的数相差上百弧度，cos、sin 会完全算错。位置编码相关计算一律用 fp32。

**第 5 步为什么拼两份**：配合前后半配对，第 i 维和第 i+48 维必须用同一个角度 m·θᵢ。拼接后 96 列的表可以直接和 96 维的 q 逐元素相乘。

**`end` = 32 × 1024 = 32768**：对应 config 的 `max_position_embeddings`，是位置表的硬上限，超过就会越界报错。两张表共约 25MB（fp32），算一次即可。它不等于模型实际能处理好的长度，见下文 YaRN 一节。

## apply\_rotary\_pos\_emb：把旋转作用到 Q 和 K

在每层 attention 里，Q、K 投影完成后、计算注意力分数之前调用；8 个 Q 头和 4 个 KV 头用同一张表。

| 参数 | 形状 |
| --- | --- |
| `q` | \[B, T, 8, 96\] |
| `k` | \[B, T, 4, 96\] |
| `cos`、`sin` | \[T, 96\]，从大表中切出当前 T 个位置 |

**rotate\_half**：`torch.cat((-x[..., 48:], x[..., :48]), dim=-1)`，把 \[x₁, x₂\] 变成 \[−x₂, x₁\]（x₁ 为前 48 维，x₂ 为后 48 维）。`...` 表示保留前面所有维度。

**旋转公式**：

```latex
q' = q \odot \cos + \text{rotate\_half}(q) \odot \sin
```

对第 i 对展开，正好是二维旋转：

```latex
\begin{aligned}
q'_i &= q_i \cos m\theta_i - q_{i+48} \sin m\theta_i \\
q'_{i+48} &= q_{i+48} \cos m\theta_i + q_i \sin m\theta_i
\end{aligned}
```

### 两种配对方式

| 配对方式 | 第 0 对 | 第 1 对 | 第 47 对 |
| --- | --- | --- | --- |
| 相邻配对（RoPE 原论文） | 维度 0、1 | 维度 2、3 | 维度 94、95 |
| 前后半配对（MiniMind、Llama、Qwen 的实现） | 维度 0、48 | 维度 1、49 | 维度 47、95 |

两者效果等价：只是维度的排列顺序不同，`Wq`、`Wk` 训练时会自动适应。选前后半配对是因为切片加拼接更简单、更快。但权重和实现必须对应，跨框架转换模型时用错配对方式会得到错误结果。

**4 维小例子**：q = \[a, b, c, d\]，配对为 (a, c)、(b, d)。`rotate_half(q)` = \[−c, −d, a, b\]，cos 行 = \[cos θ₀, cos θ₁, cos θ₀, cos θ₁\]，展开得到 (a, c) 按 θ₀ 旋转、(b, d) 按 θ₁ 旋转。

### 其他细节

- **`unsqueeze(1)` 用于广播**：cos \[T, 96\] → \[T, 1, 96\]，与 q \[B, T, 8, 96\] 从右对齐，B 和头数维度自动复制，每个位置用自己那一行。这依赖 q 的维度顺序是 \[B, T, 头数, 96\]，若为 \[B, 头数, T, 96\] 则要换插入位置，所以做成了参数。
- **`.to(q.dtype)`**：fp32 的 cos、sin 与 bf16 的 q 相乘会提升为 fp32，最后转回 bf16。
- **只旋转 Q、K，不旋转 V**：位置只需影响注意力分数（Q、K 的点积），V 是被加权求和的内容。
- **K 先旋转再 `repeat_kv`**：广播对任意头数都适用。
- **推理时的 KV cache**：每步只输入一个新 token，cos、sin 要按它的实际位置切片（第 101 个 token 取第 100 行）；缓存里的 K 存入时已经旋转过，不用重算。

## YaRN 长度外推

YaRN 只压缩训练长度内转不满一圈的低频维度，高频维度保持不变；推理时用 `--inference_rope_scaling` 开启，不需要重新训练。

**触发条件**：传入了 `rope_scaling`，且目标长度 `end` 超过 `original_max_position_embeddings`（默认 2048）。核心一行：

```python
freqs = freqs * (1 - ramp + ramp / factor)   # factor 默认 16
```

### 为什么高频不动、低频压缩

- **高频维度**（`ramp` = 0）：训练时已转了几百圈，0 到 2π 的每个角度都见过很多遍。超出训练长度的位置只是又转了几圈，模型完全认识；它们负责相邻 token 的精细距离，任何位置都一样。
- **低频维度**（`ramp` = 1）：i = 47 在 2048 个位置里只转了约 0.0027 弧度，模型只见过 \[0, 0.0027\] 这一小段。直接推理到 32768，角度变成约 0.044，是从没见过的 16 倍。频率除以 16 后，第 32768 个位置的角度正好等于原来第 2048 个位置的角度，落回见过的范围。
- **代价**：压缩后相距 16 个位置的 token 在这一维上看起来只相距 1 个。低频维度本来负责粗粒度远距离信息，影响不大。

```latex
\theta_i' = \frac{\theta_i}{16}
\quad\Rightarrow\quad
32768 \cdot \theta_i' = 2048 \cdot \theta_i
```

**与早期位置插值（PI）的区别**：PI 把所有频率统一除以 factor，高频也被压缩，相邻 token 的距离变模糊，短距离理解变差。YaRN 按频率分段处理。

**钟表类比**：钟只观察了 1 小时，分针（高频）已转满一圈，所有角度都见过；时针（低频）只走了 30°。要表示 16 小时，分针照常转，时针会走到没见过的 480°，所以让时针慢 16 倍。

### 分界线：按训练长度内转了几圈

`beta_fast` = 32、`beta_slow` = 1 的单位就是圈数；`inv_dim(b)` 反求「转了 b 圈的是第几对」。按 MiniMind 参数（d = 96、base = 1e6、L = 2048）计算：

| 训练长度内转了 | 维度对 i | `ramp` | 处理 |
| --- | --- | --- | --- |
| 超过 32 圈 | 0 到约 8 | 0 | 不动 |
| 1 到 32 圈 | 约 8 到 21 | 0 到 1 线性过渡 | 部分压缩 |
| 不到 1 圈 | 约 21 到 47 | 1 | 频率除以 16 |

### attn\_factor：修正注意力温度

cos、sin 同时乘以系数 a，旋转后 q、k 长度各放大 a 倍，注意力分数放大 a² 倍，相当于调整 softmax 温度。序列变长后注意力容易过于平均，适当放大分数能保持集中。默认 1.0（不开 YaRN 时乘了等于没乘）。

```latex
\langle a\,R_m q,\ a\,R_n k \rangle = a^2 \langle R_m q,\ R_n k \rangle
```

### 三个长度的区别

| 长度 | 值 | 含义 |
| --- | --- | --- |
| `max_seq_len` | 768 | 训练时实际喂给模型的长度，模型只在这个范围内学过 |
| `orig_max` | 2048 | YaRN 计算时假定的原始训练长度（默认值） |
| `end` / `max_position_embeddings` | 32768 | 位置表上限，代码层面能处理的最大长度 |

2048 × 16 = 32768，YaRN 的默认值和位置表上限是配套设计的。不开 YaRN 时，超过 768 后的位置角度模型没见过，效果会明显变差。YaRN 由 Bowen Peng 等人在 2023 年提出。

## 验证实验与面试要点

亲手验证「点积只依赖相对位置」和 YaRN 分界线，比背结论记得牢。

**实验 1：相对位置性质**。前两个结果应相等（都相距 2），第三个不同。

```python
import torch
torch.manual_seed(0)
dim, base = 96, 1e6
theta = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

def rope(x, m):
    ang = m * theta
    cos = torch.cat([ang.cos(), ang.cos()])
    sin = torch.cat([ang.sin(), ang.sin()])
    rot = torch.cat([-x[dim // 2:], x[:dim // 2]])
    return x * cos + rot * sin

q, k = torch.randn(dim), torch.randn(dim)
print(rope(q, 5) @ rope(k, 3))        # 相距 2
print(rope(q, 105) @ rope(k, 103))    # 相距 2，应与上一行相等
print(rope(q, 5) @ rope(k, 1))        # 相距 4，结果不同
```

**实验 2：形状和极端角度**

```python
dim, base, end = 96, 1e6, 32 * 1024
theta = 1.0 / (base ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
angles = torch.outer(torch.arange(end), theta).float()
print(theta.shape, angles.shape)     # [48] 和 [32768, 48]
print(angles[-1, [0, -1]])           # 约 32767 和约 0.044
```

**实验 3：YaRN 分界线**

```python
import math
L = 2048
inv_dim = lambda b: dim * math.log(L / (b * 2 * math.pi)) / (2 * math.log(base))
print(inv_dim(32), inv_dim(1))       # 约 8.06 和 20.1
```

### 面试常问

- [ ] RoPE 为什么能表示相对位置：旋转后点积只取决于 n − m
- [ ] 为什么只作用于 Q、K，不作用于 V
- [ ] `rope_base` 调大有什么效果：低频转得更慢，长上下文更好
- [ ] 相邻配对和前后半配对的区别，为什么等价
- [ ] 位置计算为什么必须用 fp32
- [ ] 外推方法对比：直接外推、位置插值、NTK、YaRN（按频率分段）
- [ ] 不看代码手写 `rotate_half` 和 `apply_rotary_pos_emb`

参考：[minimind-3 README 镜像](https://app.alphaneural.io/models/ycoke/minimind-3) 中的 YaRN 配置（`factor` 16、`original_max_position_embeddings` 2048、`beta_fast` 32、`beta_slow` 1）。
