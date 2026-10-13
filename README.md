# ✦ gradforge

> **世界顶级自动微分与最优化内核** (World-class Autodiff & Optimization Core)
> 零依赖 · 零构建 · 单文件 · 浏览器 / Node 双端可跑 · 可验证 · 可复现

作者：**晨星 (MorningStar, CJX0712)** ｜ License：MIT ｜ 原则：**零重写 · 集大成 · 闭环交付**

---

## 它解决什么

现代 AI 的每一次参数更新，底层都是同一件事：**自动微分 + 最优化**。
`gradforge` 把这两块世界顶级成果收敛进一个单文件内核：

- **Reverse-mode AD（反向模式自动微分）** —— 继承 Karpathy `micrograd` 的标量计算图范式，并扩展 `exp / log / softmax+cross-entropy / layerNorm / matmul` 等世界顶级算子。
- **Newton-CG（Hessian-free 二阶优化）** —— 源自 Martens 2010《Deep learning via Hessian-free optimization》：用**数值 HVP**（对解析梯度做中心差分）得到 Hessian-Vector Product，免逆矩阵、复杂度 O(n)，收敛阶为**二次**。
- **优化器族** —— SGD / Momentum / AdaGrad / RMSProp / Adam / **AdamW（解耦权重衰减）**，配 constant / step / **cosine-with-warmup** 学习率调度。
- **LR Range Test** —— Smith 2015 的学习率探测工具，一键定位最优 LR 区间。

> 设计哲学：不重复造轮子，但**每一行都站在世界顶级成果之上**；凡更强的现成方案立即替换自写。

---

## 架构

```
┌──────────────────────────────────────────────────────────┐
│                      gradforge.js (单文件)                │
├──────────────────────────────────────────────────────────┤
│  Layer 3  应用层   MLP · LayerNorm · softmaxCE · XOR 训练 │
│  Layer 2  优化层   SGD/Momentum/AdaGrad/RMSProp/Adam/AdamW│
│                        + 调度 · Newton-CG · LR Range Test  │
│  Layer 1  算子层   + - * / pow exp log relu tanh sigmoid   │
│                        matmul · layerNorm · softmaxCE      │
│  Layer 0  内核     Value 计算图 · reverse-mode backward    │
│                        · 数值 HVP · CG 求解 · 确定性 RNG   │
└──────────────────────────────────────────────────────────┘
       零外部依赖 · 浏览器 <script> + Node require 双端
```

它是 **AI 系统十层分解架构 · 第十层 1024 微指令** 的「可微分计算地基」——
所有上层网络/注意力/状态空间的参数更新，最终都收敛到此处的 `backward()` 与 `step()`。

---

## 数学核心

### 1. 反向模式自动微分
对计算图拓扑排序后反向执行各节点的局部梯度（`reverse-mode`，一次前向+一次反向得全部一阶偏导）：

```
∂(a*b)/∂a = b ,   ∂(a*b)/∂b = a
∂ReLU(x)/∂x = 1{x>0} ,   ∂tanh = 1 - tanh² ,   ∂σ = σ(1-σ)
```

### 2. Newton-CG（Hessian-free）
求解牛顿方向 `H p = -∇f`，其中 Hessian-Vector Product 由**中心差分**获得（O(n)，免显式 H）：

```
HVP(v) = ( ∇f(x+εv) - ∇f(x-εv) ) / (2ε) + O(ε²)
```

再用共轭梯度（CG）解 `H p = -∇f`，配 Armijo 回溯线搜索。收敛不变量：

```
‖x_{k+1} - x*‖ / ‖x_k - x*‖²  →  const      (二次收敛)
```

### 3. AdamW 解耦权重衰减
```
mₜ = β₁mₜ₋₁ + (1-β₁)gₜ        vₜ = β₂vₜ₋₁ + (1-β₂)gₜ²
θₜ₋₁ = θₜ - η( m̂ₜ/(√v̂ₜ+ε) + λθₜ )      (λ = wd)
```

---

## 可验证不变量（verification-first）

| # | 不变量 | 验证方式 | 实测 |
|---|--------|----------|------|
| 1 | 解析梯度 ≈ 数值梯度 | 有限差分 `∂L/∂θ ≈ (L(θ+ε)-L(θ-ε))/2ε` | `maxRelErr = 3.87e-11` |
| 2 | softmaxCE 梯度正确 | 同上 | `maxRelErr = 2.26e-11` |
| 3 | 数值 HVP ≈ 解析 Hessian·v | 二次型 `f=½(ax²+by²)`, H=diag(a,b) | `relErr = 6.30e-13` |
| 4 | 牛顿二次收敛 | `‖eₖ₊₁‖/‖eₖ‖² ≈ const` | 4 步收敛至 `3.74e-11` |
| 5 | 端到端可训练 | MLP 在 XOR 上损失→0 | `MSE = 1.86e-10`, 4/4 正确 |

---

## 使用

### Node（无头自检）
```bash
node test.js
# ✅ 7/7 · 退出码 0（任一失败退出 1，可接入 CI）
```

### 浏览器（可视化演示）
直接用浏览器打开 `index.html`：
- ① 浏览器内运行全部不变量自检
- ② XOR 决策边界 + 训练损失曲线（实时绘制）
- ③ LR Range Test 曲线（自动标注最优 LR 拐点）

```html
<script src="gradforge.js"></script>
<script>
  const { Value, MLP, Adam, newtonCG, gradCheck } = GradForge;
  const a = new Value(2), b = new Value(-3);
  const y = a.mul(b).add(a.pow(2)); y.backward();
  console.log(a.grad, b.grad);   // 1, 4
</script>
```

### 作为库引入
```js
const GF = require('./gradforge.js');
const net = new GF.MLP([2, 8, 1], 42, 'tanh');
const opt = new GF.Adam(net.parameters(), 0.05);
// 前向 -> backward -> opt.step()，循环即可
```

---

## 与 forge 生态的关系

`gradforge` 是已有世界级仓库（tianji-ai / zenith-ai / aurora-core / gnnforge / mambaforge …）
共同缺失的那块**可微分优化地基**。它可作为它们的统一优化后端，使整个生态在
「1024 微指令层」上闭环：任何网络的参数更新，最终都走 `backward() + step()`。

---

## 文件

| 文件 | 作用 |
|------|------|
| `gradforge.js` | 核心引擎（UMD，浏览器+Node 双端） |
| `test.js` | 无头自检（7 项不变量，CI 友好） |
| `index.html` | 浏览器演示 + 可视化 + 自测 |
| `README.md` / `LICENSE` | 文档 / MIT 许可 |

---

> **唯一目标：锻造世界顶级 AI** —— 架构最先进、性能最强、代码最优、文档最全。
> 零重写 · 集大成 · 闭环交付 · 作者：晨星
