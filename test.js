'use strict';
/*
 * gradforge 无头自检 (headless self-test)
 * 运行: node test.js  → 全部 ✅ 退出 0, 任一 ❌ 退出 1
 * 覆盖三重可验证不变量:
 *   (1) 一阶: 解析梯度 ≈ 有限差分数值梯度 (relErr < 1e-4)
 *   (2) 二阶: 数值 HVP ≈ 解析 Hessian·v
 *   (3) 牛顿二次收敛率 → 2 (near-optimum ‖e_{k+1}‖/‖e_k‖² ≈ const)
 *   (4) 端到端: MLP 在 XOR 上 Adam 训练 loss → ~0
 */
const GF = require('./gradforge.js');
const { Value, V, MLP, softmaxCrossEntropy, numericalHVP, newtonCG, gradCheck, Adam } = GF;

let pass = 0, fail = 0;
const ok = (name, cond, extra = '') => {
  if (cond) { pass++; console.log(`  ✅ ${name} ${extra}`); }
  else { fail++; console.log(`  ❌ ${name} ${extra}`); }
};

// ---- (1) 一阶梯度校验: MLP + MSE ----
console.log('\n[1] 一阶梯度校验 — MLP(MSE)');
{
  const mlp = new MLP([2, 6, 1], 7, 'tanh');
  const P = mlp.parameters();
  const data = [
    { x: [0, 0], y: 0 }, { x: [0, 1], y: 1 }, { x: [1, 0], y: 1 }, { x: [1, 1], y: 0 },
  ];
  const lossFn = () => {
    let s = V(0);
    for (const d of data) {
      const o = mlp.forward(d.x)[0];
      const e = o.sub(V(d.y));
      s = s.add(e.mul(e));
    }
    return s;
  };
  const r = gradCheck(lossFn, P, 1e-5, 1e-4);
  ok('MLP 解析梯度≈数值梯度', r.pass, `maxRelErr=${r.maxRel.toExponential(2)}`);
}

// ---- (1b) 一阶梯度校验: softmax 交叉熵 ----
console.log('\n[1b] 一阶梯度校验 — softmaxCrossEntropy');
{
  const logits = [V(0.3), V(-0.2), V(1.1)];
  const lossFn = () => softmaxCrossEntropy(logits, 2);
  const r = gradCheck(lossFn, logits, 1e-5, 1e-4);
  ok('softmaxCE 解析梯度≈数值梯度', r.pass, `maxRelErr=${r.maxRel.toExponential(2)}`);
}

// ---- (2) 二阶 HVP 校验: 二次型 f=0.5*(a x²+b y²), H=diag(a,b) ----
console.log('\n[2] 二阶 HVP 校验 — 二次型解析 Hessian');
{
  const x = V(1.5), y = V(-0.8);
  const a = 3, b = 2;
  const f = () => x.mul(x).mul(V(a * 0.5)).add(y.mul(y).mul(V(b * 0.5)));
  const v = [1, -1];
  const hvp = numericalHVP(f, [x, y], v, 1e-4);
  const expect = [a * v[0], b * v[1]];
  const rel = Math.max(
    Math.abs(hvp[0] - expect[0]) / (Math.abs(expect[0]) + 1e-12),
    Math.abs(hvp[1] - expect[1]) / (Math.abs(expect[1]) + 1e-12)
  );
  ok('数值HVP≈解析Hessian·v', rel < 1e-4, `relErr=${rel.toExponential(2)} (expect [${expect}])`);
}

// ---- (3) 牛顿二次收敛: f = 0.5(x²+y²) + 0.1(x⁴+y⁴), 极小点 0 ----
console.log('\n[3] 牛顿二次收敛率 (Newton-CG)');
{
  const p = [V(0.5), V(0.4)];
  const f = () =>
    p[0].mul(p[0]).mul(V(0.5)).add(p[0].pow(4).mul(V(0.1)))
      .add(p[1].mul(p[1]).mul(V(0.5))).add(p[1].pow(4).mul(V(0.1)));
  const hist = newtonCG(f, p, { maxIter: 12, cgIters: 20 });
  // 近极小点 ∇f ≈ x, 故以 gradNorm 近似误差
  const errApprox = hist.map((h) => h.gradNorm);
  let ratio = null, ratioAt = -1;
  for (let k = hist.length - 1; k > 0; k--) {
    if (errApprox[k - 1] > 1e-2 && errApprox[k - 1] < 1) {
      ratio = errApprox[k] / (errApprox[k - 1] * errApprox[k - 1]);
      ratioAt = k; break;
    }
  }
  const last = errApprox[errApprox.length - 1];
  console.log(`      loss=${hist[hist.length - 1].loss.toExponential(2)} err≈${last.toExponential(2)} iters=${hist.length} 收敛率(ratio@${ratioAt})≈${ratio ? ratio.toExponential(2) : 'n/a'}`);
  ok('Newton-CG 收敛至极小点', last < 1e-6, `(err≈${last.toExponential(2)})`);
  ok('二次收敛率存在 (中等误差区 ratio 有界)', ratio !== null && ratio > 1e-4 && ratio < 1e4, `ratio=${ratio ? ratio.toExponential(2) : 'n/a'}`);
}

// ---- (4) 端到端: XOR 上 Adam 训练 ----
console.log('\n[4] 端到端 — MLP 在 XOR 上 Adam 训练');
{
  const mlp = new MLP([2, 8, 1], 42, 'tanh');
  const P = mlp.parameters();
  const opt = new Adam(P, 0.05);
  const X = [[0, 0], [0, 1], [1, 0], [1, 1]];
  const Y = [0, 1, 1, 0];
  let loss = 0;
  for (let it = 0; it < 3000; it++) {
    GF.zeroGrads(P);
    let s = V(0);
    for (let i = 0; i < 4; i++) {
      const o = mlp.forward(X[i])[0];
      const e = o.sub(V(Y[i]));
      s = s.add(e.mul(e));
    }
    loss = s.data / 4;
    s.backward();
    opt.step();
  }
  // 预测
  let correct = 0;
  X.forEach((x, i) => {
    const p = mlp.forward(x)[0].data;
    if (Math.round(p) === Y[i]) correct++;
  });
  console.log(`      final MSE=${loss.toExponential(2)} 准确率=${correct}/4`);
  ok('XOR loss 收敛 < 2e-2', loss < 2e-2, `(MSE=${loss.toExponential(2)})`);
  ok('XOR 全分类正确', correct === 4, `(${correct}/4)`);
}

console.log(`\n========== 结果: ${pass} ✅ / ${fail} ❌ ==========`);
process.exit(fail === 0 ? 0 : 1);
