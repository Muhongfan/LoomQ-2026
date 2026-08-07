# LoomQ 人工评分证据

这份文件是人工评分材料的统一入口。请直接编辑它，只填写要申报的项目。截图、原始结果或图表统一放在 `starter_kit/evidence/files/`，也可以引用 `starter_kit/` 中已有的代码和文档。

证据包是可选的。没有申报某项人工分时，留空即可，不影响自动评分。

## 提交前填写

把要申报项目的方框改成 `[x]`，并填写对应内容：

- [ ] L1 真机
- [x] L2 交互体验
- [x] 工程与产品化
- [ ] 自定义量子 RISC-V Bonus
- [ ] 新手引导与视觉叙事 Bonus

## L1 真机

每个有效真机平台计 5 分，最多两个平台。模拟器不计真机分。每个平台复制并填写一次下面的信息：

```text
平台名称：[填写]
平台 job ID：[填写]
运行时间：[填写，带时区]
shots：[填写]
实际执行的 QASM：[填写仓库内路径]
平台返回的原始结果：[填写仓库内路径]
任务页截图：[选填，填写仓库内路径]
```

建议把文件放进 `evidence/files/`，比如：

```text
evidence/files/spinq-circuit.qasm
evidence/files/spinq-result.json
evidence/files/spinq-screenshot.png
```

工作人员会核对 job ID、运行时间、电路、shots 和原始结果。截图只能辅助说明，不能代替 job ID 和原始结果。

## L2 交互体验

请填写：

```text
启动界面或 CLI 的命令：
  1. 设置好 LOOMQ_LLM_BASE_URL / LOOMQ_LLM_API_KEY / LOOMQ_LLM_MODEL
     三个环境变量（可参考仓库根目录的 .env.example，运行
     `set -a && source .env && set +a` 加载）
  2. python3 starter_kit/chat_cli.py

测试入口或页面地址：无（纯 CLI，交互式对话界面，见上方启动命令）

补充说明：这是一个连续对话（可以直接追问、修改之前的请求，不用每次重新说一遍
完整描述）；每次生成/修复出电路后，助手会自动实际运行一遍并展示测量结果的
文本条形图，不只是甩一段代码。

适合现场体验的 3 个用户任务：
1. 用自己的话描述一个你想要的量子态（不用任何术语也行），比如"帮我做一个
   4 个粒子永远同增同减的电路"——看它能不能识别出这是 GHZ 态、生成正确代码
   并自动展示运行结果的条形图。然后直接追问"把粒子数改成 6 个"（不用重复
   "GHZ"或"纠缠"这些词）——看它能不能理解这是在延续上一轮的请求。
2. 故意给它一段有点小问题的代码，让它帮你诊断修复，比如："我想制备一个贝
   尔态，但这段代码报错了，帮我修好：H q[0]; CX q[0] q[1]"（未定义寄存器、
   门名大小写错误）——看它能不能在保持你原始目标的前提下修好。
3. 描述你的电路需求（比特数、要不要排队、要不要真机、在不在乎花钱），比如
   "我要跑一个 15 比特电路，不想排队，该用哪个平台？"——看它能不能给出正
   确的后端推荐和理由。

截图或演示视频：[选填，填写仓库内路径或稳定只读链接]
```

工作人员会在组委会统一模型环境中运行最终代码，测试新手是否看得懂、出错后能否得到有效帮助、结果是否清楚，以及多轮回答是否一致。选手自己的对话截图只用于说明产品流程，不直接证明得分。

## 工程与产品化

已有内容可以直接引用主 README 或其他项目文档，不必复制到本目录。

```text
干净环境中的构建和启动命令：
  方式一（Docker，推荐用于评委复现）：
    cd starter_kit
    docker build -t loomq-submission .
    docker run --rm loomq-submission
    # 默认 CMD 跑 python evaluator.py --json-out /tmp/loomq-public-report.json，
    # 即 submission.yaml 中 levels 为 true 的项（当前 l1+l2），l1 默认目标是
    # evaluator.py 的 --target 默认值 spinq,originq；只用 requirements.txt
    # 锁定的版本，容器内零手动干预
    # （见 ../ROADMAP.md Phase 6：docker build --platform linux/amd64 已在
    # Linux 容器内验证过 L1 公开电路通过，且 spinqit 的 macOS-only rpath
    # 问题确认不影响 Linux 容器；必须用 --platform linux/amd64，spinqit
    # 没有 arm64 wheel）

  方式二（本地 venv，用于开发/调试）：
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r starter_kit/requirements.txt
    set -a && source .env && set +a   # 加载 LOOMQ_LLM_* 三个环境变量（L2 用）
    cd starter_kit
    python3 evaluator.py --level l1 --target spinq,braket,originq   # L1 公开自测
    python3 evaluator.py --level l2                                  # L2 公开自测
    python3 chat_cli.py                                               # L2 零基础用户交互入口

架构说明：单向流水线，`starter_kit/adapter.py` 是评测唯一入口，内部委托给以下模块
（详见仓库根目录 README.md 的系统架构图，以及 ROADMAP.md 逐阶段记录）：
  circuit_ir.py（QASM2 解析为内部 Circuit/GateOp 表示）
    -> validator.py（12 门白名单 + arity + 越界/重复比特校验）
    -> lowering.py（按 gate_identities.py 把目标后端不支持的门分解为等价序列；
       当前 3 个后端原生支持全部 12 门，这一步是有文档说明的 no-op，作为兜底路径保留）
    -> emitters.py（emit_spinq / emit_braket / emit_originq，按 target_ir_contract.md
       生成三种目标格式的原生指令文本）
    -> runner.py（run()：调用对应 SDK 真正执行，并把三个后端不一致的比特序约定
       统一归一化为契约要求的 c[n-1]...c[0] 小端表示）
  L2 是独立的一层，不改动 L1 的确定性流水线：l2_agent.py 用一个 system prompt
    驱动 LLM 完成"生成 / 纠错 / 选后端"三类任务，生成结果会调用 L1 自己的执行
    路径自验证（保真度不达标则带着具体分布差异重试），adapter.py::agent_chat
    只是薄委托；chat_cli.py 在此之上加了多轮会话记忆和文本条形图可视化，
    是唯一面向终端用户的可运行入口。
  L3（Hybrid-QASM/RISC-V 混合编译）尚未实现，adapter.py::compile_hybrid 保持
    NotImplementedError，submission.yaml 中 levels.l3: false。

目标用户和使用场景：面向没有量子力学或 QASM 背景、但有具体计算意图的"跨界"用户
  （产品/设计/内容从业者、量子计算爱好者、教学场景的学生）——本题面向的"原本进
  不来的人"。典型场景：用户不知道"Bell 态""GHZ 态"这些术语，只会说"我想要 3
  个粒子永远同时变化的效果"或"这段代码报错了帮我修好"，通过 chat_cli.py 的自然
  语言对话就能拿到可在三家真实量子云平台（量旋 / 本源 / AWS Braket）上运行的
  电路和结果，不需要先学会任何一家的专属 SDK 或指令集。

完整使用流程：
  1. 配置好 LOOMQ_LLM_* 三个环境变量（见 .env.example）
  2. 运行 python3 starter_kit/chat_cli.py，用大白话描述想要的量子效果
     （例："帮我做一个 4 个粒子永远同增同减的电路"）
  3. Agent 识别意图、生成 OpenQASM 2.0 电路，自动用 L1 的执行路径实际跑一遍
     自验证，展示测量结果的文本条形图；用户可直接追问修改（如"把粒子数改成
     6 个"）而不用重复完整描述
  4. 满意后同一份 QASM 可通过 adapter.transpile(qasm, target) /
     adapter.run(qasm, target, shots) 分别转译并运行到 spinq / originq / braket
     三个后端，得到统一 JSON Schema 的结果（backend_capabilities.md 帮助选择
     该用哪个后端）
  详细分阶段实现记录见仓库根目录 ROADMAP.md（非评分材料，但记录了每一步的
  验证方法和踩过的坑）。
```

工作人员会按最终 commit 实际构建和启动，并检查文档与代码是否一致、产品是否真的降低了量子计算的使用门槛。

## 自定义量子 RISC-V Bonus

以下三项必须齐全且测试通过，才获得 8 分：

```text
指令编码规格：[填写文档路径]
模拟器扩展实现：[填写代码路径]
端到端测试命令：[填写命令或文档路径]
```

## 新手引导与视觉叙事 Bonus

请填写已有材料的路径，不要求为评分另写一套文档：

```text
零基础首次运行指南：[填写]
量子概念解释：[填写]
结果可视化：[填写]
错误恢复或无障碍引导：[填写]
```

以上四项各 1 分。普通项目 README 完整不代表自动获得 Bonus。

## 提交规则

- 所有材料都要在截止前进入最终提交的 commit，工作人员不接受截止后补交。
- 外部视频可以用稳定只读链接，源码、原始结果和复现命令应保存在仓库中。
- 整个 fork commit 的归档包不得超过 100 MiB。
- 不要提交 API Key、Token、Cookie、个人身份信息或平台账户隐私。
- 如申报 L1 真机分，在最终提交 Issue 的 `Hardware evidence` 中填写 `starter_kit/evidence/README.md`。
