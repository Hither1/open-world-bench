# 调研：开放世界 / 长程任务 benchmark 如何构造任务、说明什么问题

调研日期 2026-11-06，覆盖 12 个 benchmark 家族、约 45 篇论文。除标注 **[未核实]** 的条目外，
每条事实都对照了论文摘要 / 项目页 / 仓库。本文服务于 [重构方案](architecture-proposal.md) 第 5 节。

## 1. 总表

| Benchmark | 世界 | 任务怎么来 | 长程如何编码 | 成功如何判定 | 观测 / 动作 | 声称回答的问题 | 年份 / 会议 | 仓库 |
|---|---|---|---|---|---|---|---|---|
| MineDojo | Minecraft | 3,142 个任务：约 1.58k 程序化（wiki/模板）+ 1.56k 创造性（LLM/wiki 挖掘） | 扁平任务表，难度隐含在物品层级 | 程序化状态判定；创造性任务用 MineCLIP 视频-文本分数阈值 | 像素(+可选 inventory/voxel)；低层键鼠 | 用互联网知识驱动的开放式 agent | NeurIPS 2022 | github.com/MineDojo/MineDojo |
| MineRL BASALT | Minecraft | 4 个手写模糊任务（找洞、造瀑布、围牲口、盖房） | 无（单一模糊目标） | 人类成对视频比较 → TrueSkill | 像素；低层 | 无奖励、从人类反馈学 | NeurIPS 2021/22 竞赛 | arXiv 2107.01969 |
| Voyager | Minecraft (Mineflayer) | LLM 自动课程：根据当前状态提出下一个任务 | 涌现的科技树里程碑；独特物品数 | 程序化（库存/解锁）；学习中用 LLM 自验证 | 结构化文本状态；**代码即动作**（JS API）+ 技能库 | 终身学习、探索、技能复用 | TMLR 2024 | github.com/MineDojo/Voyager |
| DEPS / GITM / JARVIS-1 / STEVE-1 | Minecraft | 手选物品获取任务（DEPS 70+，JARVIS-1 200+，GITM 262 件） | **科技树依赖图**；ObtainDiamond = 13+ 个有序子目标 | 程序化库存检查；多 seed 成功率 | 像素或结构化文本；规划器 → 目标条件控制器 | 规划 + 记忆 + 分层控制 | 2023–24 | github.com/CraftJarvis |
| MCU | Minecraft | 3,452 个原子任务（wiki、游戏数据、旧 benchmark、LLM 头脑风暴）；LLM 生成任务配置并自验证 | 组合式多任务；simple/hard 标签 | **GPT-4o 看视频打分**，6 个维度，声称与人类一致 91.5% | 像素；低层 | 开放式通才评测、错误识别、创造性 | ICML 2025 | github.com/CraftJarvis/MCU |
| Optimus-1/2/3 | Minecraft | 67 个长程任务，7 个材料组 | 材料层级 = 难度；步数预算 3.6k–36k，来自人类完成时间 | 程序化 SR，≥30 seed；人类基线 | 640×360 RGB；STEVE-1 低层控制 | 多模态混合记忆对长程的作用 | 2024–25 **[会议未核实]** | cybertronagent.github.io |
| MineExplorer | Minecraft | 多 agent LLM 流程把原子任务组合成**隐含多跳任务图**；人工核验 | 轨迹中的隐藏前置条件 | 每实例基于规则的里程碑评测器 | ReAct 式文本/图像 | MLLM agent 的开放世界探索 | EMNLP 2026 | github.com/meituan-longcat/MineExplorer |
| Crafter | 2D 程序化生存 | 固定 22 个成就；地图按 seed 生成 | 成就间依赖构成科技树 | 成就解锁 → **各成就成功率的几何平均** | 64×64 像素；17 动作 | 探索、泛化、长程推理的能力谱 | ICLR 2022 | github.com/danijar/crafter |
| Craftax | 2D 程序化 + 地牢 | 65 个成就、4 个难度档 | 更深科技树 + 多层地牢 | 成就（难的权重更高） | 符号 8268 维或像素；43 动作 | 大规模开放式 RL（JAX） | ICML 2024 | github.com/MichaelTMatthews/Craftax |
| NLE / BALROG | Roguelike + 5 个现有环境 | 复用现有环境，统一 prompt | NetHack 用**人类数据校准的进度**：地牢层/经验等级 → 最终通关概率 | 二元 / 成就百分比 / 进度曲线 | 文本(+可选单图)；文本动作，16 步历史 | agent 推理、"知行差距"、视觉是否有用 | NeurIPS 2020 / ICLR 2025 | github.com/facebookresearch/nle；balrogai.com |
| ALFRED / ALFWorld / TEACh | AI2-THOR 家居 | PDDL 定义 7 类任务，25k 指令；ALFWorld 文本孪生；TEACh 加对话 | 多子目标序列 | 二元 SR + **目标条件完成比例** + 路径长度加权 | 像素 / 文本；离散导航 + 交互 | 指令接地、文本→具身迁移 | CVPR 2020 / ICLR 2021 / AAAI 2022 | askforalfred.com |
| EmbodiedBench | AI2-THOR + Habitat | 1,128 任务；人工 + GPT-4o 扩充 + 程序生成；6 个能力子集 | "Long Horizon" 子集 = >15 步 | 状态满足指令；导航用距离阈值 | 图像；高层技能或低层动作 | MLLM 具身能力画像 | ICML 2025 | embodiedbench.github.io |
| Embodied Agent Interface | BEHAVIOR + VirtualHome | 现有任务改写为 LTL 目标 | 四个模块：目标解释、子目标分解、动作排序、转移建模 | 模块级程序化检查 + 细粒度错误分类 | 符号状态（文本） | 定位 LLM 缺哪种能力 | NeurIPS 2024 D&B | pypi eai-eval |
| BEHAVIOR-1K | OmniGibson 家居 | 1,000 个活动，用 **BDDL** 写 init + goal 谓词 | 目标 = 谓词合取 | 每个仿真步跑谓词检查器；**满足的目标谓词比例**（部分得分） | 像素/本体感觉；连续机器人控制 | 人类中心的长程操作 | CoRL 2022 **[未核实]** | github.com/StanfordVL/BEHAVIOR-1K |
| BabyAI / MiniGrid / Procgen | 2D 网格 / 街机 | BabyAI 19 个难度等级；Procgen 16 个程序化游戏 | 课程等级；难度模式 | 二元到达目标 | 7×7 局部视野 / 64×64 像素 | 样本效率；**程序化泛化（seed 划分）** | ICLR 2019 / ICML 2020 | Farama Minigrid；openai/procgen |
| PokéAgent（RPG 赛道） | 宝可梦绿宝石 | 固定剧情 | **15 个标准化里程碑**，先比完成率，再比时间、动作数 | 里程碑完成百分比 | 画面 + 少量 RAM 状态；按键 | 长程规划；harness × 模型 | NeurIPS 2025 竞赛 | pokeagentchallenge.com |
| VideoGameBench | 10 个 90 年代游戏 + 3 个保密 | 现成游戏；dev/test 划分 | 游戏内检查点 | 检查点进度百分比 | 仅原始画面；键鼠；实时或暂停 "Lite" | VLM 能否通关 | arXiv 2505.18134 | github.com/alexzhang13/videogamebench |
| lmgame-Bench | 6 个游戏 | 现成游戏；做了污染检查 | 游戏原生进度 | 分数 / 进度 | 像素；**感知 / 记忆 / 推理模块可开关的 harness** | 解耦感知、记忆、推理 | ICLR 2026 | github.com/lmgame-org/GamingAgent |
| Cradle | RDR2、城市天际线、星露谷等 | 手选任务 | 40 分钟剧情任务 | 任务特定检查 + 人工核验 **[细节未核实]** | 仅截图；代码 → 键鼠 | 通用计算机控制 | ICML 2025 | baai-agents.github.io/Cradle |
| Lumine | 原神（+鸣潮、崩铁零样本） | 141 个语言任务（收集/战斗/NPC/解谜），蒙德域内、璃月/稻妻域外；5 小时主线 | 剧情章节；simple/hard | **人工评测**，每任务 5 次，5 分制 | 720p @5Hz；键鼠 30Hz 作为文本 token | 3D 开放世界通才 agent 配方 | arXiv 2511.08892 | lumine-ai.org |
| WebArena / OSWorld / TheAgentCompany / τ-bench / GAIA | 网页 / OS / 办公 / 文本 | 模板实例化 / 初始配置 + 评测脚本 / Docker 任务镜像 / 数据库目标状态 / 问答 | 多步意图；子检查点 | 程序化定位器、执行脚本、**子检查点部分得分**、DB 状态比对 + pass^k | DOM / 截图；文本 / API | 功能正确性、可靠性 | 2023–24 | 见 §2.8 |

## 2. 分家族要点

### 2.1 Minecraft 系
- **构造**：MineDojo 从 wiki/Reddit 挖任务 ([arXiv 2206.08853](https://arxiv.org/pdf/2206.08853))；规划派（DEPS、JARVIS-1、Optimus）手选物品目标，用**合成科技树**编码长程，ObtainDiamond "至少 13 个有序子目标" ([DEPS](https://arxiv.org/html/2302.01560v3))；Optimus-1 按材料分层，步数预算来自人类完成时间，并把钻石生成率调到 20% 保证可行 ([arXiv 2408.03615](https://arxiv.org/html/2408.03615v2))；Voyager 用 LLM 自动课程替代固定任务表 ([arXiv 2305.16291](https://arxiv.org/abs/2305.16291))；MCU 用 LLM 生成任务配置并自反思验证 ([MCU](https://craftjarvis.github.io/MCU))；MineExplorer 用多 agent 流程设计带隐藏前置的多跳任务图，再人工核验 ([arXiv 2605.30931](https://arxiv.org/abs/2605.30931))。
- **判定**：程序化库存/世界状态检查是主流。模糊任务用学习/VLM 判官：MineCLIP 阈值；GPT-4o 看视频六维打分、声称 91.5% 人类一致 ([MCU](https://arxiv.org/pdf/2310.08367))。BASALT 用人类成对比较 → TrueSkill，换到 MTurk 后平局率从 80% 降到 27–44% ([回顾](https://arxiv.org/pdf/2303.13512))。
- **接口**：两派。像素 + 低层键鼠（MineRL/MineDojo、STEVE-1、JARVIS-1、Optimus）vs 结构化文本 + **代码即动作**（Voyager、Odyssey、GITM）。JARVIS-1、Optimus 用 LLM 规划器 → STEVE-1 文本条件控制器桥接 ([JARVIS-1](https://arxiv.org/pdf/2311.05997))。
- **问题**：终身学习（Voyager）、长程记忆（JARVIS-1；Optimus-1 称记忆带来 2–6 倍提升）、错误识别/创造性（MCU）、多 agent 协作（TeamCraft 未见场景 <50%，[arXiv 2412.05255](https://arxiv.org/html/2412.05255v1)；MineCollab 通信是瓶颈，[arXiv 2504.17950](https://arxiv.org/abs/2504.17950)）。
- **批评**：MCU 作者："Minecraft agent 没有统一 benchmark，每个 agent 在自己的任务集上评"；MineDojo 很多创造性任务"连人都做不到"。MineCLIP 奖励对目标距离不敏感、可被盯天空刷分 ([arXiv 2303.10571](https://arxiv.org/pdf/2303.10571))。Minecraft 慢且重，直接催生了 Crafter/Craftax。版本碎片化（MineRL 1.16.5 / MineDojo 1.11 / Mineflayer）使跨论文数字不可比 **[推断]**。
- **代码结构**：MineDojo `minedojo.make(task_id)` + YAML 任务规格；Voyager 的 agent 仓库与环境分离，技能库 = 代码文件 + 描述向量索引；MineStudio 明确拆成 simulator/data/model/pretrain/finetune/inference/benchmark 七块，Ray 异步评测 ([arXiv 2412.18293](https://arxiv.org/html/2412.18293v1))；MCU 任务配置与 VLM 评测器分离。

### 2.2 Crafter / Craftax
- 固定 22 成就 + 按 seed 的程序化地图，成就依赖成树，"解锁很多成就需要又深又广的探索" ([poster](https://danijar.com/asset/crafter/poster.pdf))。Craftax 65 成就、4 档、地牢楼层，JAX 快 250 倍 ([arXiv 2402.16801](https://arxiv.org/abs/2402.16801))。
- 分数 = **各成就成功率的几何平均**，稀有成就权重更大；人类专家 50.5%。
- 批评：Stanić 等（ICML 2022 wkshp）证明**前馈 agent 靠读库存显示就能解锁几乎全部成就**，简单基线调参后超过先前方法 ([ICML 2022](https://icml.cc/virtual/2022/20334))；PPO 一小时内达到 Craftax-Classic 最优的 90%，Classic 对 RL 已接近饱和。BALROG 对 Crafter 用成就百分比而非几何平均，跨论文数字不可比。
- 结构：pip 安装的 Gym 环境，`info["achievements"]` 字典；agent 在单独仓库。环境/agent 分离是这里的常态。

### 2.3 NLE / BALROG
- BALROG 复用 6 个环境，统一 prompt 和 16 步历史 ([arXiv 2411.13543](https://arxiv.org/html/2411.13543v1))。NetHack 进度用**人类对局数据校准**：地牢层/经验等级 → 最终通关概率，取两条曲线最大值；最佳模型 o1-preview 约 1.5%。
- 发现："知行差距"（模型说吃腐肉危险然后去吃）；**视觉常常有害**（Claude 3.5 Sonnet 视觉模式 29.08% vs GPT-4o 文本 32.34%）。
- 批评：NetHack Challenge 2021 符号 bot 大幅击败深度 RL ([报告](https://nethackchallenge.com/report.html))，固定规则下手写先验占优；Gemini 在 TextWorld 得 0 分是因为安全 API 拦截输入，属于 API 侧混杂因素。
- 结构：`agent.py`（prompt 策略）/ `client.py`（模型 API 抽象）/ `env_wrapper.py`（统一环境接口）/ evaluator，加模型只改配置。**是 LLM 游戏 benchmark 里最干净的 agent/env/evaluator 三分，可直接当模板。**

### 2.4 ALFRED / ALFWorld / TEACh / EmbodiedBench / EAI
- ALFRED 任务用 PDDL 定义；ALFWorld 从同一 PDDL 生成 TextWorld 孪生，去掉分步指令只留目标 ([arXiv 2010.03768](https://arxiv.org/pdf/2010.03768))。EmbodiedBench 六个子集里 "Long Horizon" 定义为 >15 步 ([arXiv 2502.09560](https://arxiv.org/html/2502.09560v3))。
- 判定：二元 SR + **目标条件完成比例** + 路径长度加权；EAI 改为模块级检查（目标解释、子目标分解、动作排序、转移建模）+ 错误分类（幻觉、affordance、规划） ([NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b631da756d1573c24c9ba9c702fde5a9-Abstract.html))。
- 发现："没有视觉数据，空间任务掉 40–70%"。
- 批评：120 个场景的 seen/unseen 划分容易被场景记忆 **[推断]**；EAI 只覆盖符号状态不覆盖感知。

### 2.5 BabyAI / MiniGrid / Procgen
- 显式课程（19 级）；Procgen 用 train/test seed 划分同时测样本效率和泛化 ([ICML 2020](https://proceedings.mlr.press/v119/cobbe20a.html))。对现代 agent 过于简单（Craftax 作者原话）。seed → 关卡的映射是"held-out seed 泛化"的范本。

### 2.6 VirtualHome / BEHAVIOR-1K
- BEHAVIOR-1K 用 **BDDL** 写 1,000 个活动，`:init` + `:goal` 谓词，对象绑定 WordNet synset；每步跑谓词检查器；主指标 = **满足的目标谓词比例** ([B1K challenge](https://behavior.stanford.edu/challenge/archive/2025/call_for_participation.html))。
- 批评：第三方 Behavior-Skill 论文不得不添加基础 BDDL 无法表达的技能谓词 ([arXiv 2608.30536](https://arxiv.org/pdf/2608.30536))，谓词词汇表是可检查性的上限；OmniGibson（Isaac Sim）很重。
- 结构：任务规格语言与模拟器解耦，sampler 从谓词实例化初始状态。**是本次调研中最强的"任务即数据"模式。**

### 2.7 开放世界游戏 agent（Pokémon / VideoGameBench / lmgame / Cradle / Lumine）
- PokéAgent RPG 赛道：固定剧情、**15 个标准化里程碑**，排名先比完成率再比时间和动作数 ([arXiv 2603.15563](https://arxiv.org/html/2603.15563))。VideoGameBench 10 个商业游戏 + 3 个保密测试游戏，检查点进度 ([arXiv 2505.18134](https://arxiv.org/pdf/2505.18134))。Lumine 141 任务分四类，蒙德域内、璃月/稻妻域外，另有 5 小时主线；**人工评测**，每任务 5 次 ([arXiv 2511.08892](https://arxiv.org/html/2511.08892))。Cradle 手选 40 分钟 RDR2 剧情任务 ([ICML 2025](https://mlanthology.org/icml/2025/tan2025icml-cradle))。
- 接口：原始画面 + 实时按键（VGB；PokéAgent 加少量 RAM 状态）；Lumine 720p@5Hz 入、键鼠 30Hz 出，推理时暂停游戏；lmgame 的 harness 可开关感知/记忆/推理模块 ([ICLR 2026](https://arxiv.org/pdf/2505.15146))。
- 批评：**harness 混杂**。PokéAgent 报告裸 VLM 约 0%，CLI harness 几千步后失去连贯；Claude vs Gemini 玩宝可梦的社区讨论基本是在比脚手架而非模型 ([rentry](https://rentry.org/gpp-about-20250612))。实时玩法惩罚推理延迟（VGB 加了暂停模式）。GameBench 发现 GPT-4 有时低于随机 ([arXiv 2406.06613](https://arxiv.org/pdf/2406.06613))。TextStarCraft II 把观测转文本，测的是文本适配器上的策略而非感知。
- **截至今天没有找到公开的、针对原神的标准化 benchmark**；Lumine 是唯一的原神 agent 论文，人工评测、未发布 harness **[仅项目页，未核实]**。

### 2.8 文本 / 网页长程 agent benchmark（对照）
- WebArena：241 模板 → 812 意图；信息型任务 `exact_match / must_include / fuzzy_match`，状态改变型用程序化定位器（DB、API、JS） ([arXiv 2307.13854](https://arxiv.org/html/2307.13854))。
- OSWorld：369 任务，每个 = 初始状态配置 + 执行式评测脚本 ([arXiv 2404.07972](https://arxiv.org/abs/2404.07972))。
- TheAgentCompany：175 个 Docker 任务镜像，各带 `init.sh / eval.py / task.md`；分数 = 主结果 + **子检查点**部分得分；agent 模型和环境 NPC 模型分开配置 ([repo](https://github.com/TheAgentCompany/TheAgentCompany))。
- τ-bench：最终 DB 状态比对标注目标；**pass^k** 可靠性 ([arXiv 2406.12045](https://arxiv.org/abs/2406.12045))。
- 结论：网页世界已收敛到 **任务 = (初始配置, 指令, 评测脚本)** 这一自包含单元，加子检查点部分得分和重复试验可靠性指标。

### 2.9 开放式与探索
- Go-Explore（cell 存档、先返回再探索；Nature 2021），批评是 cell 表示需要观测空间的详细知识。OMNI / OMNI-EPIC 用基础模型判断"有趣度"选任务，OMNI-EPIC 直接**用代码生成环境**并随学习进度调难度 ([ICLR 2025](https://arxiv.org/pdf/2405.15568))。Eureka 让 GPT-4 从环境源码写奖励代码 ([ICLR 2024](https://arxiv.org/pdf/2310.12931))。Genie 3 实时生成世界但只有几分钟一致性且无 ground-truth 状态，不能做检查器 ([DeepMind](https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/))。
- 模式：LLM 做课程生成器，前提是环境暴露可条件化的程序化状态和程序化验证器；Voyager 的 LLM 自验证是已知弱点 **[推断]**。

### 2.10 3D 引擎 / 程序化世界平台
- UnrealZoo（100+ 写实 UE 世界，ICCV 2025）、Holodeck（GPT-4 + 51k Objaverse 资产生成 AI2-THOR 场景，CVPR 2024）、ProcTHOR（程序化房屋）、LEGENT（Unity，大型室外会变慢）、Obstacle Tower（程序化楼层、未见实例评测）、**Craftium**（Minetest + Gymnasium，每个环境 = 世界 + 定义奖励/终止的 Lua mod，支持无限程序化世界，[arXiv 2407.03969](https://arxiv.org/abs/2407.03969v1)）。
- Craftium 的"任务 = 世界 + 脚本定义奖励/终止"与 BEHAVIOR 的 BDDL 是 3D 世界里最干净的两种声明式任务模式。

### 2.11 长程规划评测本身
- PlanBench：o1 有"量级提升"但"随长度迅速退化"，作者呼吁加成本/效率/保证指标 ([arXiv 2409.13373](https://arxiv.org/html/2409.13373v1))。
- *Illusion of Diminishing Returns*（NeurIPS 2025 / ICLR 2026）：把执行从规划里隔离出来；每步准确率的小提升会复合成指数级更长的可完成任务；**自我条件化**：模型看到上下文里自己之前的错误后更容易出错，规模不能修、thinking 能缓解 ([arXiv 2509.09677](https://arxiv.org/pdf/2509.09677))。
- METR 时间视野：agent 50% 成功率对应的人类任务时长，2019 年以来约 7 个月翻倍，2023 后约 4.3 个月 ([arXiv 2503.14499](https://arxiv.org/pdf/2503.14499v1))。
- 含义：要报告**每步 / 每里程碑准确率和"视野 vs 成功率"曲线**，不只是终局成功；记录失败是规划还是执行；控制上下文以应对自我条件化。

### 2.12 综述与元批评（2024–26）
- *A Survey on LLM-Based Game Agents* ([arXiv 2404.02039](https://arxiv.org/abs/2404.02039))；*Survey on Evaluation of LLM-based Agents* ([arXiv 2503.16416](https://arxiv.org/abs/2503.16416v1))；*Intelligent Automation for Embodied Benchmark Construction*（2026-06）：自动化"把成本转移到验证、可审计性、版本控制和长期治理" ([arXiv 2606.12207](https://arxiv.org/abs/2606.12207))。
- *Agentic Benchmark Checklist*：任务有效性 / 结果有效性 / 报告规范；审计的 10 个 benchmark 里 7 个结果有效性有缺陷；τ-bench 让一个平凡 agent 通过了 38% 的任务 ([arXiv 2507.02825](https://arxiv.org/pdf/2507.02825))。*AI Agents That Matter*（TMLR 2025）：成本受控的 Pareto 评测；没有 holdout 就会 shortcut 过拟合 ([arXiv 2407.01502](https://arxiv.org/pdf/2407.01502))。*BenchGuard*（COLM 2026）：LLM 审计发现规格损坏和僵硬的评测脚本 ([arXiv 2604.24955](https://arxiv.org/abs/2604.24955))。

## 3. 综合

**任务构造的反复出现模式**
1. **科技树 / 依赖图**（Minecraft、Crafter、Craftax）：视野 = 合成 DAG 的深度；难度档 = 材料层级 + 人类推导的步数预算。
2. **固定叙事上的里程碑链**（PokéAgent 15 个里程碑、VGB 检查点、Cradle/Lumine 章节）：天然适合 RPG，但单一固定世界招致记忆。
3. **声明式目标谓词**（BDDL、PDDL/ALFRED、τ-bench DB 状态）：任务 = init + goal；sampler 实例化初态；部分得分自然产生。
4. **按 seed 程序化采样任务**（Procgen、Crafter、Craftax、TeamCraft 55k 变体、ProcTHOR/Holodeck）：泛化 = held-out seed / 场景。
5. **LLM 生成任务 + 验证**（MCU、MineExplorer、Voyager、OMNI-EPIC）：扩覆盖，但成本转到验证。
6. **能力切片子集**（EmbodiedBench 六子集、EAI 四模块、lmgame 的 harness 开关）用于定位失败。

**评测的反复出现模式**：二元成功 + **部分得分**（目标条件比例、BDDL 谓词比例、TAC 子检查点、PokéAgent 里程碑百分比）；**对人类数据归一化的进度**（BALROG 通关概率曲线、BASALT 按任务 TrueSkill、Crafter 几何平均加权稀有成就）；**效率**（平均步数/时间、路径加权 SR）；**可靠性**（τ-bench pass^k、Optimus-1 ≥30 seed、Lumine 每任务 5 次）；**成本**（Kapoor 等的 Pareto）。判官：状态可见就用程序化检查；VLM 判官（MCU）和人类（BASALT、Lumine）只补模糊/创造性目标。

**反复出现的批评**：成就/奖励被浅层策略刷分（Crafter 读库存、MineCLIP 盯天空）；固定世界记忆与弱 holdout；VLM/LLM 判官噪声且一致性协议不公开；人工评测贵、平局率高；**harness 混杂**（脚手架压过模型身份；"每个 agent 评自己的任务集"）；评测器本身的结果有效性 bug（ABC、BenchGuard）；实时延迟把速度和能力混在一起；像素评测昂贵；API 侧混杂（安全过滤清零）；长上下文的自我条件化。

## 4. 对 Teyvat 式开放世界 benchmark 的启示

1. **任务即数据。** 采用 BDDL / Craftium 式声明规格（YAML/JSON）：`init`（seed、区域、队伍、库存、任务 flag）、`goal` 为世界状态谓词合取、`milestones` 为有序或 DAG 的谓词列表、`budget`（步数/时间，像 Optimus-1 那样来自人类参考运行）、`tier`。部分得分 = 满足的里程碑比例；同时报二元成功。
2. **把视野显式编码成子目标 DAG，不只是剧情线。** 混合科技树式合成链、任务链式里程碑、生存谓词；给每个任务标 DAG 深度，画"成功率 vs 深度"曲线（METR / Sinha 式）。
3. **程序化世界 + seed 划分 + held-out。** 训练/开发 seed 公开，测试 seed/区域保留（Lumine 的蒙德 vs 璃月模式）；可留几个保密任务（VGB）。
4. **程序化检查器优先，判官其次。** 评测器作为独立插件读模拟器状态（WebArena/OSWorld 的逐任务脚本；BDDL 每步检查器）。VLM 判官只用于创造性/审美目标，并公布人类一致性协议和样本量。
5. **agent、env、evaluator 三分**（BALROG 的 `agent.py / client.py / env_wrapper.py` / evaluator；TheAgentCompany 的 agent-LLM 与 env-LLM 分开配置）。暴露 Gymnasium 式 `reset(seed, task_id) / step(action)` 加一层文本/技能 API，让同一组任务能跑像素级、结构化状态、代码即动作三种 agent，并在结果表里标注观测/动作制度。
6. **分层接口并如实报告。** 像素 + 低层控制（Lumine/VGB）、结构化状态 + 技能库（Voyager / EmbodiedBench 的 prompt 内技能表）、纯文本孪生（ALFWorld）。ALFWorld 模式能廉价消融感知 vs 规划；BALROG 表明视觉可能有害。
7. **轨迹日志与重放。** 记录完整动作/观测轨迹、seed、渲染设置（AGENTS.md 已有此约定）；确定性：同 seed 同世界；展示渲染从轨迹离线 Cycles 渲染，绝不为展示重新模拟。
8. **可靠性与成本指标。** 重复试验的 pass^k；每任务 ≥N 个 seed；排行榜上标注每次运行的 token/美元成本；报告推理延迟制度（暂停 vs 实时）。
9. **诊断而不只是排名。** EAI 式模块检查、EmbodiedBench 式能力子集、错误分类（感知/定位 vs 规划 vs 执行；知行差距；自我条件化）。
10. **审计 benchmark 本身。** 跑平凡/随机/脚本基线抓任务有效性泄漏（ABC：平凡 agent 过了 τ-bench 38%）；每任务有人类参考完成；对世界和评测器做版本固定。
11. **队伍 / NPC 机制是空白地带。** 调研的 benchmark 没有一个在 3D 开放世界里用程序化检查器评测队伍管理或 NPC 任务对话；最接近的是 TEACh（对话 → 状态改变）和 MineCollab（通信瓶颈）。把 NPC/任务状态定义成谓词，对话结果才可检查。

**未核实 / 空白**：Optimus-1 会议；BEHAVIOR-1K 会议（凭记忆 CoRL 2022）；NLE 精确任务表；Cradle 成功检查细节；TeamCraft 仓库地址；未找到公开的原神专用 benchmark（Lumine 是唯一原神 agent 论文，人工评测，未发布 harness）。Obstacle Tower 和 Holodeck 细节仅来自摘要/仓库。
