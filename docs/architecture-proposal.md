# 重构方案：环境 / 任务 / 智能体 / 评测 分层

状态：设计提案；步 0–7 已于 2026-11-06/07 实施并合入本分支（见 §3 的逐步状态与 §6）。§1 保留对重构前布局的描述，作为动机记录。目标是让 OWB 在不改变任何模拟器行为的前提下，
变成"一个模拟器可以插进来、一个任务可以声明出来、一个 agent 可以换掉"的框架。

## 1. 现状诊断

### 1.1 两个顶层包，一套半接口

| 包 | 内容 | 问题 |
| --- | --- | --- |
| `open_world_bench/` | Teyvat 模拟器全部（terrain / world / party / quests / render / env），加上 `reference_tasks.py`、`reference_policy.py` | 名字和 `owb` 冲突；任务与参考策略和模拟器混在一个包里 |
| `owb/` | Wildlands、Skylands 模拟器（core / world / physics / sim / entities / env / render），加上 `bench/`（通用接口与 LLM baseline）、`tasks/minecraft.py`、`scripts/`、`render/presentation.py` | "通用层"住在某一个模拟器的包里，并在模块顶层 import 该模拟器 |

`owb.bench.framework` 在 import 时就依赖 `owb.env`（Wildlands 词汇表）和 `owb.sim.construction`（Skylands 建造常量）；
对 Teyvat 则在函数内 lazy import `open_world_bench`。也就是说通用层对三个后端的依赖方式各不相同。

### 1.2 任务定义有四种互不兼容的写法

| 任务 | 位置 | 形态 | 评测怎么算 |
| --- | --- | --- | --- |
| Teyvat 主线 "Windborne Ledger" | `open_world_bench/quests/{schema,engine,library}.py` | 声明式 `QuestGraph`（quest → stage → objective，带前置依赖），由 `QuestEngine` 消费事件流 | `quests.all_complete`、`completion()`，在 env.step 里算 |
| Teyvat 参考课程 A/B | `open_world_bench/reference_tasks.py`（591 行） | 命令式 Python：`Course` 既放置实体、又持有机制状态机、又判定里程碑；`ReferenceTaskEnv` 继承 env 并重写 `_build` | `course.verify()`、把每个 milestone 包成一个 `ACTIVATE` objective 塞回 `QuestGraph` |
| Wildlands 参考课程 A/B | `owb/env/wildlands_reference.py`（577 行） | 坐标元组列表 + `WildlandsReferenceEnv` 继承 env，机制状态机写在 env 子类里 | env 子类内部 |
| Minecraft | `owb/tasks/minecraft.py` | 声明式 `MinecraftTask` + 评测侧 `MilestoneTracker`，只看累计统计量 | 完全在 env 外 |

四种写法里只有 Minecraft 的是"env 不知道任务存在"的正确分层；Teyvat `QuestGraph` 其次。
两个参考课程把**关卡布置、机制逻辑、里程碑判定、特权状态**全部耦合进 env 子类，
这正是 AGENTS.md 里"改地形需要重新验证 ground truth"这类约束的来源：因为没有边界，改任何一处都可能影响全部。

### 1.3 Agent 层：传输方式和决策逻辑纠缠

`owb/bench/baselines/`：`OpenAIPolicy` 同时负责

1. 观测 → 文本 / 图像 prompt（`describe_observation`，直接 import Wildlands 的 `ITEM_IDS`、Skylands 的 `PART_COLUMNS`）
2. 动作 JSON schema 与校验（`action_schema` / `decode_action`）
3. 记忆（`memory` 字符串 + `history` deque）
4. 调用 OpenAI Responses API

`ClaudePolicy` / `CodexPolicy` / `GeminiPolicy` 通过**伪造一个 Responses 形状的 client** 继承它。
后果：README 说这个 benchmark 测的是 **memory**，但 memory 恰恰是唯一不能独立替换的组件；
想比较"无记忆 / 文本便签 / 文件系统记忆 / 检索记忆"必须复制整个 policy 类。

特权参考策略也散在两处：`open_world_bench/reference_policy.py` 和 `owb/bench/baselines/wildlands_reference.py`，且和 LLM baseline 放在同一个 `baselines` 命名空间下，容易误用。

### 1.4 通用 schema 是硬编码的

`BenchmarkEnv.observation_space` 固定列出 `party(30)`、`inventory(32,16)`、`construction(8,9)`……
每接一个新后端（mobile_robot、Minecraft）都要改中心表、升 `SCHEMA_VERSION`、改 `describe_observation`、改 `EpisodeRecorder` 的词汇表输出。
`_REGISTRY` 也是写死的三项，`register_environment()` 存在但没有任何 entry point 机制。

### 1.5 其他可维护性债

- `jsonable()` 三份：`owb/bench/framework.py`、`owb/scripts/teyvat_reference.py`、`owb/env/wildlands_reference.py`（叫 `plain`）。
- 三个脚本（`benchmark.py`、`teyvat_reference.py`、`wildlands_reference.py`）各自实现 argparse、录制、校验、哈希。
- `owb/render/scene_export.py` 用 `isinstance(env, OpenWorldEnv)` 分发并 import Teyvat 内部 mesh 模块；展示渲染层知道每个模拟器的内部。
- `EpisodeRecorder` 同时负责 JSONL 日志和 Cycles 视频，录制与渲染耦合。
- 配置：`BenchmarkConfig` + 每个模拟器自己的 `EnvConfig`，CLI 参数手写，没有可复现的 run manifest 文件。

好的方面（要保留）：`NativeAdapter` 这个思路是对的；`QuestEngine` 可以喂合成事件做单测；Minecraft tracker 的"评测只看统计量、agent 拿不到"原则；录制里的 `verification.json` / `state_digest` 哈希可以直接作为重构的回归金标准；测试已有 15 个文件。

## 2. 目标分层

```
owb/
  core/        通用工具：rng、jsonable、Event、版本号、配置基类          （不依赖任何模拟器）
  spaces/      Modality / Control 声明与可组合的 Dict space 构造           （不依赖任何模拟器）
  sims/        每个模拟器一个子包，纯"世界"：物理、实体、传感器渲染、事件
    teyvat/    ← 现 open_world_bench/（terrain, world, party, elements, character, camera, render）【步 1 已完成】
    voxel/     ← 现 owb/{core,world,physics,sim,entities,env,render(传感器部分)}；Wildlands 与 Skylands
               共用一个体素模拟器，Skylands 只是 worldgen + construction 变体，所以不拆成两个包【步 1 已完成】
    minecraft/ ← MineStudio 适配（可选依赖）
    mobile_robot/ ← RoboCasa 适配（可选依赖）
  envs/        把 sim 包成 Gymnasium env：BenchmarkEnv + 每个 sim 的 Adapter + 注册表
  tasks/       任务规格（声明式）+ 世界布置 hook + 评测器；不 import envs
    schema.py  TaskSpec / Milestone / Predicate
    evaluator.py  MilestoneEvaluator（消费事件 + 状态查询）
    teyvat/, wildlands/, minecraft/  各自的 spec 文件（YAML/JSON）+ 布置器 + 机制
  agents/      Agent 协议 + 可组合部件
    base.py      Agent(reset, act, metadata)
    perception/  obs → 模型输入（按 modality 注册的 describer）
    harness/     Harness 协议 + 实现：StatelessAPI / StatelessCLI(claude,codex,gemini) / SessionCLI / WorkspaceCLI
    policy/      LLMPolicy（感知 + harness + 动作解码，模型无关）
    oracle/      特权参考控制器（明确标注 privileged，读 task 内部状态）
  eval/        Runner、Recorder、Replay/verify、Aggregation、Budget
  presentation/ 场景导出协议 + Blender Cycles 渲染（消费 sim 提供的 SceneExporter）
  cli/         单一入口 `owb run|replay|verify|render|tasks|agents`
configs/       run manifest（YAML）：env + task + agent + budget + seeds
```

依赖方向只允许向下：`cli → eval → agents / envs / tasks / presentation → sims → spaces → core`。
`tasks` 和 `agents` 互不 import；`tasks` 不 import `envs`（只依赖 sim 暴露的事件和状态查询协议）。

### 2.1 Sim：世界只做世界

每个 sim 暴露同一个最小协议，不继承 gym：

```python
class Simulator(Protocol):
    cfg: Any
    def reset(self, seed: int) -> None: ...
    def tick(self, control: NativeControl, dt: float) -> list[Event]:  # 一步物理
    def sensors(self) -> dict[str, np.ndarray]:          # rgb / depth / 自有传感器
    def state(self) -> StateView:                        # 只读查询：pos, hp, inventory, time, entity(id)
    def world_hooks(self) -> WorldHooks:                 # 供任务布置器用：spawn_at / flat_patch / fill_box
    def scene_exporter(self) -> SceneExporter | None:    # 供 presentation 用
```

Teyvat 的 `OpenWorldEnv.step` 里现在混着：物理循环、`quests.consume`、reward shaping、discovery reward、终止判定。
重构后 `step` 只剩物理循环与事件收集；quest 进度、reward、终止全部移到 `tasks` 层。
Wildlands 的 `_generate_world / _reset_mechanics / _extra_entities` 这组 template-method hook 已经接近 `WorldHooks`，可以直接演化。

### 2.2 Spaces：按 modality 注册，不再中心硬编码

```python
@dataclass(frozen=True)
class Modality:
    name: str                 # "party", "inventory", "construction", "lidar" ...
    space: gym.Space
    describe: Callable[[np.ndarray], Any]   # 给 agents/perception 用的命名化描述
    vocab: dict | None = None               # ITEM_IDS 这类词汇表，随 recorder 写进 config.json

@dataclass(frozen=True)
class Control:
    name: str; space: gym.Space; doc: str
```

Adapter 声明 `modalities: tuple[Modality,...]` 和 `controls`；`BenchmarkEnv` 据此拼 Dict space 并自动生成 `modality_mask / command_mask`。
`describe_observation`、`EpisodeRecorder` 的词汇表、`action_schema` 都从同一份声明生成，新后端不再改中心文件。
`SCHEMA_VERSION` 保留，但只在共享字段（`move/look/proprio/instruction`）变化时递增；后端私有 modality 自己带版本。

### 2.3 Tasks：规格、布置、评测三分

```python
@dataclass(frozen=True)
class Predicate:           # 可序列化、可组合的完成条件
    kind: str              # event_count / stat_increase / region / inventory_has / state_equals / sequence / all / any
    args: dict

@dataclass(frozen=True)
class Milestone:
    id: str; title: str; stage: str
    done: Predicate
    requires: tuple[str, ...] = ()       # 里程碑 DAG
    hidden_from_agent: bool = True
    role: str = ""                       # learning / interference / transfer / composition（Teyvat/Wildlands 设计里的"记忆角色"）

@dataclass(frozen=True)
class TaskSpec:
    id: str; sim: str; title: str; instruction: str
    milestones: tuple[Milestone, ...]
    budget: Budget                        # steps / sim_seconds / model_calls / tokens
    setup: str                            # 布置器注册名，如 "teyvat.relay_course"
    setup_params: dict                    # 随 seed 采样的参数（element 排列等）由布置器产生并记入 manifest
```

- **TaskSpec** 用 YAML/JSON 写（类似 BEHAVIOR 的 BDDL、MCU 的 task yaml），Minecraft 两个任务可以原样转过来；Teyvat `QuestGraph` 是它的一个子集（stage + prerequisites）。
- **Setup（布置器）** `setup(sim.world_hooks(), rng, spec) -> SetupResult`：放实体、修地形、注册机制。现 `Course._build_a/_build_b` 和 `build_questline` 的"放东西"部分搬到这里。
- **Mechanism（机制）** relay / sentinel / vent / counterweight / flow network 这类状态机独立成模块，接收 `Event` 与 `StateView`，发出新 `Event`；sim 不知道它们存在。现在它们散在 `Course`、`CourseCombat`、`CourseWorld`、`WildlandsReferenceEnv` 里。
- **Evaluator** `MilestoneEvaluator(spec)`：`reset(state)`、`update(events, state, step) -> fired`、`progress`、`summary()`。Minecraft `MilestoneTracker` 直接是它的 `stat_increase` 实现。
- **TaskEnv wrapper**（在 `envs/`）：`TaskEnv(benchmark_env, spec)` 在 reset 时调布置器，在 step 时跑机制与评测器，把 `instruction` 写进 obs，把 `progress/success` 写进 info。agent 永远拿不到 evaluator 对象。

reward 变成 `tasks` 层一个可选的 `RewardShaper`（objective reward、shaping、death penalty 等现 `RewardConfig`），默认 sparse，使录制里的 reward 可解释。

### 2.4 Agents：harness 是研究变量，不是实现细节

拍板（2026-11-06）：OWB 要比较的 "memory" 不是 prompt 里的一段便签，而是 **coding-agent 式 harness**
——一个拥有工作目录、可持久化会话、可读写文件的 agent 运行时（MFM 的 filesystem memory 就是这种）。
因此 agent 层的核心抽象不是 `Memory`，而是 `Harness`：

```python
class Harness(Protocol):                         # agents/harness/
    """一个 episode 内持续存在的 agent 运行时。"""
    def open(self, episode: EpisodeContext) -> None     # 建 workspace、起会话、写入任务说明
    def decide(self, obs_parts: list[Part], schema: dict) -> Decision   # 一次决策
    def close(self) -> HarnessSummary                   # 回收；返回 workspace 快照、token/调用统计
    capabilities: HarnessCaps                           # 下面的沙箱声明

@dataclass(frozen=True)
class HarnessCaps:
    persistent_session: bool      # 跨决策保留模型上下文（现状：False，每步新进程）
    workspace: Path | None        # agent 可读写的目录（现状：None，临时目录且无工具）
    tools: frozenset[str]         # 允许的工具名（现状：空集 / 仅 StructuredOutput）
    network: bool                 # 现状：仅模型 API 本身
    max_turns_per_decision: int   # 现状：Claude 3
```

四类实现，构成记忆维度的 ablation 轴：

| Harness | 会话 | workspace | 工具 | 对应现状 |
| --- | --- | --- | --- | --- |
| `StatelessAPI` | 否 | 否 | 否 | `OpenAIPolicy`（便签 + 6 步历史全靠 prompt 重放） |
| `StatelessCLI` | 否 | 否 | 否 | `ClaudePolicy / CodexPolicy / GeminiPolicy` |
| `SessionCLI` | 是 | 否 | 否 | 新：`claude -p --resume` 之类，只保留上下文 |
| `WorkspaceCLI` | 是 | 是 | 文件读写（+可选 shell） | 新：MFM 式文件系统记忆 |

`Perception`（obs → Part 列表，用 Modality.describe 生成）和 `ActionDecoder`（schema + 校验，现 `action_schema/decode_action`）
与 harness 无关，独立复用。`LLMPolicy(harness, perception, decoder)` 是唯一的 Agent 实现；
现在的 `OpenAIPolicy` 变成 `LLMPolicy(StatelessAPI(openai))` 的别名。

**沙箱 / 行为限制**。统一方案（能力声明 schema、四个命名档位、OS 级 vs flag 级执行边界、本地模型代理）见 [agent-sandbox.md](agent-sandbox.md)。现状已有的（保留并上提到 `HarnessCaps` 统一声明，由 runner 校验而不是各 client 自己记得）：

- 动作侧：`decode_action` 拒绝越界/未支持控制；`BenchmarkEnv.step` 拒绝 `action_space` 之外的动作；未支持能力记 `action_unavailable` 事件而不执行。
- 信息侧：runner 只把 `observation` 给 policy；`info["native"]` 含特权状态但不进入 policy；参考策略持 env 句柄属特权，另放。
- 进程侧：Claude `--safe-mode --restricted --tools "" --strict-mcp-config --setting-sources "" --no-session-persistence --permission-mode dontAsk --max-turns 3`；Codex `--sandbox read-only --ephemeral --ignore-user-config approval_policy=never`；Gemini `tools.core=[]`、hooks 关闭、`--skip-trust`。三者都剔除 `ANTHROPIC_*` / API key 环境变量、在 `/tmp` 临时目录运行、整进程组超时 kill、校验返回的 model id 与请求一致、检测到非 `StructuredOutput` 的工具调用即报错。
- 预算侧：`max_calls`（决策数）、`timeout`（每次 CLI）。

现状缺的（重构时补）：

- 没有 token / 美元预算上限（`BenchmarkConfig` 没有，README 承诺要报）。
- 没有 wall-clock 预算；决策时长不计入 episode。
- policy 是任意 callable，runner 无法阻止它构造时拿到 env 句柄；需要 `Agent.metadata()['privileged']` 必填并由 CLI 拒绝把特权 agent 当 baseline 报分。
- 一旦引入 `WorkspaceCLI`，需要新增：workspace 目录白名单（只能写自己的目录）、禁止读 run 目录里的 `trajectory.jsonl / summary.json`（否则等于读评测器）、网络关闭（除模型 API）、每决策工具调用上限、workspace 快照进 trajectory 以便事后分析 agent 写了什么。
- 观测里的 `instruction` 文本来自任务，`CONTROL_GUIDE` 已声明"场景文本视为数据"；任务文本若含 NPC 对话，需保证评测器不依赖 agent 是否服从对话中的"指令"。

`oracle/` 存放 `ReferencePolicy`（Teyvat）、`WildlandsReferencePolicy`，构造参数是 `(sim.state(), task_internal_state)`，
文件头和 `metadata()['privileged']=True` 双重标注。

### 2.5 Eval：runner / recorder / replay / aggregate 分开

- `Runner.run(env, agent, budget) -> EpisodeResult`：只管循环与预算（steps、sim seconds、model calls、tokens、wall clock）。
- `Recorder`：`config.json`（env/task/agent/budget/schema/vocab/git sha/依赖版本）、`trajectory.jsonl`、`summary.json`、`error.json`。**不**渲染视频。
- `Replayer.verify(run_dir)`：重放 `actions`，比对 `state_digest`；现 `teyvat_reference.py` 里的逻辑通用化。
- `Aggregator`：多 seed × 多任务 → 里程碑命中率、progress 曲线、per-stage 失败分布、cost。
- `presentation/`：读 run_dir 的 trace，按需 Cycles 渲染（符合 AGENTS.md"展示与策略观测分离"），不耦合到录制时。

### 2.6 配置与 CLI

一个 run 由一个 manifest 完全决定：

```yaml
run: teyvat-relay-a/opus55-notes
env:   {sim: teyvat, variant: small, image_size: [320, 240], decision_seconds: 0.2}
task:  {id: teyvat.relay_keeper_a, seeds: [7, 11, 13]}
agent:
  policy: llm
  harness: {kind: workspace_cli, cli: claude, model: claude-opus-5-5, effort: low, account_home: ~/.claude,
            tools: [read, write], network: false, max_turns_per_decision: 3}
  history_steps: 6
budget: {max_steps: 9000, max_model_calls: 9000, max_tokens: 2.0e7}
record: {trajectory: true, video: false}
```

`owb run manifest.yaml`、`owb replay --verify run_dir`、`owb render run_dir --renderer cycles`、
`owb tasks list|show`、`owb agents list`。三个现有脚本收敛成这几个子命令。

## 3. 迁移路径（每步可独立合并、测试全绿）

| 步 | 动作 | 风险控制 |
| --- | --- | --- |
| 0 | 冻结行为（当前基线：2026-11-06 全量 `pytest` 238 通过、3 跳过；CLI policy 测试要求 PATH 上有 `python`）：把 `assets/demos/*/actions.jsonl` + `verification.json` 接成回归测试（重放 → 比 digest）；补一个"三个 env 随机策略 200 步的 obs/info 哈希"快照测试 | 之后每一步都用它兜底 |
| 1 | **已完成 2026-11-06。** `open_world_bench/` → `owb/sims/teyvat/`；`owb/{core,world,physics,sim,entities,env}` + 传感器渲染 → `owb/sims/voxel/`；`owb/_compat.py` meta-path finder 让所有旧路径带 DeprecationWarning 解析到同一对象；`tests/test_compat_imports.py` 24 个用例 | 全量测试 262 通过（原 238 + 24）；随机策略 smoke 输出与基线逐字节一致 |
| 2 | **已完成。** `owb/core`：唯一的 `jsonable`、`SCHEMA_VERSION`、`event_type`；旧 `owb.core.*` 子模块仍经 shim 解析到体素模拟器 | `tests/test_core.py` |
| 3 | **已完成。** `owb/spaces`：`Modality` / `Control` / `SpaceRegistry`，内建 7 modality + 9 control；`BenchmarkEnv` 的 space、四个 mask、recorder 的 `schema` 块、`describe_observation` 都由声明生成 | 接口快照哈希逐字节不变；未声明化的残留见 §6 |
| 4 | **已完成。** `owb/agents/{harness,perception,decoder,policy,oracle}`；四个 harness（OpenAI API、Claude/Codex/Gemini CLI）带 `HarnessCaps`；`LLMPolicy` + `Budgets(max_calls, max_total_tokens)`；旧 policy 名为兼容别名，argv/env/错误码逐项等价 | 旧 policy 测试全部未改动通过；`tests/test_agents.py` | 现有 `test_*_policy.py` 用假 client 做合同测试，可直接迁移 |
| 5 | **已完成。** `owb/tasks/{schema,evaluator,registry,env,setups}`；Minecraft 两个任务改为 YAML；Teyvat 问题线映射并与 `QuestEngine` 逐步等价；两个参考课程拆成 `tasks/{teyvat,wildlands}/mechanisms/*`、`setup.py`、`course.py` 与四个 TaskSpec YAML；特权控制器移到 `owb/agents/oracle` | 四条参考轨迹 digest 逐字节不变；`tests/test_task_*.py`、`test_reference_specs.py` |
| 6 | **已完成。** `owb/eval/{runner,recorder,sinks,replay,policies}`：录制只写 JSON，视频经 `FrameSink`；`Budget(max_model_calls, max_wall_clock_s)` 以 `termination_reason=budget` 干净结束；`verify_run(run_dir)` 通用重放校验；`policy.close()` 必调 | 三个 env 的 config/trajectory/summary/gif 与重构前逐字节一致 |
| 7 | **已完成。** `owb` 控制台命令：`run / replay --verify / tasks / agents / envs / render`；manifest（YAML）校验并写入 run 目录；`benchmark.py` 变为弃用薄包装；特权策略需 `--allow-privileged` | `tests/test_cli.py` 44 例；`configs/examples/*.yaml` |

步 1–3 约一两天；步 4 一天；步 5 最长，两个参考课程各半天到一天；步 6–7 一天。

## 4. 拍板记录（2026-11-06）

| 点 | 决定 |
| --- | --- |
| 包名 | 统一到 `owb`；`open_world_bench` 留 re-export shim 一个版本。 |
| 任务规格格式 | YAML 文件 + dataclass 加载器。目标是可扩展、人能维护：一个任务一个文件，diff 可读，机制与布置器仍是 Python。接口同理：Modality / Control 用声明而不是中心表。 |
| 参考课程 | 见下。先按"演示 + 回归金标准"处理：原样迁到 `tasks/<sim>/reference_courses/`，其机制模块（relay / sentinel / vent / counterweight / flow）抽成可复用件；是否把它们升级为正式任务本体，等 YAML 任务规格落地后再定。 |
| reward | 保留为 `RewardShaper` 可选件，默认关（sparse）。 |
| memory | 不是便签接口，而是 coding-agent 式 harness（会话 + workspace + 工具），见 2.4。 |

**"参考课程"是什么。** `open_world_bench/reference_tasks.py` 和 `owb/env/wildlands_reference.py` 里的 Course A/B：
把 `docs/teyvat-long-horizon-tasks.md`、`docs/wildlands-long-horizon-tasks.md` 设计的 12/24 个里程碑任务，
压缩成一个可执行的紧凑关卡（蛇形排布的 8×N 个场地、授权的机制状态机），再用一个**特权参考控制器**
（读任务内部状态、但只能发普通动作）从头走完，录下动作轨迹、每个里程碑的 `state_digest` 和 `verification.json`。
它们证明"这些任务在模拟器里是可完成的、机制是可验证的"，并生成了 README 里的演示视频；
它们**不是** agent 要解的正式任务套件（完整版还是设计稿），也不是学习型 agent 的基线。
对重构而言，它们最大的价值是回归金标准：重放 `actions.jsonl` 必须复现 digest。

## 5. 与调研结论的对应

详见 [related-benchmarks.md](related-benchmarks.md)。本方案采纳的外部惯例：Gymnasium env + 独立 evaluator（Crafter / MineDojo / BALROG）、
声明式任务文件（BEHAVIOR BDDL、MCU yaml）、里程碑部分得分而非二元成功（MCU、Craftax）、seed 决定世界 + 任务参数并写入 manifest、
特权 oracle 与 baseline 严格分开、trace 可重放校验。

## 6. 实施结果与剩余事项（2026-11-07）

**测试**：全量 460 通过、3 跳过（跳过：缺 `../MFM` 对照、缺 `cv2`）。重构前的 309 个上游测试全部未改动通过；
新增 151 个：兼容导入 33、金标准 11、agents 8、tasks 25、core/spaces/eval 18、CLI 44、参考课程 12。

**行为冻结证据**：四条参考轨迹（Teyvat A/B、Wildlands A/B）逐步 digest 一致；三个通用 env 150 步 + 三个原生 gym id 60 步的观测/info 哈希快照一致；随机策略 smoke 的录制产物与重构前逐字节一致；
`import owb, owb.*` 在 `-W error::DeprecationWarning` 下干净，旧路径各发一次弃用警告并解析到同一对象。

**尚未做、按优先级**：

1. **Sim 协议（§2.1）**：Teyvat `OpenWorldEnv.step` 仍混有 `quests.consume`、reward shaping、终止判定；`RewardShaper` 尚未抽出。Wildlands/Skylands 的 `StateView` 为空。
2. **`TaskEnv` 不执行 `spec.setup`**：目前只有两个参考课程 env 自己调用布置器。参考课程里仍有按里程碑 id 硬编码的事实（A02/B02 提示、B10/B11 授权、B18 温度依赖等）待迁入 setup 数据。
3. **Predicate 表达力**：持续时间条件（"稳定 5 秒"）、episode 末仍需成立的活世界条件、合法提前完成的计分、机制状态谓词 `state_equals`。Wildlands 课程的里程碑因此仍以机制自己发的 `milestone_complete` 事件触发。
4. **沙箱统一（agent-sandbox.md §4）**：manifest 已接受 `profile / network / persist_workspace` 等键，但只校验与 harness 固定 caps 相等，不做 OS 级执行；本地模型代理、`cli-session` / `cli-workspace` 档位未实现。
5. **预算**：`max_total_tokens` 由 `LLMPolicy` 抛异常结束而非经 runner 干净截断；美元预算未加。
6. **声明化残留**：`describe_observation` 的 prompt 键序、`action_schema` 的手写边界、`build_move` 需 construction 的规则、modality 版本号未写入 `config.json`（为保持哈希不变而保留）。
7. **仍有的重复**：`owb/scripts/teyvat_reference.py` 的 `jsonable` 与 `owb/tasks/wildlands/mechanisms/base.py` 的 `plain`（后者额外对 set 排序）。
8. **文档**：`docs/embodied-agent.md`、`rendering.md`、`skylands.md` 的命令示例仍是 `python -m owb.scripts.benchmark`（仍可用）。
9. **已知环境差异**：Teyvat 特权控制器在本机实时运行时 `look` 有最后一位 float32 噪声（717/1118 步，最大 4.8e-7），重构前已如此，疑为 numpy/BLAS 差异；重放录制动作不受影响。

