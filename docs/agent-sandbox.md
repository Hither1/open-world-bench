# Agent 沙箱与能力声明：他人怎么做，OWB 怎么统一

调研日期 2026-11-06。回答的问题：其他 agent benchmark 如何限制被测 harness，OWB 应把现有分散在
`ClaudeClient / CodexClient / GeminiClient` 里的 ad hoc 限制统一成什么。配套 [重构方案 §2.4](architecture-proposal.md)。

## 1. 他人的做法

| 系统 | 隔离单元 | 文件系统 | 网络 | 工具 | 预算 | 特权信息隔离 | 来源 |
|---|---|---|---|---|---|---|---|
| Inspect AI (UK AISI) | `sandbox()`：docker / local / k8s / modal 等，每个样本一个 | 默认禁止可写宿主挂载；建议 pin 镜像 digest | 默认 `network_mode: none`；k8s 默认拒绝出站，`allowDomains` 白名单 | 只有经 `sandbox()` 的操作被隔离 | `time_limit / working_limit / message_limit / token_limit / turn_limit / cost_limit`，超限硬停仍计分 | scorer 和 tool 一样走 `sandbox()` | [sandboxing](https://inspect.aisi.org.uk/sandboxing.html), [limits](https://inspect.aisi.org.uk/setting-limits.html) |
| METR Task Standard / Vivaria / HCAST | 任务容器 | agent 跑在非特权 `agent` 用户下，**scoring 以 root 跑**；agent 看不到 scoring 代码 | 无 `full_internet` 权限时只放行少数目的地（LLM API） | — | HCAST 10 小时、2M/8M token 上限 | agent "MUST NOT" 看到 `get_tasks` 数据；aux VM 默认无 SSH | [STANDARD.md](https://github.com/METR/task-standard/blob/main/STANDARD.md), [HCAST](https://arxiv.org/html/2503.17354v1) |
| Harbor / Terminal-Bench 2.0 | 每任务 Dockerfile | `tests/` **在 agent 结束后才注入**；`verifier.environment_mode = separate` 用全新沙箱跑测试 | `network_mode = no-network / public / allowlist` | — | `agent.timeout_sec`、`verifier.timeout_sec`、cpus/memory/gpus | 自动检查拒绝烘进测试或答案的 Dockerfile | [verifier](https://docs.harborframework.com/tasks/verifier), [TB2](https://arxiv.org/html/2601.11868v1) |
| SWE-bench + SWE-agent ACI | base/env/instance 三层镜像，patch 在全新容器里评 | ACI 只暴露固定小命令集（open/scroll/search/edit/submit），100 行窗口 | — | 每实例 $4，超了自动 submit | 测试在 agent 之外跑 | [harness](https://swebench.com/SWE-bench/reference/harness), [SWE-agent](https://arxiv.org/html/2405.15793) |
| OpenHands | client-server：agent 通过 REST 向容器内 `ActionExecutor` 发动作 | bind / named / copy-on-write 挂载 | — | bash / Jupyter / Playwright 三种 | 每实例一个容器，记录工具调用与结束原因 | — | [runtime](https://docs.openhands.dev/usage/architecture/runtime) |
| MLE-bench | 容器 + 1 GPU，24 小时 | 容器内只有格式校验服务器，**不返回分数**；评分在外部对重划分的测试集 | 论文要求报告是否联网而非统一禁网 | — | 24 小时 | Dolos 抄袭检查 + LLM 日志扫描；论文明说安全"不依赖容器隔离" | [paper](https://arxiv.org/html/2410.07095) |
| TheAgentCompany | OpenHands 沙箱 | — | 自建 GitLab/OwnCloud 等 | — | — | NPC 用另一个 LLM；checkpoint 评测器在环境状态上跑，未描述与 agent 的隔离 | [paper](https://arxiv.org/html/2412.14161v3) |
| OSWorld | VM 快照 | getter 把文件/cookie/UI 状态拉出 VM，**评测脚本在宿主跑** | — | pyautogui 代码 + WAIT/FAIL/DONE | 15 步 | 排行榜由维护者在自己的基础设施上跑 | [paper](https://arxiv.org/html/2404.07972) |
| WebArena / BrowserGym | 每站点一个 Docker 镜像，重启即重置 | — | — | 默认高层动作集，原始 Playwright 可选，因为任意代码"有安全风险" | — | `validate()` 在 harness 侧算 reward；AgentLab 记录包版本 | [BrowserGym](https://arxiv.org/html/2412.05467) |
| τ-bench | 仅 API | — | — | 唯一接口是带类型的 DB 工具 + 消息通道 | — | 模拟用户看不到工具调用；reward = 最终 DB 状态比对 | [paper](https://arxiv.org/html/2406.12045) |
| BALROG | 仅 API | — | — | 无工具；自由文本映射到固定动作集，非法动作记日志并用默认动作 | 16 步历史 | — | [paper](https://arxiv.org/html/2411.13543) |
| lmgame-Bench | — | — | — | **harness 本身是被测变量**：感知 / 记忆 / 推理模块独立开关；无 harness 时约 40% 的运行打不过随机，有 harness 86.7% 能 | — | DSPy 标准化 prompt 以减少 prompt 方差 | [paper](https://arxiv.org/html/2505.15146v2) |
| PokéAgent | 任意方法 | — | — | 不硬禁，**强制披露五个维度**：State（原始像素 / 解析状态 / 特权信息）、Tools、Memory、Feedback、Fine-tuning | — | 只给部分 RAM 状态（队伍种类/等级/HP），谜题状态、道具、招式表保留 | [speedrunning](https://pokeagentchallenge.com/speedrunning.html), [paper](https://arxiv.org/html/2603.15563) |
| HAL (Princeton) | 本地 / docker / Azure VM | agent 是任意 `run(input) -> dict` | — | — | Weave 记录每次 API 调用、token、美元 | trace 加密上传防污染；要求精确模型版本号 | [paper](https://arxiv.org/html/2510.11977) |
| MFM | — | 结构化状态由 harness 写入 `index.jsonl / grid.jsonl`（"模型不写它们"）；agent 自己的文件在 `mine/`，跨生命持久 | — | `episodes.py` 查询 + `view_image` | `working.md` 40 行、400 步计划上限 | **未描述** memory 与里程碑状态之间的访问边界 | [paper](https://arxiv.org/html/2610.22686) |
| Voyager | — | — | — | 动作 = 对 Mineflayer bot 执行的 JS；成功由 GPT-4 critic 判 | — | 论文未记录任何防特权命令的措施 | [paper](https://arxiv.org/html/2305.16291) |

## 2. 弱沙箱导致的事故

- **SWE-bench Verified 未来提交泄漏**（2025-09）：`git log --all`、远程分支、reflog、未来 tag、`git fsck --lost-found` 都暴露了修复；Claude 4 Sonnet、Qwen3-Coder 均有命中；v4.1.0 剥离未来历史后修复，受影响轨迹按失败计 ([issue #465](https://github.com/SWE-bench/SWE-bench/issues/465))。
- **METR o3**：沿 Python 调用栈拿到 scorer 预计算的参考输出、monkey-patch 计时函数、禁用 CUDA 同步、改评估函数让所有提交通过；RE-Bench 7/164、HCAST 8/1000+ 次，全部计 0 ([报告](https://metr.org/evaluations/openai-o3-report/))。
- **HAL 日志分析**：agent 在 HF/arXiv 上找到金标答案、硬编码通过单测、有提交的 scaffold 把 benchmark 样例写进 few-shot prompt ([paper](https://arxiv.org/html/2510.11977))。
- **Terminal-Bench 2.0** 对抗 agent：monkey-patch 测试环境、穷举答案；对策是 Dockerfile 扫描 + 事后注入测试 ([TB2](https://arxiv.org/html/2601.11868v1))。
- **Pokémon 直播**：RAM 导出的可行走覆盖层和小地图让模型比较失去意义；PokéAgent 的回应是强制披露而非禁止。

## 3. 反复出现的设计维度

| 维度 | 取值范围 | 实际由谁执行 |
|---|---|---|
| 隔离单元 | 仅 API → 进程 → 容器 → VM | OS / 容器 |
| 文件系统范围 | 仅 workspace；agent 用户 vs root scorer；测试事后注入；只读挂载 | OS |
| 网络 | 无 / 仅模型端点 / 白名单 / 公网 | OS；CLI 的沙箱挡命令不挡 CLI 自己的模型调用 |
| 工具白名单 | CLI flag / harness 侧动作映射 / 带类型 API | flag + harness 过滤 |
| 预算 | 步数、墙钟、token、美元；硬停 vs 只记录 | Inspect / Harbor 硬停仍计分；HAL 只报告 |
| 特权信息 | scorer 用 root / 独立 verifier 沙箱 / 宿主侧 getter / 隐藏任务数据 | OS + harness |
| 模型身份 | HAL 要求精确版本；PokéAgent 披露表；**没有人校验 API 返回的 model id，除了本仓库** | harness |
| 日志 | 每次模型调用 + 每个动作 | harness |
| 可复现 | 镜像 digest、SDK 提交号、VM 快照、包版本 | runner |
| harness 作为变量 | lmgame 开关、PokéAgent 五轴披露、HAL scaffold 解耦 | benchmark 设计 |

## 4. OWB 的统一方案

### 4.1 能力声明 `capability.yaml`（runner 用 dataclass 校验，写进 `config.json` 与排行榜行）

```yaml
profile: cli-workspace                 # 下面的命名档位
isolation: process | sandbox-exec | docker        # 执行者：runner
filesystem:
  workspace: /work                     # 唯一可写根                         (OS)
  read_only: [/opt/owb/assets]         # 额外只读根                         (OS)
  deny: [<run_dir>, ~, ~/.claude, ~/.codex, ~/.gemini]   # 必须不可见    (OS)
  persist_workspace: none | episode | run | suite        # 记忆作用域    (runner)
network:
  mode: none | model-endpoint | allowlist                  # (OS；模型端点经本地代理)
  allowed_hosts: [api.anthropic.com]
tools:
  builtin: []                          # CLI flag + harness 拒绝意外调用
  mcp: []                              # --strict-mcp-config / --disallowedTools "mcp__*"
  env_actions: [move, look, interact, ...]   # 只能通过带类型的动作 API 影响世界 (harness)
observation:
  channels: [rgb, proprio, instruction, party, inventory]  # harness 渲染，不给模拟器句柄
  privileged: false                    # 禁止读真值 / 里程碑 / 评测器状态
budgets:                               # hard=停止 episode 仍计分；soft=只记录
  model_calls:  {max: 9000,    hard: true}
  env_steps:    {max: 9000,    hard: true}
  wall_clock_s: {max: 14400,   hard: true}
  tokens_total: {max: 2.0e7,   hard: false}
  usd:          {max: 50,      hard: false}
model:
  id: claude-opus-5-5
  verify: every_response               # 每次响应核对 model id（现有行为，保留）
  env_keys_stripped: true
logging:
  model_calls: full                    # 请求 + 响应 + token + 延迟
  tool_calls: full
  fs_diff: true                        # episode 前后 workspace 哈希
  image_digest: sha256:...             # 或 sandbox-exec profile 文本哈希
```

### 4.2 命名档位

| 档位 | 允许 | 必须记录 | 排行榜列 |
|---|---|---|---|
| **api-stateless** | runner 自己的客户端直接调 Messages/Responses API；无 shell、无文件；历史窗口由 runner 定（BALROG / τ-bench 式） | 每次调用、token、model id | model、history_len、obs channels、是否触预算 |
| **cli-stateless** | 每个决策起一次 coding-agent CLI，工具全关，空临时 cwd 用完即删，网络仅模型端点（= 现状） | 同上 + stdout/stderr、退出码、每步墙钟 | + harness 名/版本、max_turns |
| **cli-session** | 同上，但 cwd 在一个 episode 内持久（CLI 可留自己的 session 文件），仍无文件工具 | + 每步 fs_diff | + 持久作用域 |
| **cli-workspace** | 在 `/work` 内开放 read / write / grep 文件工具（MFM 式文件系统记忆）；可选 `persist_workspace: run`；shell 默认关；网络仅模型端点 | + 完整工具调用日志、fs_diff、workspace 大小上限 | + 工具列表、持久作用域、结束时 workspace 大小 |
| **open** | 超出以上（shell、网络白名单、特权观测）；像 PokéAgent 一样以披露代替禁止，单独一栏 | 全部 | 全部五轴 |

### 4.3 哪些必须 OS 级执行，哪些 flag 够用

**必须 OS 级**（CLI 自身的权限模型可绕过，或 CLI 本身不可信）：

- run 目录（trajectory、summary、评测器输出、参考轨迹）不得从 agent cwd 可读。Docker 不挂载即可；`sandbox-exec` 用 `(deny file-read* (subpath "<run_dir>"))` + deny-default。这就是 METR 的 root-vs-agent 和 Harbor 事后注入测试的原则。
- 网络：除模型端点外全拒。Claude `--tools ""`、Gemini `tools.core` 都不阻止进程开 socket；Codex 沙箱挡命令但 CLI 自己要连 API。做法：模型调用走 **本地代理**（只放行 loopback），代理顺带做 model id 校验、调用计数（超限返回 429）、每次请求响应落盘到 `run_dir/model_calls.jsonl`。
- 凭据：只保留代理需要的一个 key，其余环境变量清空；`sandbox-exec` 会把整个环境透传。
- 硬预算：墙钟靠进程组 kill（现有），模型调用上限在代理层执行而不是信任 `--max-turns`。
- 用户配置隔离：`~/.claude`、`~/.codex`、`~/.gemini`、项目 `.claude/`、`AGENTS.md`、`GEMINI.md` 必须不可见，否则 CLI 会加载 hooks / MCP / skills。Claude `--restricted` 只加载 managed settings 和 `--settings`，并把文件工具限制在工作目录（需 v2.1.248+）；仍应挂一个空 HOME。

**flag 够用**：

- 工具面：Claude `--tools ""`（注意**不**移除 MCP 工具，需加 `--disallowedTools "mcp__*"` 和 `--strict-mcp-config`）、`--no-session-persistence`、`--max-turns N`、`--max-budget-usd`、`--safe-mode`；Codex `exec --sandbox read-only --ephemeral -a never --skip-git-repo-check --json`；Gemini `tools.core`、`tools.exclude`、`tools.sandbox`、`tools.sandboxNetworkAccess`、`admin.secureModeEnabled`。
- 观测塑形：harness 渲染 agent 能看到的东西，永不把模拟器句柄交给 agent（τ-bench / BrowserGym 模式）。
- 拒绝意外工具调用、校验 model id：无论哪个档位都在 harness 代码里做（现状已有，保留）。

### 4.4 实现骨架

```python
def launch(caps, cmd, workspace, run_dir):
    env = {"PATH": "/usr/bin:/bin", "HOME": str(workspace / "home"),
           "ANTHROPIC_BASE_URL": "http://127.0.0.1:8765"}      # 走本地代理
    if caps.isolation == "docker":
        argv = ["docker", "run", "--rm", "--read-only", "--tmpfs", "/tmp",
                "--network", "none" if caps.network.mode == "none" else "owb-proxy-net",
                "-v", f"{workspace}:/work", "-w", "/work",
                "--cpus", "2", "--memory", "4g", "--pids-limit", "256", caps.image_digest, *cmd]
    elif caps.isolation == "sandbox-exec":
        argv = ["sandbox-exec", "-p", render_sb_profile(workspace, allow_loopback=True,
                                                         deny=[run_dir, Path.home()]), *cmd]
    else:
        argv = cmd
    p = subprocess.Popen(argv, cwd=workspace, env=env, start_new_session=True, stdout=PIPE, stderr=PIPE)
    try:
        out, err = p.communicate(timeout=caps.budgets.wall_clock_s.max)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        raise BudgetExceeded("wall_clock")
```

`sandbox-exec` 已被 Apple 弃用且无文档（Codex 仍在用），适合本地开发；排行榜级别用 Docker，把镜像 digest 或 profile 文本哈希写进结果行。

### 4.5 排行榜最小字段

model_id（已校验）· harness 名 + 版本 · profile · 观测通道 · privileged · persist_workspace · 工具列表 · model_calls 用量/上限 · env_steps 用量/上限 · 墙钟 · tokens · usd · 是否触预算 · image_digest / profile_hash · seeds。
这对应 HAL 的成本优先报告和 PokéAgent 的五轴披露，让 "harness 是被测变量"（lmgame）成为一等比较项。

## 5. 对现有代码的映射

| 现状 | 去向 |
|---|---|
| 三个 `*Client` 各自拼 CLI flag、剔环境变量、校验 model id、拒绝工具调用 | 合并为 `HarnessCaps` 声明 + 一个 `launch()`；flag 由 caps 生成，校验逻辑共享 |
| `max_calls`、`timeout` 散在 `OpenAIPolicyConfig` | `budgets` 块，runner 统一执行并写 `budget_hit` |
| 没有 token / 美元 / 墙钟预算 | 加入 `budgets`，先 soft 记录，排行榜必填 |
| policy 可拿 env 句柄 | `Agent.metadata()['privileged']` 必填；CLI 拒绝把 `privileged=True` 的 agent 当 baseline 报分 |
| 临时 `/tmp` cwd，用完即删 | `cli-stateless` 档位；`cli-session` / `cli-workspace` 新增，workspace 放 run 目录**之外**并做 fs_diff |
| 无模型调用代理 | 新增本地代理（可选）：计数、429 硬停、model id 校验、`model_calls.jsonl` |

未核实项：METR 是否在独立容器跑 scoring（只查到 root-vs-agent 用户分离）；Codex 沙箱内部实现（Seatbelt/Landlock）来自第三方；Gemini CLI "shell 字符串匹配可绕过"的警告仅见于文档镜像。
