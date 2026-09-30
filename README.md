# PokerBot HU(两人桌)服务 —— 部署 / 启动 / 接口文档

模型 **iter330**(AlphaHoldem,7 风格验收全正 +56.5) + 翻/转 **GTO 查库**(`gto_lib_wide.db`)+ 护栏 + 剥削层,
封装成跨语言、多桌可调用的 HTTP 服务(`serve/app.py`)。默认端口 8000。与 3 人服务(`app3`,:8001)完全独立。

---

## 一、部署文件(最简自包含)


```
AlphaHoldem-HU/
├── serve/            app.py bot.py engine_state.py __init__.py requirements.txt
├── engine/           cards.py evaluator.py hunl_env.py __init__.py
├── solver_data/      gto_lib_db.py 及其依赖(查库层;不连库可整个删)
├── network.py encoder.py config.py play_vs_bot.py exploit_layer.py preflop_guard.py
│                     preflop_table.py river_solver.py           (共享底座)
└── runs/spec_mid_L4A6/model_iter330.pt                          (HU 定版模型)
```

**不含**:138G 的 `gto_lib_wide.db`(单独 rsync 传服务器 `/data/`)、TexasSolver(要开河解才装,见第五节)、
以及 `.npz/.jsonl/hu-data/runs 快照` 等一切训练/解算产物(运行时不用)。

运行时只加载两个数据文件:**模型 `.pt`** +(可选)**库 `.db`**。

## 二、环境安装(Linux / macOS,Python 3.10+)

```bash
pip install torch numpy                 # CPU 版即可,决策每步 ~毫秒
pip install -r serve/requirements.txt   # fastapi / uvicorn / pydantic
```

## 三、启动命令

```bash

# (A) 带库(推荐,=验收口径,A 档最强)
MODEL_PATH=runs/spec_mid_L4A6/model_iter330.pt DB_PATH=/data/gto_lib_wide.db \
EXPLOIT=1 PREFLOP_GUARD=1 \
python3 -m uvicorn serve.app:app --host 0.0.0.0 --port 8000

# (B) 纯 NN(不带库,启动快、省内存;A 档略弱几 bb)
MODEL_PATH=runs/spec_mid_L4A6/model_iter330.pt \
python3 -m uvicorn serve.app:app --host 0.0.0.0 --port 8000
```

看到 `[serve] 模型=... 库=... 河解=... 就绪` + `Uvicorn running on http://0.0.0.0:8000` 即成功。

> ⚠ **关于 `gto_lib_wide.db`(约 136GB)**:体积过大**无法随包上传/分发**,不在 `AlphaHoldem-HU/` 里,也不在代码仓库。
> **需要带库(方式 A)部署的,请联系 Eric 获取该库文件**,放到服务器后把 `DB_PATH` 指向它。
> 若暂时拿不到库,可先用**方式 B(纯 NN)**上线,库到位后加 `DB_PATH` 重启即可。

### 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `MODEL_PATH` | (必填) | 模型 `.pt` 路径 |
| `DB_PATH` | 空 | 138G 宽库路径;不设=纯 NN(翻/转不查库) |
| `NORM` | `500` | 归一化基准(=训练筹码上限);覆盖 20~500bb |
| `GREEDY` | `0` | 0=采样(更像真人、抗剥削);1=贪心 |
| `EXPLOIT` | `1` | 剥削层(读对手 VPIP/PFR/AF,对站砍诈唬等) |
| `PREFLOP_GUARD` | `1` | 翻前护栏(深筹垃圾牌大额投入降级) |
| `RIVER_SOLVE` | `0` | 河牌现解(需 TexasSolver,见第五节) |
| `SOLVER_DIR` | `TexasSolver-v0.2.0-MacOs` | TexasSolver 目录(开河解才用) |
| `RIVER_ITERS` | `40` | 河解迭代数 |

> 覆盖 20~500bb;`start_stacks` >500bb(超深筹)未训、决策不保证,serving 端建议夹到 ≤500。

## 四、上线自检

```bash
python3 serve/selftest.py                                          # 无 torch:局面重建
python3 serve/selftest.py --model runs/spec_mid_L4A6/model_iter330.pt --db /data/gto_lib_wide.db
curl -s localhost:8000/health          # {"ok":true}
```

## 五、河解 TexasSolver(可选,默认关)

`RIVER_SOLVE=1` 才需要;`river_solver.py` 会调用 TexasSolver 的 `console_solver` 二进制实时现解河牌树。

- **官方地址**:https://github.com/bupticybee/TexasSolver  → Releases 页下载对应平台版本
  (https://github.com/bupticybee/TexasSolver/releases)。
- **Linux 服务器**:下 **Linux 版**(解压后含 `console_solver` 可执行文件 + `resources/` 目录),放到某目录:
  ```bash
  # 例:放到 /opt/TexasSolver,里面有 console_solver 和 resources/
  chmod +x /opt/TexasSolver/console_solver
  RIVER_SOLVE=1 SOLVER_DIR=/opt/TexasSolver RIVER_ITERS=40 \
  MODEL_PATH=runs/spec_mid_L4A6/model_iter330.pt DB_PATH=/data/gto_lib_wide.db \
  python3 -m uvicorn serve.app:app --host 0.0.0.0 --port 8000
  ```
- **本地 macOS**:项目自带 `TexasSolver-v0.2.0-MacOs/`(x86 二进制;Apple Silicon 自动 `arch -x86_64` 走 Rosetta,
  需 `softwareupdate --install-rosetta`)。`SOLVER_DIR=TexasSolver-v0.2.0-MacOs`。
- ⚠ **Mac 二进制在 Linux 跑不了**,反之亦然,按服务器平台下对应版本。
- **建议**:一般**不开**。iter330 不开河解就 7 风格全正;河解只给 A 约 +6bb/100,却让**每次河牌决策卡几秒**。

## 六、接口一览

| 方法 | 路径 | 何时调 | 作用 |
|---|---|---|---|
| POST | `/decide` | 轮到机器人行动 | 返回该做的动作 |
| POST | `/hand_end` | **每手结束(必调)** | 传本手双方全部动作,更新对手读数 |
| POST | `/reset_table` | 对手换人/离桌 | 清该桌对手读数 |
| GET | `/health` | 探活 | `{"ok":true}` |

## 七、入参(`/decide` 与 `/hand_end` 同体)

牌用 `"Ah"`:rank ∈ `23456789TJQKA`,suit ∈ `shdc`。**HU 为两人桌**:座位 `0/1`,`start_stacks` 两个。

| 字段 | 类型 | 必填 | 默认 | 说明 / 备注 |
|---|---|---|---|---|
| `table_id` | string | 否 | `"default"` | 桌号;多桌用不同值隔离对手读数 |
| `hand_id` | string | 否 | `null` | 手牌 id(同桌可多手)。出参回传;`/hand_end` 按 `(table_id,hand_id)` 去重;不传则不去重 |
| `hero_seat` | int | **是** | — | 机器人座位 `0/1` |
| `button` | int | **是** | — | 按钮(=小盲)座位 `0/1`;翻前先动、翻后有位置 |
| `sb` | float | 否 | `0.5` | 小盲(bb 为单位) |
| `bb` | float | 否 | `1.0` | 大盲 |
| `start_stacks` | float[2] | **是** | — | 两家本手**下盲前**起始筹码 `[seat0,seat1]`(bb)。>500 超深筹未训 |
| `hero_hole` | string[2] | **是** | — | 机器人两张底牌,如 `["As","Ad"]` |
| `board` | string[] | 否 | `[]` | 公共牌 `0/3/4/5` 张;张数须与最深 street 一致 |
| `actions` | Action[] | 否 | `[]` | 到当前为止的**有序**动作;`/hand_end` 须含**本手全部**动作 |

**Action(`actions` 元素):**

| 字段 | 类型 | 必填 | 说明 / 备注 |
|---|---|---|---|
| `street` | int | **是** | `0`翻前 `1`翻牌 `2`转牌 `3`河牌 |
| `actor` | int | **是** | 动作者座位 `0/1` |
| `type` | string | **是** | `fold`/`check`/`call`/`bet`/`raise`/`allin` |
| `to` | float | `bet/raise` 必填 | 该街**累计下注到**多少 bb(非本次加注额);其它可省 |

## 八、出参(`/decide`)

| 字段 | 类型 | 说明 / 备注 |
|---|---|---|
| `action` | string | `fold`/`check`/`call`/`raise`/`allin` —— 机器人该做的动作 |
| `to` | float \| null | 该街**累计下注到**多少 bb(raise/call/allin);check/fold 见 `add` |
| `add` | float | 还需**投入**多少 bb(供你在自己引擎落地) |
| `bucket` | int | 内部离散动作号(调试用) |
| `meta.source` | string | `library`(查库 GTO)/ `nn`(模型)/ `river_solve`(河牌现解) |
| `meta.opp_label` | string\|null | `tag`/`station`/`maniac`/`nit`/`null` —— 剥削层对对手判定 |
| `meta.guards` | string[] | 本步触发的护栏,如 `["preflop","commit","air_fold","exploit","river_nut"]` |
| `meta.street` | int | 决策所在街 `0/1/2/3` |
| `table_id` / `hand_id` | string | 原样回传(供服务端关联 / 校验幂等) |

`/hand_end` 返回 `{"ok":true}`;`/reset_table`(body `{"table_id":"t1"}`)返回 `{"ok":true}`;`/health` 返回 `{"ok":bool}`。

> **幂等**:`/decide` 默认采样、非确定性。网络重试请服务端按 `(table_id,hand_id,决策点)` 缓存首次响应、勿重复调 bot;
> `/hand_end` 由 bot 侧按 `(table_id,hand_id)` 去重,重复调不重复计读数。

## 九、调用示例

```bash
# 探活
curl -s localhost:8000/health

# /decide —— 翻牌一手(hero=BB seat1,按钮 seat0 翻前加注、hero 跟,翻牌 hero 先动)
curl -s localhost:8000/decide -H 'Content-Type: application/json' -d '{
  "table_id":"t1","hand_id":"20260930-0001","hero_seat":1,"button":0,
  "sb":0.5,"bb":1.0,"start_stacks":[200,200],
  "hero_hole":["As","Ad"],"board":["Qs","Jh","2h"],
  "actions":[{"street":0,"actor":0,"type":"raise","to":3},{"street":0,"actor":1,"type":"call"}]}'
# → {"action":"check","to":0.0,"add":0.0,"bucket":1,
#    "meta":{"source":"library","opp_label":"unknown","guards":[],"street":1},
#    "table_id":"t1","hand_id":"20260930-0001"}

# 一手结束(传双方全部动作,喂对手读数)——必调
curl -s localhost:8000/hand_end -H 'Content-Type: application/json' -d '{
  "table_id":"t1","hand_id":"20260930-0001","hero_seat":1,"button":0,
  "start_stacks":[200,200],"hero_hole":["As","Ad"],"board":["Qs","Jh","2h","7c","9d"],
  "actions":[{"street":0,"actor":0,"type":"raise","to":3},{"street":0,"actor":1,"type":"call"},
             {"street":1,"actor":1,"type":"check"},{"street":1,"actor":0,"type":"check"},
             {"street":2,"actor":1,"type":"check"},{"street":2,"actor":0,"type":"check"},
             {"street":3,"actor":1,"type":"check"},{"street":3,"actor":0,"type":"check"}]}'

# 对手换人 → 清读数
curl -s localhost:8000/reset_table -H 'Content-Type: application/json' -d '{"table_id":"t1"}'
```

## 十、多桌 & 运营要点

- 每桌用不同 `table_id` → 对手读数各自独立。
- 每手结束**必调 `/hand_end`**(传全手动作),否则剥削层学不到对手、退化为纯模型。
- 换对手/离桌调 `/reset_table`。
- 单进程用锁串行推理,中小并发够;高并发多进程 `--workers N` 时**同一桌必须路由到同一进程**(否则对手读数分裂)——
  用网关按 `table_id` 一致性哈希。
- 高并发也可直接多开单进程 + 加大机器,按 `table_id` 分流。
