# HU(两人桌)serving 部署清单(serve/app.py + iter330 + 宽库)

与 3 人服务(`app3`)**互相独立**:各自一个进程/端口(HU 建议 8000,3 人 8001);可同机也可分机。
HU 与 3 人的**唯一大区别**:HU **要 138G 的宽库 `gto_lib_wide.db`**(翻/转查库,iter330 验收就是带库的);3 人不用库。

## 一、要打包的代码(小,一条 tar)

在训练机项目根 `alphaholdem-python/` 下:

```bash
cd /path/to/alphaholdem-python
tar czf hu_deploy.tgz \
  --exclude='__pycache__' --exclude='*.pyc' --exclude='*.log' --exclude='*.bak' \
  serve/app.py serve/bot.py serve/engine_state.py serve/selftest.py serve/requirements.txt \
  engine \
  network.py encoder.py config.py play_vs_bot.py exploit_layer.py preflop_guard.py \
  preflop_table.py river_solver.py \
  solver_data \
  runs/spec_mid_L4A6/model_iter330.pt
```

清单说明(依赖闭环):
- **serving**:`serve/{app,bot,engine_state,selftest}.py` + `requirements.txt`。
- **共享底座**:`engine/`、`network.py`、`encoder.py`、`config.py`、`play_vs_bot.py`(load_net)、
  `exploit_layer.py`、`preflop_guard.py`;兜底带 `preflop_table.py`、`river_solver.py`。
- **查库层**:整个 `solver_data/`(`gto_lib_db` 依赖 `gto_lib/board_match/gto_opponent/lib_to_sqlite`,都在此)。
- **HU 模型**:`runs/spec_mid_L4A6/model_iter330.pt`。

## 二、138G 宽库单独传(不进 tar)

`gto_lib_wide.db` 太大,单独用 rsync 传到服务器(断点续传):
```bash
rsync -avP gto_lib_wide.db  user@server:/data/gto_lib_wide.db
```
- **可选:不带库也能跑**——`DB_PATH` 不设时,HU 走纯 NN(无翻/转查库)。iter330 裸底座本就 7 风格全正,
  只是少了库的翻/转 GTO 覆盖(命中约 50-65%),对最难的平衡 reg(A)略有影响。想省 138G 传输可先不带库上线、后补。

## 三、【不要】打包

- `.venv/`(服务器重装)、`hu-data/`(677M 训练数据)、`threemax/`(3 人那套,除非同机也部署 3 人)、
  其它 `runs/`、`gto_libdir*`(库构建中间产物)。
- `TexasSolver-v0.2.0-MacOs/`(Mac 二进制,Linux 跑不了;河解默认关,不需要——见第六节)。

## 四、服务器环境

```bash
pip install torch numpy
pip install -r serve/requirements.txt      # fastapi / uvicorn / pydantic
```

## 五、启动(HU 定版配置)

```bash
tar xzf hu_deploy.tgz
MODEL_PATH=runs/spec_mid_L4A6/model_iter330.pt \
DB_PATH=/data/gto_lib_wide.db \
EXPLOIT=1 PREFLOP_GUARD=1 RIVER_SOLVE=0 \
python3 -m uvicorn serve.app:app --host 0.0.0.0 --port 8000
```
- `DB_PATH` 指向 138G 宽库(翻/转查库);不设则纯 NN(见第二节)。
- `EXPLOIT=1 PREFLOP_GUARD=1`、`GREEDY=0`(采样,抗剥削)、`NORM=500`——与 iter330 验收口径一致。
- 覆盖 20-500bb;>500bb 超深筹未训,serving 端建议把有效筹码夹到 ≤500。

## 六、河解(可选,默认关,一般不用)

`RIVER_SOLVE=1` 才需要,且**必须 Linux 版 TexasSolver**(自带的是 Mac x86 二进制,服务器跑不了):
从官方 Releases 下 Linux 版 `console_solver`(含 `resources/`)放到 `SOLVER_DIR`,再:
```bash
RIVER_SOLVE=1 SOLVER_DIR=/opt/TexasSolver RIVER_ITERS=40 ... python3 -m uvicorn serve.app:app ...
```
iter330 不开河解就 7 风格全正,**河解非必需**,增益约 A +6bb/100、每次河牌卡几秒。默认关即可。

## 七、上线自检

```bash
# 无 torch 也能验局面重建:
python3 serve/selftest.py
# 带模型+库:
python3 serve/selftest.py --model runs/spec_mid_L4A6/model_iter330.pt --db /data/gto_lib_wide.db
# 起服务后:
curl -s localhost:8000/health
```

## 八、调用契约

**接口/入参/出参与 3 人版结构相同**,只是 HU 是**两人桌**:`start_stacks` 只有 2 个、`hero_seat`/`button`/`actor` ∈ `{0,1}`。
字段含义、`hand_id`/`table_id` 回传与去重、幂等说明,见 `serve/README.md`(HU 契约)与 `serve/DEPLOY_3max.md`(入/出参表)。
`/decide`、`/hand_end`(每手必调喂读数)、`/reset_table`(换对手清读数)、`/health` 四个接口一致。

## 九、两条产品线并存(同机)

| 服务 | 脚本 | 端口 | 模型 | 库 | 隔离 |
|---|---|---|---|---|---|
| HU | `serve.app` | 8000 | iter330 | `gto_lib_wide.db`(138G) | 独立进程 |
| 3 人 | `serve.app3` | 8001 | iter885 | 无(坍缩走训练网) | 独立进程 |

两个进程各自 `MODEL_PATH`/端口,互不影响;`exploit_layer` 等共享代码对两边都安全(HU 用默认 `station_loose_floor=0.40`,
3 人内部传 `0.0`)。
