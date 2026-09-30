"""TexasSolver 输出 JSON → 翻后软目标 JSONL(BC 就绪)。

遍历 solver 输出的博弈树:每个 action_node 就是一个"轮到某人决策"的局面。对该 player 范围里的每一手
组合,solver 给了一个混合策略(各动作概率)。我们把它转成一条 BC 训练样本:
  某方(hero)在【board + 翻前后的下注历史 + 筹码】下,拿【某手牌】时的【目标动作分布(软标签)】。

输出 JSONL 每行:
  {"id","depth_bb","eff_bb","board","street","pot","player",
   "seq":[[player,bucket,street],...],          # hero 决策前的翻后动作历史(bucket=我们的动作档)
   "hand":"AsAh",                                # 具体两张(solver 组合)
   "strategy_raw":{"CHECK":0.6,"BET 33":0.4},   # solver 原始标签→概率
   "strategy_bucket":{"1":0.6,"4":0.4}}         # 映射到我们动作档→概率(软目标)

用法:
  # 批量(配合 solver_gen 的 manifest):
  python3 solver_parse.py --manifest manifest.json --out solver_postflop.jsonl
  # 单文件验证(手动给 meta):
  python3 solver_parse.py --single some_result.json --board Qs,Jh,2h --pot 5 --eff 200 --depth 200 --out t.jsonl
"""
import argparse
import json
import os
import random
import sys
import time

_SAMPLE = 1.0   # 行抽样保留概率(1.0=全留);main 里按 --sample 设置

# 复用主工程的档位定义
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import RAISE_FRACTIONS, FOLD, CALL, ALLIN  # noqa: E402

_FR = sorted(RAISE_FRACTIONS.items(), key=lambda kv: kv[1])  # [(bucket,frac)...]


def frac_to_bucket(frac):
    if frac >= 3.5:
        return ALLIN
    return min(_FR, key=lambda kv: abs(kv[1] - frac))[0]


def parse_label(label):
    """'BET 33.0' -> ('BET',33.0);'CHECK'->('CHECK',0);'ALLIN'->('ALLIN',0)。"""
    parts = label.strip().split()
    kind = parts[0].upper()
    amt = float(parts[1]) if len(parts) > 1 else 0.0
    return kind, amt


def label_to_bucket(label, pot_before, to_call, my_contrib):
    kind, amt = parse_label(label)
    if kind in ("CHECK", "CALL"):
        return CALL
    if kind == "FOLD":
        return FOLD
    if kind == "ALLIN":
        return ALLIN
    # BET/RAISE 统一用【到额/底池】—— 与 bc_etl / preflop_etl 训练侧完全同口径(否则合并/微调时加注标签冲突)
    if kind in ("BET", "RAISE"):
        return frac_to_bucket(amt / max(pot_before, 1e-9))
    return CALL


def apply_action(label, pot, contrib, player):
    """返回动作后的 (pot, contrib) 副本。contrib=该街两人已投。"""
    kind, amt = parse_label(label)
    c = list(contrib)
    opp = 1 - player
    if kind == "CHECK":
        pass
    elif kind == "CALL":
        add = max(c[opp] - c[player], 0.0); c[player] += add; pot += add
    elif kind == "FOLD":
        pass
    elif kind == "BET":
        c[player] += amt; pot += amt
    elif kind == "RAISE":
        add = max(amt - c[player], 0.0); c[player] = amt; pot += add
    elif kind == "ALLIN":
        pass
    return pot, c


def find_children(node):
    """兼容不同字段名:action_node 用 'childrens';chance_node 用 'dealcards'/'childrens'。"""
    for k in ("childrens", "dealcards", "deal_cards", "children"):
        if isinstance(node.get(k), dict):
            return k, node[k]
    return None, {}


def walk(node, ctx, seq, fout, cnt):
    """流式:每产生一行立即写 fout(不在内存堆积),cnt=[计数]。"""
    if not isinstance(node, dict):
        return
    nt = node.get("node_type", "")
    if nt.startswith("action"):
        player = node.get("player")
        strat = node.get("strategy", {}) or {}
        combo_probs = strat.get("strategy", {})
        actions = strat.get("actions") or node.get("actions") or []
        pot_before = ctx["pot"]
        contrib = ctx["contrib"]
        to_call = max(contrib[1 - player] - contrib[player], 0.0)
        # 每个动作 → 我们的档位(既给决策软目标用,也给历史 seq 用)
        lab2bkt = {a: label_to_bucket(a, pot_before, to_call, contrib[player]) for a in actions}
        if combo_probs and actions:
            board_s = ",".join(ctx["board"])
            for combo, probs in combo_probs.items():
                if len(probs) != len(actions):
                    continue
                if _SAMPLE < 1.0 and random.random() > _SAMPLE:
                    continue                         # 行抽样:按概率丢弃,控总量
                raw = {actions[i]: round(float(probs[i]), 6) for i in range(len(actions))}
                bkt = {}
                for i, p in enumerate(probs):
                    b = lab2bkt[actions[i]]
                    bkt[b] = bkt.get(b, 0.0) + float(p)
                row = {
                    "id": ctx["id"], "depth_bb": ctx["depth_bb"], "eff_bb": ctx["eff"],
                    "board": board_s, "street": ctx["street"],
                    "pot": round(pot_before, 3), "player": player,
                    "seq": seq, "hand": combo,
                    "strategy_raw": raw,
                    "strategy_bucket": {str(k): round(v, 6) for k, v in bkt.items()},
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                cnt[0] += 1
        # 递归子节点。seq 记 [player, bucket, street](档位而非原始标签,方便 ETL 直接重建历史)
        _, children = find_children(node)
        for label, child in children.items():
            npot, ncontrib = apply_action(label, ctx["pot"], ctx["contrib"], player)
            nctx = dict(ctx, pot=npot, contrib=ncontrib)
            hbkt = lab2bkt.get(label, CALL)
            walk(child, nctx, seq + [[player, hbkt, ctx["street"]]], fout, cnt)

    elif nt.startswith("chance"):
        _, deals = find_children(node)
        for card, child in deals.items():
            nboard = ctx["board"] + [card]
            nctx = dict(ctx, board=nboard, street=ctx["street"] + 1, contrib=[0.0, 0.0])
            walk(child, nctx, seq, fout, cnt)
    # terminal / showdown → 停


def walk_lib(node, ctx, path, lib):
    """导库遍历:把每个 action 节点的【全量策略】存进 lib(不抽样、不转档,保留 solver 原始标签+概率)。
    节点 key = 'board|动作线'(board 含已发的翻/转牌,动作线=到该点走过的原始 label 序列),
    board 变化(转牌发出)+ 动作线一起唯一确定一个决策点。查表时按同样规则重建 key 命中。"""
    if not isinstance(node, dict):
        return
    nt = node.get("node_type", "")
    if nt.startswith("action"):
        player = node.get("player")
        strat = node.get("strategy", {}) or {}
        combo_probs = strat.get("strategy", {})
        actions = strat.get("actions") or node.get("actions") or []
        if combo_probs and actions:
            key = ",".join(ctx["board"]) + "|" + ">".join(path)
            lib["nodes"][key] = {
                "player": player,
                "board": ",".join(ctx["board"]),
                "street": ctx["street"],
                "pot": round(ctx["pot"], 3),
                "actions": list(actions),
                "strat": {combo: [round(float(p), 5) for p in probs]
                          for combo, probs in combo_probs.items() if len(probs) == len(actions)},
            }
        _, children = find_children(node)
        for label, child in children.items():
            npot, ncontrib = apply_action(label, ctx["pot"], ctx["contrib"], player)
            walk_lib(child, dict(ctx, pot=npot, contrib=ncontrib), path + [label], lib)
    elif nt.startswith("chance"):
        _, deals = find_children(node)
        for card, child in deals.items():
            walk_lib(child, dict(ctx, board=ctx["board"] + [card],
                                 street=ctx["street"] + 1, contrib=[0.0, 0.0]), path, lib)
    # terminal/showdown → 停


def dump_lib_one(js_path, meta, out_dir):
    """一个已解 flop(dump-2)JSON → 一份 per-board 紧凑策略库(gzip json)。返回 (节点数, 路径)。"""
    import gzip
    with open(js_path) as f:
        tree = json.load(f)
    board = [c.strip() for c in meta["board"].split(",") if c.strip()]
    start_street = max(len(board) - 2, 1)
    ctx = {"board": board, "street": start_street, "pot": float(meta["pot"]), "contrib": [0.0, 0.0]}
    lib = {"meta": {"id": meta["id"], "board": meta["board"], "flop": ",".join(board[:3]),
                    "pot": meta["pot"], "eff": meta.get("eff"), "depth_bb": meta.get("depth_bb")},
           "nodes": {}}
    walk_lib(tree, ctx, [], lib)
    del tree
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, meta["id"] + ".lib.json.gz")
    with gzip.open(path, "wt", encoding="utf-8") as g:
        json.dump(lib, g, ensure_ascii=False)
    return len(lib["nodes"]), path


def _collect_gametruth(node, board, path, out):
    """独立复走原始 solver 树,按【游戏真实到达方式】记录每个 action 节点:
    board=已发牌(chance 处增长)、path=走过的原始 label 序列。刻意不复用 walk_lib,
    这样能与导库/服务侧的 key 构造互为独立验证(错位就会对不上)。"""
    if not isinstance(node, dict):
        return
    nt = node.get("node_type", "")
    if nt.startswith("action"):
        strat = node.get("strategy", {}) or {}
        cp = strat.get("strategy", {}) or {}
        acts = strat.get("actions") or node.get("actions") or []
        if cp and acts:
            out.append((list(board), list(path), list(acts), cp))
        _, children = find_children(node)
        for label, child in children.items():
            _collect_gametruth(child, board, path + [label], out)
    elif nt.startswith("chance"):
        _, deals = find_children(node)
        for card, child in deals.items():
            _collect_gametruth(child, board + [card], path, out)


def verify_lib(js_path, lib_path, n_sample=300, tol=2e-4, seed=0):
    """导库正确性校验:独立复走原始 JSON,对抽样节点用【服务层 gto_lib.query】回读库,断言概率一致。
    能抓:节点 key 错位、combo 写法不符、gzip/json 损坏、概率错位、缺节点。
    返回 (checked, mismatch, examples, n_nodes)。"""
    import gto_lib
    lib = gto_lib.load_lib(lib_path)
    board0 = [c.strip() for c in lib["meta"]["board"].split(",") if c.strip()]
    with open(js_path) as f:
        tree = json.load(f)
    nodes = []
    _collect_gametruth(tree, board0, [], nodes)
    del tree
    rng = random.Random(seed)
    sample = nodes if len(nodes) <= n_sample else rng.sample(nodes, n_sample)
    checked = mism = 0
    examples = []
    for board, path, acts, cp in sample:
        bstr = ",".join(board)
        combos = list(cp)
        pick = combos if len(combos) <= 3 else rng.sample(combos, 3)
        for combo in pick:
            raw = cp[combo]
            if len(raw) != len(acts):
                continue
            got = gto_lib.query(lib, bstr, path, combo)
            checked += 1
            if got is None:
                mism += 1
                if len(examples) < 6:
                    examples.append(("缺节点", bstr, ">".join(path), combo))
                continue
            exp = {a: round(float(p), 5) for a, p in zip(acts, raw)}
            if any(abs(exp[a] - got.get(a, -9.0)) > tol for a in exp):
                mism += 1
                if len(examples) < 6:
                    examples.append(("不一致", bstr, ">".join(path), combo, exp, got))
    return checked, mism, examples, len(nodes)


def parse_one(js_path, meta, fout):
    sz = os.path.getsize(js_path) / 1e9
    if sz > 0.8:
        print(f"  [警告] {os.path.basename(js_path)} 有 {sz:.1f}GB,json.load 可能撑爆内存;"
              f"建议重解时用 --dump-rounds 2(去掉河层)。")
    with open(js_path) as f:
        tree = json.load(f)
    board = [c.strip() for c in meta["board"].split(",") if c.strip()]
    start_street = max(len(board) - 2, 1)          # 3张翻牌=1 / 4张转牌=2 / 5张河牌=3(河牌单街解正确标街)
    ctx = {"id": meta["id"], "depth_bb": meta["depth_bb"], "eff": meta["eff"],
           "board": board, "street": start_street,
           "pot": float(meta["pot"]), "contrib": [0.0, 0.0]}
    cnt = [0]
    prefix = [list(x) for x in meta.get("prefix", [])]   # 河牌解:拼上翻+转动作历史(单街解才有)
    walk(tree, ctx, prefix, fout, cnt)
    del tree
    return cnt[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", help="solver_gen 生成的 manifest.json(批量)")
    ap.add_argument("--single", help="单个 solver 输出 json(验证用)")
    ap.add_argument("--board", help="single 模式:翻牌 如 Qs,Jh,2h")
    ap.add_argument("--pot", type=float, help="single 模式:翻牌起始底池")
    ap.add_argument("--eff", type=float, help="single 模式:有效筹码")
    ap.add_argument("--depth", type=float, help="single 模式:深度bb")
    ap.add_argument("--id", default="single")
    ap.add_argument("--out", default=None, help="BC 行输出 jsonl(非导库模式必填)")
    ap.add_argument("--only", default=None, help="只解析 manifest 里这个 id(逐场景流水线用)")
    ap.add_argument("--append", action="store_true", help="追加写 --out(不覆盖)")
    ap.add_argument("--delete-json", action="store_true", help="解析完删掉该 solver 输出 json(省磁盘)")
    ap.add_argument("--sample", type=float, default=1.0,
                    help="行抽样保留概率(0~1;1=全留)。184翻牌全量建议 ~0.03 控总量")
    ap.add_argument("--dump-lib", default=None,
                    help="导库模式:不产 BC 行,而是把每块板的【全量策略】导成 per-board 紧凑库到该目录(供 GTO 顶档对手查表)")
    ap.add_argument("--verify", action="store_true",
                    help="导库后立即校验库能复现原始 JSON(抽样对比服务层查询);不过则保留 JSON、不删、--only 时非0退出")
    ap.add_argument("--verify-n", type=int, default=300, help="校验抽样节点数")
    args = ap.parse_args()

    global _SAMPLE
    _SAMPLE = max(min(args.sample, 1.0), 1e-6)
    if not args.dump_lib and not args.out:
        ap.error("非导库模式需 --out")

    jobs = []
    if args.manifest:
        man = json.load(open(args.manifest))
        for s in man:
            if args.only and s["id"] != args.only:
                continue
            if os.path.exists(s["out_file"]):
                jobs.append((s["out_file"], s))
            elif not args.only:
                print(f"  [跳过] 未找到输出 {s['out_file']}")
    elif args.single:
        jobs.append((args.single, {"id": args.id, "depth_bb": args.depth or args.eff,
                                    "eff": args.eff, "board": args.board, "pot": args.pot}))
    else:
        ap.error("需 --manifest 或 --single")

    total = 0
    t_start = time.time()

    # ==== 导库模式:每块板导一份全量策略库,不产 BC 行 ====
    if args.dump_lib:
        if not args.only:
            print(f"[{time.strftime('%H:%M:%S')}] 导库 {len(jobs)} 块板 → {args.dump_lib}")
        for i, (js, meta) in enumerate(jobs, 1):
            t0 = time.time()
            try:
                nnodes, path = dump_lib_one(js, meta, args.dump_lib)
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}]   [错误] {js}: {e}")
                if args.only:
                    sys.exit(1)
                continue
            total += nnodes
            if args.verify:                                   # 关②:校验过才允许删 JSON
                chk, mism, ex, ntot = verify_lib(js, path, n_sample=args.verify_n)
                if mism > 0:
                    print(f"[{time.strftime('%H:%M:%S')}]   [校验失败] {meta['id']}: "
                          f"{mism}/{chk} 抽样对不上(共{ntot}节点)→ 保留JSON、不删")
                    for e in ex:
                        print("        ", e)
                    if args.only:
                        sys.exit(1)                            # 逐场景:非0退出 → shell && 不标完成、JSON留存
                    continue
                if args.only:
                    print(f"[校验✓] {meta['id']}: {chk} 抽样节点全对({ntot}节点)")
            if args.delete_json:
                try:
                    os.remove(js)
                except OSError:
                    pass
            if not args.only:
                print(f"[{time.strftime('%H:%M:%S')}]   [{i}/{len(jobs)}] {meta['id']}: "
                      f"{nnodes} 节点  用时 {time.time()-t0:.1f}s → {os.path.basename(path)}")
        if not args.only:
            print(f"[{time.strftime('%H:%M:%S')}] [导库完成] 共 {total} 节点 → {args.dump_lib}")
        return

    mode = "a" if args.append else "w"
    if not args.only:
        print(f"[{time.strftime('%H:%M:%S')}] 开始解析 {len(jobs)} 个场景 (抽样={_SAMPLE})")
    with open(args.out, mode) as fo:
        for i, (js, meta) in enumerate(jobs, 1):
            t0 = time.time()
            try:
                n = parse_one(js, meta, fo)
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}]   [错误] {js}: {e}")
                continue
            total += n
            if args.delete_json:
                try:
                    os.remove(js)
                except OSError:
                    pass
            if not args.only:
                print(f"[{time.strftime('%H:%M:%S')}]   [{i}/{len(jobs)}] {meta['id']}: "
                      f"{n} 条  用时 {time.time()-t0:.1f}s")
    if not args.only:
        print(f"[{time.strftime('%H:%M:%S')}] [解析完成] 共 {total} 条 → {args.out}  "
              f"总用时 {(time.time()-t_start)/60:.1f}分")


if __name__ == "__main__":
    main()
