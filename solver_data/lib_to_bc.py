"""GTO 库(.lib.json.gz)→ BC 训练行(solver_postflop.jsonl 同格式)。
用于:原始 solver JSON 已删、但库还在时,从库反推 BC 数据蒸馏新底座(1755 翻牌覆盖)。

正确性依据(见 --selftest):
  - 档位 label_to_bucket 只取 amt/pot_before(不用 to_call);库每节点精确存了 pot →
    反推的 strategy_bucket / seq 档位与 solver_parse 从原始树算的【完全一致,非近似】。
  - seq 沿动作线查前缀节点,直接读其 player/pot/street(库已存),前缀在库里唯一存在(按 board 长度定位)。

  python3 lib_to_bc.py --selftest                                  # 合成树:与 solver_parse.walk 逐行 diff
  python3 lib_to_bc.py --lib-dir ../gto_libdir --out solver_lib_bc.jsonl [--sample 1.0] [--depths short,mid,deep]
"""
import argparse
import glob
import gzip
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from solver_parse import label_to_bucket                      # noqa: E402


def _split_key(key):
    """'b1,b2,b3|L1>L2' → (['b1','b2','b3'], ['L1','L2'])。"""
    bs, _, ls = key.partition("|")
    board = [c for c in bs.split(",") if c]
    line = [x for x in ls.split(">") if x] if ls else []
    return board, line


def _reconstruct_seq(nodes, board_list, line):
    """沿 line 逐个动作重建 seq=[[player,bucket,street]]:每个动作在其【决策节点(前缀)】上取 player/pot/street。
    前缀节点的 board 长度未知 → 依次试 flop3/turn4/river5,库里唯一命中。查不到返回 None。"""
    seq = []
    for j in range(len(line)):
        prefix = line[:j]
        P = None
        for blen in (3, 4, 5):
            if blen > len(board_list):
                break
            key = ",".join(board_list[:blen]) + "|" + ">".join(prefix)
            if key in nodes:
                P = nodes[key]
                break
        if P is None:
            return None
        bkt = label_to_bucket(line[j], P["pot"], 0.0, 0.0)
        seq.append([P["player"], bkt, P["street"]])
    return seq


def lib_to_rows(lib, sample=1.0, rng=None):
    """一份已加载 lib → 逐个 BC 行(dict)。sample<1 时按概率丢弃(控总量)。"""
    rng = rng or random
    meta = lib["meta"]
    nodes = lib["nodes"]
    for key, node in nodes.items():
        actions = node.get("actions") or []
        strat = node.get("strat") or {}
        if not actions or not strat:
            continue
        board_list, line = _split_key(key)
        pot = node["pot"]
        player = node["player"]
        street = node["street"]
        seq = _reconstruct_seq(nodes, board_list, line)
        if seq is None:
            continue
        lab2bkt = {a: label_to_bucket(a, pot, 0.0, 0.0) for a in actions}
        board_s = ",".join(board_list)
        for combo, probs in strat.items():
            if len(probs) != len(actions):
                continue
            if sample < 1.0 and rng.random() > sample:
                continue
            raw = {actions[i]: round(float(probs[i]), 6) for i in range(len(actions))}
            bkt = {}
            for i, p in enumerate(probs):
                b = lab2bkt[actions[i]]
                bkt[b] = bkt.get(b, 0.0) + float(p)
            yield {
                "id": meta.get("id"), "depth_bb": meta.get("depth_bb"), "eff_bb": meta.get("eff"),
                "board": board_s, "street": street, "pot": round(pot, 3), "player": player,
                "seq": seq, "hand": combo,
                "strategy_raw": raw,
                "strategy_bucket": {str(k): round(v, 6) for k, v in bkt.items()},
            }


def _load(path):
    with gzip.open(path, "rt", encoding="utf-8") as g:
        return json.load(g)


def convert_dir(lib_dir, out, sample=1.0, depths=None, seed=0):
    """扫 lib_dir(含子目录)所有 .lib.json.gz → 追加 BC 行到 out。depths=只保留这些深度带名。"""
    rng = random.Random(seed)
    from board_match import _NAME_BB
    paths = sorted(glob.glob(os.path.join(lib_dir, "**", "*.lib.json.gz"), recursive=True))
    nfile = nrow = 0
    with open(out, "w", encoding="utf-8") as fo:
        for p in paths:
            if depths:
                band = os.path.basename(os.path.dirname(p)).lower()
                if band in _NAME_BB and band not in depths:
                    continue
            lib = _load(p)
            for row in lib_to_rows(lib, sample, rng):
                fo.write(json.dumps(row, ensure_ascii=False) + "\n")
                nrow += 1
            nfile += 1
    print(f"[lib→bc] {nfile} 份库 → {nrow} 行 → {out}")
    return nrow


def _selftest():
    import tempfile
    import solver_parse as sp
    sp._SAMPLE = 1.0
    # 合成 solver 树:翻牌 OOP(1) CHECK/BET3 → CHECK 后发转牌 Td → IP(0) CHECK/BET5(有翻+转,测街道)
    tree = {
        "node_type": "action_node", "player": 1,
        "strategy": {"actions": ["CHECK", "BET 3"],
                     "strategy": {"AhKh": [0.6, 0.4], "7c2d": [0.9, 0.1]}},
        "childrens": {
            "CHECK": {"node_type": "chance_node", "dealcards": {"Td": {
                "node_type": "action_node", "player": 0,
                "strategy": {"actions": ["CHECK", "BET 5"],
                             "strategy": {"AhKh": [0.5, 0.5], "7c2d": [0.7, 0.3]}},
                "childrens": {"CHECK": {"node_type": "showdown_node"},
                              "BET 5": {"node_type": "terminal_node"}}}}},
            "BET 3": {"node_type": "terminal_node"}}}
    d = tempfile.mkdtemp()
    js = os.path.join(d, "t.json"); json.dump(tree, open(js, "w"))
    meta = {"id": "srp_mid_b0", "board": "Qs,Jh,2h", "pot": 5, "eff": 127.5, "depth_bb": 130}
    # 参照:solver_parse 从原始树产 BC 行
    ref_path = os.path.join(d, "ref.jsonl")
    with open(ref_path, "w") as fo:
        sp.parse_one(js, meta, fo)
    ref = [json.loads(l) for l in open(ref_path)]
    # 我方:dump_lib_one → 库 → lib_to_rows
    sp.dump_lib_one(js, meta, d)
    lib = _load(os.path.join(d, "srp_mid_b0.lib.json.gz"))
    mine = list(lib_to_rows(lib, 1.0))
    # 逐行 diff:按 (board, seq, hand) 建键,比 strategy_bucket / player / street / pot
    def keyed(rows):
        m = {}
        for r in rows:
            k = (r["board"], json.dumps(r["seq"]), r["hand"])
            m[k] = r
        return m
    R, M = keyed(ref), keyed(mine)
    print(f"  参照行={len(ref)} 我方行={len(mine)}  键集合相同={set(R)==set(M)}")
    assert set(R) == set(M), ("键不一致", set(R) ^ set(M))
    bad = 0
    for k in R:
        for f in ("player", "street", "strategy_bucket"):
            if R[k][f] != M[k][f]:
                bad += 1
                if bad <= 5:
                    print(f"    差异 {k} 字段{f}: 参照{R[k][f]} vs 我方{M[k][f]}")
        if abs(float(R[k]["pot"]) - float(M[k]["pot"])) > 1e-6:
            bad += 1
    assert bad == 0, f"{bad} 处字段不一致"
    print("[selftest] OK —— 库反推 BC 与 solver_parse 原始树逐行完全一致(seq/strategy_bucket/player/street/pot)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib-dir", help="库根目录(含 short/mid/deep 子目录或扁平)")
    ap.add_argument("--out", default="solver_lib_bc.jsonl")
    ap.add_argument("--sample", type=float, default=1.0, help="行抽样保留比例(库全量可能很大)")
    ap.add_argument("--depths", default="", help="只转这些深度带(逗号,如 short,mid);空=全部")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest(); return
    if not args.lib_dir:
        ap.error("需 --lib-dir(或 --selftest)")
    depths = {x.strip() for x in args.depths.split(",") if x.strip()} or None
    convert_dir(args.lib_dir, args.out, args.sample, depths)


if __name__ == "__main__":
    main()
