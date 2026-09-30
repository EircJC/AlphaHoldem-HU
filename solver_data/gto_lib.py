"""GTO 顶档策略库:加载 + 查询(solver_parse --dump-lib 产出的 per-board 紧凑库)。

库结构(每块板一份 <id>.lib.json.gz):
  {"meta":{id,board,flop,pot,eff,depth_bb},
   "nodes":{ "board|动作线": {player, board, street, pot, actions:[标签], strat:{combo:[概率...]}} }}

节点 key = 'board|动作线':board=该点已发的翻/转牌(逗号分隔);动作线='>'连接到该点走过的原始 label。
查询:给 (当前board, 已走label序列, hero的combo) → {动作标签: 概率}。

  python3 gto_lib.py --selftest        # 合成 solver 树端到端自测(建库→查库)
  python3 gto_lib.py --show some.lib.json.gz   # 看一份库的统计 + 根节点样例
"""
import argparse
import gzip
import json
import os


def load_lib(path):
    with gzip.open(path, "rt", encoding="utf-8") as g:
        return json.load(g)


def node_key(board, labels):
    """board=逗号分隔字符串或 list;labels=已走过的原始 label 列表。"""
    if isinstance(board, (list, tuple)):
        board = ",".join(board)
    return board + "|" + ">".join(labels)


def query(lib, board, labels, combo):
    """→ {动作标签: 概率} 或 None(节点/combo 不在库)。combo 如 'AhKh'(solver 组合写法)。"""
    node = lib["nodes"].get(node_key(board, labels))
    if node is None:
        return None
    probs = node["strat"].get(combo)
    if probs is None:
        return None
    return dict(zip(node["actions"], probs))


def get_node(lib, board, labels):
    """→ 整个节点(player/actions/strat/pot…)或 None。供服务层导航用。"""
    return lib["nodes"].get(node_key(board, labels))


def _selftest():
    import tempfile
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    import solver_parse as sp

    # 合成一棵 mini solver 树(仿 TexasSolver 结构):
    #   翻牌 OOP(player1) CHECK/BET 3 → CHECK 后发转牌 Td → IP(player0) CHECK/BET 5
    tree = {
        "node_type": "action_node", "player": 1,
        "strategy": {"actions": ["CHECK", "BET 3"],
                     "strategy": {"AhKh": [0.6, 0.4], "7c2d": [0.95, 0.05]}},
        "childrens": {
            "CHECK": {
                "node_type": "chance_node",
                "dealcards": {
                    "Td": {
                        "node_type": "action_node", "player": 0,
                        "strategy": {"actions": ["CHECK", "BET 5"],
                                     "strategy": {"AhKh": [0.5, 0.5], "7c2d": [0.8, 0.2]}},
                        "childrens": {"CHECK": {"node_type": "showdown_node"},
                                      "BET 5": {"node_type": "terminal_node"}},
                    }
                },
            },
            "BET 3": {"node_type": "terminal_node"},
        },
    }
    d = tempfile.mkdtemp()
    jspath = os.path.join(d, "test.json")
    json.dump(tree, open(jspath, "w"))
    meta = {"id": "srp_test_b0", "board": "Qs,Jh,2h", "pot": 5, "eff": 200, "depth_bb": 200}
    nnodes, libpath = sp.dump_lib_one(jspath, meta, d)
    print(f"[建库] {nnodes} 节点 → {os.path.basename(libpath)}")
    lib = load_lib(libpath)

    # 根节点(翻牌 OOP,动作线空):AhKh 应 CHECK 0.6 / BET 3 0.4
    r = query(lib, "Qs,Jh,2h", [], "AhKh")
    print("  根 AhKh:", r)
    assert r == {"CHECK": 0.6, "BET 3": 0.4}, r
    assert query(lib, "Qs,Jh,2h", [], "7c2d") == {"CHECK": 0.95, "BET 3": 0.05}
    # 转牌节点(走过 CHECK、发 Td,IP 决策):board 变、动作线=CHECK
    t = query(lib, "Qs,Jh,2h,Td", ["CHECK"], "AhKh")
    print("  转 CHECK 后 AhKh:", t)
    assert t == {"CHECK": 0.5, "BET 5": 0.5}, t
    # 节点元信息:根 player=1、转 player=0、街号递增
    assert get_node(lib, "Qs,Jh,2h", [])["player"] == 1
    tn = get_node(lib, "Qs,Jh,2h,Td", ["CHECK"])
    assert tn["player"] == 0 and tn["street"] == 2, tn
    # 不存在的 combo / 节点 → None
    assert query(lib, "Qs,Jh,2h", [], "9d9c") is None
    assert query(lib, "Qs,Jh,2h", ["BET 3"], "AhKh") is None    # BET 3 是终局,没节点
    print("[selftest] OK —— 建库/按 board+动作线+combo 查询/节点元信息/缺失返回None 全部正确")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", help="查看一份 .lib.json.gz 的统计 + 根节点样例")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest(); return
    if args.show:
        lib = load_lib(args.show)
        nodes = lib["nodes"]
        streets = {}
        for k, v in nodes.items():
            streets[v["street"]] = streets.get(v["street"], 0) + 1
        print(f"板={lib['meta'].get('flop')}  节点数={len(nodes)}  各街节点: {streets}")
        # 根节点(flop、动作线空)
        flop = lib["meta"]["flop"]
        root = nodes.get(node_key(flop, []))
        if root:
            some = list(root["strat"])[:3]
            print(f"根节点 player={root['player']} actions={root['actions']}")
            for c in some:
                print(f"  {c}: {dict(zip(root['actions'], root['strat'][c]))}")
        return
    ap.error("需 --selftest 或 --show")


if __name__ == "__main__":
    main()
