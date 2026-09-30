"""GTO 反馈引擎(HU 产品核心原语)。

输入一个 solver 解算输出 JSON(某局面的【稠密】解:全部手牌、全部节点),
按【动作历史 + 发牌】导航到玩家的决策点,取出该手牌的 GTO 混合频率,给玩家选的动作评分。

用法(先用 TexasSolver 稠密解一个板:solver_gen 不抽样、solver_parse 不需要——直接读 JSON):
  # 根节点(翻牌先手第一手决策),问 AsKs 的 GTO 频率,并给"CHECK"打分
  python3 gto_feedback.py --json outputs/srp_deep_b0.json --hand AsKs --action CHECK
  # 走一段历史再问:先手过牌、后手下注50%、发转牌 Ac、再到某决策点
  python3 gto_feedback.py --json spot.json --path "CHECK,BET 50.0,CALL,Ac" --hand AsKs --action "BET 66.0"

--path:逗号分隔的【导航令牌】,按顺序消费:
   · 在 action_node 上 = 动作标签(childrens 的键,如 "CHECK"/"BET 50.0"/"CALL"/"FOLD")
   · 在 chance_node 上 = 发出的牌(如 "Ac",转牌/河牌)
--hand:玩家两张(solver 组合式,如 "AsKs")。--action:玩家实际选的动作标签。

产出:该点 GTO 各动作频率 + 玩家动作的频率 + 判读(GTO主线/低频/几乎不做=失误)。
注:本版是【频率反馈】(TexasSolver v0.2.0 不导 EV);EV 损失是后续升级。
"""
import argparse
import json
import sys


def find_children(node):
    for k in ("childrens", "dealcards", "deal_cards", "children"):
        if isinstance(node.get(k), dict):
            return node[k]
    return {}


def _norm(label):
    """归一化动作/发牌标签,容错数值格式与大小写:'BET 3'/'BET 3.0'/'BET 3.000000' → 'BET 3.0';'Ac'→'AC'。"""
    parts = str(label).strip().split()
    kind = parts[0].upper()
    if len(parts) > 1:
        try:
            return f"{kind} {float(parts[1]):.1f}"
        except ValueError:
            return kind
    return kind


def navigate(tree, tokens):
    """按令牌走到目标节点。返回 (node, 错误信息or None)。令牌数值格式容错。"""
    node = tree
    for i, tok in enumerate(tokens):
        nt = node.get("node_type", "")
        children = find_children(node)
        if not children:
            return None, f"第{i+1}个令牌 '{tok}' 处已是终止节点,无法继续"
        nmap = {_norm(k): k for k in children}
        key = nmap.get(_norm(tok))
        if key is None:
            return None, (f"第{i+1}个令牌 '{tok}' 不在可选项里。"
                          f"{'该处是发牌节点,令牌应是牌' if nt.startswith('chance') else '该处是决策节点,令牌应是动作'}。"
                          f"可选:{list(children.keys())[:12]}")
        node = children[key]
    return node, None


def node_strategy(node):
    """action_node → (actions列表, {combo:[freqs]})。"""
    strat = node.get("strategy", {}) or {}
    return strat.get("actions") or node.get("actions") or [], strat.get("strategy", {})


def feedback(node, hand, action):
    """返回 dict:该手牌的 GTO 频率、玩家动作频率、判读。"""
    actions, combo_strat = node_strategy(node)
    if not actions:
        return {"error": "该节点不是决策节点(无策略),检查 --path"}
    # 组合大小写/顺序容错:solver 键形如 'AsKs';尝试原样与反序
    if len(hand) == 4:                       # 大小写容错:点数大写、花色小写
        hand = hand[0].upper() + hand[1].lower() + hand[2].upper() + hand[3].lower()
    probs = combo_strat.get(hand)
    if probs is None and len(hand) == 4:
        probs = combo_strat.get(hand[2:] + hand[:2])
    if probs is None:
        return {"error": f"该手牌 {hand} 不在此节点范围内(或稠密度不足)。"
                         f"范围内样例:{list(combo_strat.keys())[:8]}"}
    gto = {actions[i]: round(float(probs[i]), 4) for i in range(len(actions))}
    norm_gto = {_norm(a): f for a, f in gto.items()}   # 数值格式容错匹配玩家动作
    player_freq = norm_gto.get(_norm(action))
    verdict = None
    if player_freq is None:
        verdict = f"动作 '{action}' 不在此点可选里。可选:{actions}"
    else:
        top = max(gto.values())
        if player_freq >= 0.35:
            verdict = "✅ GTO 主线(高频)"
        elif player_freq >= 0.10:
            verdict = "🟡 GTO 会做,但非主线(低频/混合)"
        elif player_freq > 0.005:
            verdict = "🟠 极低频 —— 大多数时候不该这么打"
        else:
            verdict = f"❌ 失误 —— GTO 几乎不这么打(该点主线是 {max(gto, key=gto.get)} {top:.0%})"
    return {"hand": hand, "gto": gto, "player_action": action,
            "player_freq": player_freq, "verdict": verdict}


def _selftest():
    # 合成一个小树:根 player1 CHECK/BET 50.0;BET 后 player0 FOLD/CALL
    tree = {
        "node_type": "action_node", "player": 1, "actions": ["CHECK", "BET 50.0"],
        "strategy": {"actions": ["CHECK", "BET 50.0"],
                     "strategy": {"AsKs": [0.7, 0.3], "7c2d": [0.98, 0.02]}},
        "childrens": {
            "CHECK": {"node_type": "action_node", "player": 0, "actions": ["CHECK", "BET 50.0"],
                      "strategy": {"actions": ["CHECK", "BET 50.0"],
                                   "strategy": {"AsKs": [0.4, 0.6]}}, "childrens": {}},
            "BET 50.0": {"node_type": "action_node", "player": 0, "actions": ["FOLD", "CALL"],
                         "strategy": {"actions": ["FOLD", "CALL"],
                                      "strategy": {"AsKs": [0.1, 0.9], "7c2d": [0.85, 0.15]}},
                         "childrens": {}}},
    }
    # 根:AsKs 选 CHECK(0.7)= 主线
    r = feedback(tree, "AsKs", "CHECK")
    assert r["verdict"].startswith("✅"), r
    # 根:7c2d 选 BET(0.02)= 失误/极低频
    r2 = feedback(tree, "7c2d", "BET 50.0")
    assert "❌" in r2["verdict"] or "🟠" in r2["verdict"], r2
    # 导航到 BET 后,AsKs 选 FOLD(0.1)= 低频
    node, err = navigate(tree, ["BET 50.0"])
    assert err is None, err
    r3 = feedback(node, "AsKs", "FOLD")
    assert "🟡" in r3["verdict"] or "🟠" in r3["verdict"], r3
    print("[selftest] OK — 导航 + 频率查询 + 判读 都正常")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="solver 解算输出 JSON(稠密:全手牌全节点)")
    ap.add_argument("--path", default="", help="逗号分隔的导航令牌(动作标签/发牌),空=根节点")
    ap.add_argument("--hand", help="玩家两张,如 AsKs")
    ap.add_argument("--action", help="玩家实际选的动作标签,如 CHECK / 'BET 50.0'")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest(); return
    if not (args.json and args.hand and args.action):
        ap.error("需 --json --hand --action(或 --selftest)")

    with open(args.json) as f:
        tree = json.load(f)
    tokens = [t.strip() for t in args.path.split(",") if t.strip()]
    node, err = navigate(tree, tokens)
    if err:
        print("[导航错误]", err); sys.exit(1)
    r = feedback(node, args.hand, args.action)
    if "error" in r:
        print("[错误]", r["error"]); sys.exit(1)
    print(f"=== 局面:玩家={args.hand}  历史={tokens or '(翻牌起手)'} ===")
    print("  该点 GTO 频率:")
    for a, f in sorted(r["gto"].items(), key=lambda kv: -kv[1]):
        bar = "█" * int(f * 30)
        print(f"    {a:12s} {f*100:5.1f}%  {bar}")
    print(f"  玩家选了:{args.action}  → GTO 频率 {(r['player_freq'] or 0)*100:.1f}%")
    print(f"  判读:{r['verdict']}")


if __name__ == "__main__":
    main()
