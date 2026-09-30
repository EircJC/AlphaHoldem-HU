"""方案A 内核:从 dump-2 的 flop 解里,沿【翻牌线→转牌→转牌线】把双方 reach 乘下来,
得到【进河牌前的到达范围】(自洽,非猜测),供 river_gen 生成便宜的河牌单街解。

原理:节点的 strategy 给每个 combo 各动作的频率。沿一条线走:
  · 到【行动节点】:轮到的一方,每个 combo 的权重 ×= 它选该动作的频率;
  · 到【发牌节点(转/河)】:双方都去掉含该牌的 combo(阻断);
走到线尾,两边 combo→权重 就是到达范围。

⚠️ 依赖 TexasSolver JSON 的两点格式,需用你机器上真解片段确认(见 __main__ 的 --inspect):
   ① 行动节点的 player 值 → OOP/IP 映射(本文件假设 player 0=OOP 先手, 1=IP);
   ② strategy 的 combo 键写法(如 'AhKs')与动作顺序。
确认后 combo 归一(canon)会自动对齐两种写法。
"""
import argparse
import json

from solver_parse import label_to_bucket   # 与翻+转数据同口径的 动作→bucket

RANKS = "23456789TJQKA"
SUITS = "cdhs"
OOP, IP = 1, 0                                    # 【真解确认】root(翻后先手=OOP)=player1;IP=player0


# ---------- 范围字符串 → {canon_combo: weight} ----------
def _cards_of(tok):
    """'AKs'/'AKo'/'AA' → 该手所有具体 combo(('Ah','Ks') 等,牌用字符串)。"""
    r1, r2 = tok[0], tok[1]
    out = []
    if r1 == r2:                                  # 对子:C(4,2)=6
        s = [c for c in SUITS]
        for i in range(4):
            for j in range(i + 1, 4):
                out.append((r1 + s[i], r2 + s[j]))
    elif tok[2:3] == "s":                         # 同花:4
        for c in SUITS:
            out.append((r1 + c, r2 + c))
    else:                                         # 非同花:12
        for a in SUITS:
            for b in SUITS:
                if a != b:
                    out.append((r1 + a, r2 + b))
    return out


def canon(c1, c2):
    """两张牌 → 归一 combo 键(与写法/顺序无关):按字符串排序拼接。"""
    return "".join(sorted([c1, c2]))


def expand_range(range_str):
    """'AA,AKs,A5o:0.5,...' → {canon: weight}。支持 tok 末尾 ':w';支持 '22+'/'A2s+'(向上)。"""
    out = {}
    for raw in range_str.replace(" ", "").split(","):
        if not raw:
            continue
        w = 1.0
        tok = raw
        if ":" in tok:
            tok, ws = tok.split(":", 1)
            try:
                w = float(ws)
            except ValueError:
                w = 1.0
        toks = [tok]
        if "-" in tok:                            # dash 范围:A5s-A2s / JJ-99
            toks = _expand_dash(tok)
        elif tok.endswith("+"):                   # 展开 '+'(向上到 A)
            toks = _expand_plus(tok[:-1])
        for t in toks:
            for a, b in _cards_of(t):
                out[canon(a, b)] = w
    return out


def _expand_plus(tok):
    """'22+'→22..AA;'A2s+'→A2s..AKs(高牌固定, 低牌向上到高牌-1)。"""
    if tok[0] == tok[1]:                          # 对子
        i = RANKS.index(tok[0])
        return [RANKS[j] * 2 for j in range(i, 13)]
    hi, lo, suf = tok[0], tok[1], tok[2:]
    hi_i, lo_i = RANKS.index(hi), RANKS.index(lo)
    return [hi + RANKS[j] + suf for j in range(lo_i, hi_i)]


def _expand_dash(tok):
    """'A5s-A2s'→A5s,A4s,A3s,A2s;'JJ-99'→JJ,TT,99。"""
    a, b = tok.split("-", 1)
    if len(a) == 2 and len(b) == 2 and a[0] == a[1]:      # 对子区间
        lo, hi = sorted([RANKS.index(a[0]), RANKS.index(b[0])])
        return [RANKS[j] * 2 for j in range(lo, hi + 1)]
    suf = a[2:]                                            # 同高牌的 suited/offsuit 区间
    lo, hi = sorted([RANKS.index(a[1]), RANKS.index(b[1])])
    return [a[0] + RANKS[j] + suf for j in range(lo, hi + 1)]


def _combo_class(k):
    """canon combo 键(如 '2c2d'/'AhKs')→ 手牌类 '22'/'AKs'/'AKo'(TexasSolver 只认这个)。"""
    c1, c2 = k[:2], k[2:]
    r1, r2 = RANKS.index(c1[0]), RANKS.index(c2[0])
    if r1 == r2:
        return c1[0] * 2
    hi, lo = (c1, c2) if r1 > r2 else (c2, c1)
    return hi[0] + lo[0] + ("s" if c1[1] == c2[1] else "o")


def combos_to_range_str(reach, board=(), thresh=1e-4):
    """{canon combo: weight} → TexasSolver 范围串(【手牌类】记法,v0.2.0 不认具体 combo)。
    类权重 = 存活 reach 之和 ÷ 该类【未被板牌阻断】的 combo 总数
    (把翻/转弃掉的 combo 当 0 稀释进去,避免 TexasSolver 把它们重新灌回该类)。归一到峰值。"""
    if not reach:
        return ""
    bset = set(board)
    csum = {}
    for k, w in reach.items():
        cls = _combo_class(k)
        csum[cls] = csum.get(cls, 0.0) + w
    cw = {}
    for cls, s in csum.items():
        n_ok = sum(1 for a, b in _cards_of(cls) if a not in bset and b not in bset)
        cw[cls] = s / max(n_ok, 1)                     # ÷未阻断combo数(非仅存活数)
    mx = max(cw.values()) or 1.0
    parts = [f"{c}:{cw[c] / mx:.4f}" for c in cw if cw[c] / mx >= thresh]
    return ",".join(parts)


# ---------- JSON 树导航(镜像 solver_parse,已在真数据验证) ----------
def find_children(node):
    for k in ("childrens", "dealcards", "deal_cards", "children"):
        if isinstance(node.get(k), dict):
            return node[k]
    return {}


def node_strategy(node):
    strat = node.get("strategy", {}) or {}
    return strat.get("actions") or node.get("actions") or [], strat.get("strategy", {})


def _norm(lbl):
    p = str(lbl).strip().split()
    return (p[0].upper() + (f" {float(p[1]):.1f}" if len(p) > 1 else "")) if p else ""


def reach_along(tree, root_ip_str, root_oop_str, line):
    """沿 line 走,返回到达该点的 (ip_reach, oop_reach)。
    line = 令牌列表:行动节点上=动作标签(如 'CHECK'/'BET 50');发牌节点上=牌(如 'Ac')。"""
    _KEYCACHE.clear()
    ip = expand_range(root_ip_str)
    oop = expand_range(root_oop_str)
    node = tree
    for tok in line:
        nt = node.get("node_type", "")
        children = find_children(node)
        if not children:
            raise ValueError(f"令牌 '{tok}' 处已是终止/无子节点")
        if nt.startswith("action") or node.get("player") is not None:
            player = node.get("player")
            actions, combo_strat = node_strategy(node)
            nmap = {_norm(a): a for a in actions}
            key = nmap.get(_norm(tok))
            if key is None:
                raise ValueError(f"动作 '{tok}' 不在 {actions}")
            idx = actions.index(key)
            reach = oop if player == OOP else ip     # 只重加权【行动方】
            for combo in list(reach):
                probs = combo_strat.get(_combo_key(combo, combo_strat))
                reach[combo] *= (probs[idx] if probs and idx < len(probs) else 0.0)
            node = children.get(key) or children.get(tok)
        else:                                        # 发牌节点:tok=牌,双方去阻断
            for reach in (ip, oop):
                for combo in list(reach):
                    if tok in (combo[:2], combo[2:]):
                        del reach[combo]
            node = children.get(tok)
        if node is None:
            raise ValueError(f"找不到子节点 '{tok}'")
    return ip, oop


_KEYCACHE = {}


def _combo_key(canon_combo, combo_strat):
    """把归一 combo 对到该节点 strategy 里真实写的键(两张顺序/写法可能不同)。"""
    # combo_strat 的键 canon 化后建索引,缓存到该 dict id
    cid = id(combo_strat)
    idx = _KEYCACHE.get(cid)
    if idx is None:
        idx = {}
        for k in combo_strat:
            if len(k) >= 4:
                idx[canon(k[:2], k[2:])] = k
        _KEYCACHE[cid] = idx
    return idx.get(canon_combo)


# ---------- 枚举"进河牌"点(转牌行动结束) ----------
def _parse_label(lbl):
    p = str(lbl).strip().split()
    return p[0].upper(), (float(p[1]) if len(p) > 1 else 0.0)


def _apply(label, pot, contrib, player, flop_pot):
    """动作后更新 (pot, contrib)。contrib=本街两人投入;pot=总底池。镜像 solver_parse。"""
    kind, amt = _parse_label(label)
    c = list(contrib)
    opp = 1 - player
    if kind == "CALL":
        add = max(c[opp] - c[player], 0.0); c[player] += add; pot += add
    elif kind in ("BET", "RAISE"):
        add = amt - c[player]; c[player] = amt; pot += max(add, 0.0)
    elif kind == "ALLIN":
        pass
    return pot, c


def enumerate_river_entries(tree, root_ip, root_oop, flop_pot, eff, board3,
                            turn_cap=0, line_cap=0):
    """遍历 dump-2 树,产出所有【进河牌】点:(board4, pot, behind, ip_reach, oop_reach)。
    turn_cap>0: 每个翻牌线最多取这么多张转牌(采样控量);line_cap>0: 总条数上限。"""
    _KEYCACHE.clear()                                 # 防跨调用 id() 复用取到旧索引
    out = []
    ip0, oop0 = expand_range(root_ip), expand_range(root_oop)
    for c in board3:                                  # 翻牌已在桌上 → 去掉用到翻牌的 combo
        ip0 = {k: v for k, v in ip0.items() if c not in (k[:2], k[2:])}
        oop0 = {k: v for k, v in oop0.items() if c not in (k[:2], k[2:])}

    def walk(node, ip, oop, pot, contrib, committed, street, board, line):
        if line_cap and len(out) >= line_cap:
            return
        nt = node.get("node_type", "")
        children = find_children(node)
        if not children:
            return
        if nt.startswith("chance") or (not node_strategy(node)[0] and any(len(k) == 2 for k in children)):
            cards = list(children.items())               # 发牌节点(street1 发转牌)
            if turn_cap:
                cards = cards[:turn_cap]
            for card, child in cards:
                nip = {k: v for k, v in ip.items() if card not in (k[:2], k[2:])}
                noop = {k: v for k, v in oop.items() if card not in (k[:2], k[2:])}
                walk(child, nip, noop, pot, [0.0, 0.0], committed, street + 1, board + [card], line)
            return
        player = node.get("player")                      # 行动节点
        actions, cstrat = node_strategy(node)
        for ai, act in enumerate(actions):
            child = children.get(act)
            if child is None:
                continue
            kind = _parse_label(act)[0]
            to_call = max(contrib[1 - player] - contrib[player], 0.0)
            bkt = label_to_bucket(act, pot, to_call, contrib[player])   # 该动作的 bucket
            nline = line + [[player, bkt, street]]       # 累积翻/转动作历史
            npot, ncontrib = _apply(act, pot, contrib, player, flop_pot)
            ncommitted = list(committed)
            ncommitted[player] += (ncontrib[player] - contrib[player])
            reach = (oop if player == OOP else ip)
            nreach = {}
            for combo, w in reach.items():
                probs = cstrat.get(_combo_key(combo, cstrat))
                p = probs[ai] if probs and ai < len(probs) else 0.0
                if w * p > 1e-9:
                    nreach[combo] = w * p
            nip, noop = (ip, nreach) if player == OOP else (nreach, oop)
            child_kids = find_children(child)
            # 转牌(street2)行动结束的叶子(平跟/双过,非弃牌)= 进河牌点
            if street == 2 and not child_kids and kind in ("CALL", "CHECK"):
                behind = eff - max(ncommitted)
                out.append((list(board), round(npot, 2), round(behind, 2),
                            dict(nip), dict(noop), nline))     # 带上翻+转动作历史
                if line_cap and len(out) >= line_cap:
                    return
            elif child_kids:
                walk(child, nip, noop, npot, ncontrib, ncommitted, street, board, nline)

    walk(tree, ip0, oop0, flop_pot, [0.0, 0.0], [0.0, 0.0], 1, list(board3), [])
    return out


# ---------- 自测(合成小树)----------
def _selftest():
    # 合成:根=OOP(player0)行动 CHECK/BET 50;各分支下 IP(player1)行动;再发一张转牌 Ac
    tree = {
        "node_type": "action_node", "player": 1,                 # root=OOP=player1
        "strategy": {"actions": ["CHECK", "BET 50"],
                     "strategy": {"AsKs": [0.6, 0.4], "7c2d": [0.9, 0.1]}},
        "childrens": {
            "CHECK": {"node_type": "action_node", "player": 0,   # IP=player0
                      "strategy": {"actions": ["CHECK"], "strategy": {"AsKs": [1.0], "7c2d": [1.0]}},
                      "childrens": {"CHECK": {
                          "node_type": "chance_node",
                          "dealcards": {"Ac": {"node_type": "action_node", "player": 1,
                                               "strategy": {"actions": ["CHECK"], "strategy": {"AsKs": [1.0]}},
                                               "childrens": {}}}}}},
            "BET 50": {"node_type": "action_node", "player": 0, "strategy": {"actions": ["FOLD", "CALL"],
                       "strategy": {"AsKs": [0.2, 0.8]}}, "childrens": {}}},
    }
    ip, oop = reach_along(tree, "AKs", "AKs,72o", ["CHECK", "CHECK", "Ac"])
    # OOP: AsKs 权重应 ×0.6(CHECK);IP: AsKs ×1.0;发 Ac 后 含A/含c 的 combo 被删
    ak = canon("As", "Ks")
    assert abs(oop.get(ak, 0) - 0.6) < 1e-9, oop.get(ak)
    assert abs(ip.get(ak, 0) - 1.0) < 1e-9, ip.get(ak)
    assert canon("Ac", "Kc") not in ip                 # 若有含Ac的combo会被删(这里无)
    # 72o 里含 2c 的 combo:'2c7?'/'7?2c' → 发 Ac 不删(不含Ac/含c的看有没有 c 花)
    s = combos_to_range_str(oop)
    assert "AKs" in s                                # 现输出手牌类记法(TexasSolver v0.2.0 口径)
    print("[selftest] OK — 范围展开/reach沿线/阻断/序列化 正常")


def _inspect(js):
    """打印一个真解 JSON 的关键结构,供你确认 player 映射 + combo 键写法。"""
    tree = json.load(open(js))
    print("root node_type:", tree.get("node_type"), " player:", tree.get("player"))
    a, cs = node_strategy(tree)
    print("root actions:", a)
    print("root strategy 键样例(前5):", list(cs.keys())[:5])
    ch = find_children(tree)
    print("root 子节点键(前8):", list(ch.keys())[:8])
    # 找一个 chance(发牌)节点
    def dfs(n, d=0):
        if d > 6:
            return
        if n.get("node_type", "").startswith("chance") or "dealcards" in n:
            print("发牌节点 键样例:", list(find_children(n).keys())[:6]); return
        for c in find_children(n).values():
            if isinstance(c, dict):
                dfs(c, d + 1)
    dfs(tree)


def _parse_input(txt):
    """读 solver_gen 的输入文件 → dict(pot, eff, board[list], ip, oop)。"""
    d = {}
    for line in open(txt):
        s = line.strip()
        if s.startswith("set_pot"):
            d["pot"] = float(s.split()[1])
        elif s.startswith("set_effective_stack"):
            d["eff"] = float(s.split()[1])
        elif s.startswith("set_board"):
            d["board"] = s.split()[1].split(",")
        elif s.startswith("set_range_ip"):
            d["ip"] = s.split(None, 1)[1]
        elif s.startswith("set_range_oop"):
            d["oop"] = s.split(None, 1)[1]
    return d


def _entries(js, txt, turn_cap, line_cap):
    tree = json.load(open(js))
    cfg = _parse_input(txt)
    ent = enumerate_river_entries(tree, cfg["ip"], cfg["oop"], cfg["pot"], cfg["eff"],
                                  cfg["board"], turn_cap=turn_cap, line_cap=line_cap)
    print(f"进河牌点数: {len(ent)}  (翻牌={cfg['board']} pot={cfg['pot']} eff={cfg['eff']})")
    for board4, pot, behind, ip, oop, prefix in ent[:4]:
        print(f"  board4={board4} pot={pot} behind={behind}  "
              f"IP范围{len(ip)}手/OOP范围{len(oop)}手  翻+转历史={prefix}")
        print("    IP:", combos_to_range_str(ip, board4)[:110], "...")
        print("    OOP:", combos_to_range_str(oop, board4)[:110], "...")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--inspect", help="真解 JSON:打印结构供确认格式")
    ap.add_argument("--entries", help="真解 JSON:枚举进河牌点(配合 --input)")
    ap.add_argument("--input", help="对应的 solver_gen 输入文件(读 pot/eff/board/范围)")
    ap.add_argument("--turn-cap", type=int, default=2, help="每翻牌线最多取几张转牌(0=全)")
    ap.add_argument("--line-cap", type=int, default=20, help="总进河牌点上限(0=不限)")
    args = ap.parse_args()
    if args.inspect:
        _inspect(args.inspect)
    elif args.entries:
        _entries(args.entries, args.input, args.turn_cap, args.line_cap)
    else:
        _selftest()
