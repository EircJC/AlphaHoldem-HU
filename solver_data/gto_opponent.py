"""GTO 顶档对手:翻前查表 + 翻/转查库 + 河牌现解 + 兜底,拼成一个 policy(env)→动作档。

翻/转的核心 = 动作线导航:
  库节点 key = 'board|solver标签线'。而牌局历史是 (player, 动作档)。要在库里走到当前节点,
  必须把每个历史动作【映射成 solver 标签】拼成 label 线。映射用 solver_parse.label_to_bucket 的
  【同一档位口径】:每个候选标签算出它的档 → 历史动作档精确匹配,匹配不上按档就近(=下注尺寸就近)。
  库返回的标签同样用 label_to_bucket 映回动作档。任何一步走不到 → 回退兜底并计数。

  python3 gto_opponent.py --selftest        # 合成库端到端自测(多步导航+尺寸映射+采样)
"""
import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import gto_lib                                              # noqa: E402
from solver_parse import label_to_bucket, FOLD, CALL, ALLIN  # noqa: E402


_LAST_NAV = None          # 最近一次导航失败详情(board/path/history),供覆盖率报告抽样打印
# 每块库在内存约 200-300MB(5016节点×~420combo),必须限缓存块数,否则随机翻牌下缓存无限堆积→OOM。
_LIB_CACHE_MAX = int(os.environ.get("GTO_LIB_CACHE", "24"))   # 默认 24 块 ≈ 6GB;24GB 机安全


def _load_cached(cache, path):
    """LRU:命中则移到末尾;未命中则加载,超上限淘汰最旧。cache 需为 OrderedDict。"""
    lib = cache.get(path)
    if lib is not None:
        cache.move_to_end(path)
        return lib
    lib = gto_lib.load_lib(path)
    cache[path] = lib
    if len(cache) > _LIB_CACHE_MAX:
        cache.popitem(last=False)          # 丢最久未用的
    return lib


def _label_bucket(label, pot):
    """solver 标签 → play_vs_bot 动作档(和 solver_parse 训练侧完全同口径)。"""
    return label_to_bucket(label, pot, 0.0, 0.0)


def _pick_label(node, target_bucket):
    """把 node 的可用 solver 标签映到动作档,找与 target_bucket 相同的;没有则按档就近(尺寸就近)。"""
    pot = node.get("pot", 1.0)
    cand = [(lab, _label_bucket(lab, pot)) for lab in node["actions"]]
    for lab, b in cand:
        if b == target_bucket:
            return lab
    return min(cand, key=lambda lb: abs(lb[1] - target_bucket))[0]


def _combo_strat(node, hole):
    """node.strat 里取 hole(试两种排列)→ 概率列表(与 node.actions 对齐)或 None。"""
    s = node["strat"]
    if hole in s:
        return s[hole]
    rev = hole[2:4] + hole[0:2]                             # 'AhKh' → 'KhAh'
    return s.get(rev)


def _sample(labels, probs, greedy, rng):
    if greedy or rng is None:
        return labels[max(range(len(labels)), key=lambda i: probs[i])]
    tot = sum(probs)
    if tot <= 1e-12:
        return labels[0]
    x = rng.random() * tot
    acc = 0.0
    for lab, p in zip(labels, probs):
        acc += p
        if x <= acc:
            return lab
    return labels[-1]


def _flopturn_nav(get_node, remapped_board, remapped_hole, history, cur_street, legal, greedy, rng):
    """翻/转导航核心(与库存储无关):get_node(board_key, path)→节点 dict 或 None。
    内存库和 SQLite 库都用这一套 → 行为完全一致。返回 (动作档 或 None, 原因)。"""
    global _LAST_NAV
    board = [c.strip() for c in remapped_board.split(",") if c.strip()]
    path = []
    for st in (1, 2):
        board_key = ",".join(board[:2 + st])               # flop=3 / turn=4
        for (player, bucket) in history.get(st, []):
            node = get_node(board_key, path)
            if node is None:
                _LAST_NAV = (f"no_node@{board_key} path=[{'>'.join(path)}] "
                             f"next_bucket={bucket} h1={history.get(1)} h2={history.get(2)}")
                return None, "nav_no_node"
            path.append(_pick_label(node, bucket))
        if st == cur_street:
            break
    node = get_node(",".join(board[:2 + cur_street]), path)
    if node is None:
        _LAST_NAV = (f"no_cur@{','.join(board[:2 + cur_street])} path=[{'>'.join(path)}] "
                     f"h1={history.get(1)} h2={history.get(2)}")
        return None, "nav_no_cur"
    probs = _combo_strat(node, remapped_hole)
    if probs is None:
        return None, "hole_not_in_range"
    lab = _sample(node["actions"], probs, greedy, rng)
    bucket = _label_bucket(lab, node.get("pot", 1.0))
    if legal is not None and bucket < len(legal) and not legal[bucket]:
        legals = [i for i in range(len(legal)) if legal[i]]
        if not legals:
            return None, "no_legal"
        bucket = min(legals, key=lambda i: abs(i - bucket))
    return bucket, "ok"


def flopturn_action(lib, remapped_board, remapped_hole, history, cur_street, legal, greedy, rng):
    """翻/转:重放历史走到当前节点 → 采样(内存库版)。见 _flopturn_nav。
    remapped_board=重映射后整板 'Qs,Jh,2h,Th';history={1:[(player,bucket)],2:[...]};cur_street 1或2。"""
    return _flopturn_nav(lambda bk, p: gto_lib.get_node(lib, bk, p),
                         remapped_board, remapped_hole, history, cur_street, legal, greedy, rng)


def flopturn_bucket(lib_index, env, seat, greedy=False, rng=None, lib_cache=None):
    """查 GTO 库,给 seat 在翻/转的动作档。返回 (bucket 或 None, reason)。
    可复用于:GTO 对手(seat0)、模型自己(seat1,play_vs_bot --flopturn-lib)。
    lib_index 可为 MultiDepthIndex(按有效筹码就近选带)或扁平 dict。lib_cache 跨手复用已加载库。"""
    import board_match
    from engine import card_str
    if lib_cache is None:
        lib_cache = collections.OrderedDict()
    try:
        board = [card_str(c) for c in env.board]
        hole = [card_str(c) for c in env.holes[seat]]
        eff = min(env.start_stacks)
        idx = lib_index.select(eff) if hasattr(lib_index, "select") else lib_index
        m = board_match.match(idx, ",".join(board), ",".join(hole))
        if m is None:
            return None, "board_not_covered"
        _solved, rboard, rhole = m
        path = idx[board_match.canon_sig(board[:3])][1]
        lib = _load_cached(lib_cache, path)          # LRU 限量,防 OOM
        history = {1: list(env.history[1]), 2: list(env.history[2])}
        return flopturn_action(lib, rboard, rhole, history, env.street,
                               list(env.legal_mask()), greedy, rng)
    except Exception as e:                                  # 任何异常都回退,别让单手崩掉整场
        return None, f"exc:{type(e).__name__}"


def make_gto_policy(seat, lib_index, preflop_fn, river_fn, fallback_pol, greedy=False, rng=None):
    """返回 pol(env)→动作档。preflop_fn/river_fn=复用 play_vs_bot 里已测的翻前查表/河牌现解(返回档或None);
    fallback_pol=兜底策略(通常神经网络)。pol.counters 记录各阶段命中/回退,供覆盖率统计。"""
    import board_match
    from engine import card_str
    lib_cache = collections.OrderedDict()          # LRU(_load_cached 限量,防 OOM)
    counters = {"preflop": [0, 0], "flopturn": [0, 0], "river": [0, 0], "reasons": {}}

    def pol(env):
        street = env.street
        s = env.to_act
        if street == 0:
            b = preflop_fn(env) if preflop_fn else None
            counters["preflop"][0 if b is not None else 1] += 1
            return b if b is not None else fallback_pol(env)
        if street in (1, 2):
            b, reason = flopturn_bucket(lib_index, env, s, greedy, rng, lib_cache)
            counters["reasons"][reason] = counters["reasons"].get(reason, 0) + 1
            if reason in ("nav_no_node", "nav_no_cur") and _LAST_NAV is not None:
                dbg = counters.setdefault("nav_dbg", [])
                if len(dbg) < 15:
                    dbg.append(f"{reason}: {_LAST_NAV}")
            counters["flopturn"][0 if b is not None else 1] += 1
            return b if b is not None else fallback_pol(env)
        # 河牌
        b = river_fn(env) if river_fn else None
        counters["river"][0 if b is not None else 1] += 1
        return b if b is not None else fallback_pol(env)

    pol.counters = counters
    return pol


def _selftest():
    # 合成一份库:翻牌 Qs,Jh,2h。OOP(1)过/下3 → IP(0)过/下3 → OOP(1)面对下注 弃/跟/加12。
    b0 = label_to_bucket("BET 3", 5.0, 0, 0)               # IP 下注 3 的动作档
    lib = {"meta": {"flop": "Qs,Jh,2h"}, "nodes": {
        "Qs,Jh,2h|": {"player": 1, "pot": 5.0, "actions": ["CHECK", "BET 3"],
                      "strat": {"AhKh": [0.7, 0.3]}},
        "Qs,Jh,2h|CHECK": {"player": 0, "pot": 5.0, "actions": ["CHECK", "BET 3"],
                           "strat": {"AhKh": [0.4, 0.6]}},
        "Qs,Jh,2h|CHECK>BET 3": {"player": 1, "pot": 8.0, "actions": ["FOLD", "CALL", "RAISE 12"],
                                 "strat": {"AhKh": [0.1, 0.5, 0.4]}},
    }}
    # 情形:OOP 面对 IP 翻牌下注(历史=OOP过、IP下3),该 OOP 决策。cur_street=1。
    hist = {1: [(1, CALL), (0, b0)]}
    # 贪心 → 应选概率最高的 CALL(0.5)→ 动作档 CALL
    b, r = flopturn_action(lib, "Qs,Jh,2h", "AhKh", hist, 1, None, True, None)
    print(f"  面对下注 贪心 → 档={b} 原因={r}(应 CALL={CALL})")
    assert r == "ok" and b == CALL, (b, r)
    # 根节点(无历史)贪心 → CHECK(0.7)→ 档 CALL
    b2, r2 = flopturn_action(lib, "Qs,Jh,2h", "AhKh", {}, 1, None, True, None)
    assert r2 == "ok" and b2 == CALL, (b2, r2)
    # 底牌不在范围 → hole_not_in_range 回退
    b3, r3 = flopturn_action(lib, "Qs,Jh,2h", "2c2d", {}, 1, None, True, None)
    assert b3 is None and r3 == "hole_not_in_range", (b3, r3)
    # 采样模式:多次采样应同时出现 CALL 和 RAISE(0.5/0.4)
    import random
    rng = random.Random(0)
    seen = set()
    for _ in range(200):
        bb, _rr = flopturn_action(lib, "Qs,Jh,2h", "AhKh", hist, 1, None, False, rng)
        seen.add(bb)
    print(f"  采样出现的档: {sorted(seen)}(应含 CALL 和某加注档)")
    assert CALL in seen and len(seen) >= 2, seen
    # 组合反排列也能查到
    b4, r4 = flopturn_action(lib, "Qs,Jh,2h", "KhAh", {}, 1, None, True, None)
    assert r4 == "ok", (b4, r4)
    print("[selftest] OK —— 多步导航/尺寸就近映射/采样/组合反排列/缺失回退 全部正确")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest(); return
    ap.error("需 --selftest")


if __name__ == "__main__":
    main()
