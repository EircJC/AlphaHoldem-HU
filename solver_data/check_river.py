"""河牌数据体检:跑完 river 解析(river_postflop.jsonl 或 1-flop 小测)后,自动核对几处易错点。
用法: python3 check_river.py river_postflop.jsonl        (或 /tmp/rtest.jsonl)
无报错 + 全绿 = 河牌数据的街标注/动作历史/筹码特征/软目标都对。
"""
import json
import sys

POT0 = 5.0                                            # srp 翻牌起始底池


def main(path):
    n = bad = 0
    no_hist = neg_stack = bad_soft = bad_street = bad_board = 0
    street_hist = {1: 0, 2: 0, 3: 0}
    sample = None
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        n += 1
        # 1) 街=3
        if int(r.get("street", -1)) != 3:
            bad_street += 1
        # 2) board 5 张
        board = [c for c in r.get("board", "").split(",") if c]
        if len(board) != 5:
            bad_board += 1
        # 3) seq 街道:含翻(1)/转(2)历史?街道非递减?
        seq = r.get("seq", [])
        sts = [int(s[2]) for s in seq]
        for st in sts:
            if st in street_hist:
                street_hist[st] += 1
        has_pre = any(st in (1, 2) for st in sts)
        if not has_pre:
            no_hist += 1
        if sts != sorted(sts):
            bad += 1
        # 4) 软目标和≈1
        soft = sum(float(v) for v in r.get("strategy_bucket", {}).values())
        if not (0.98 <= soft <= 1.02):
            bad_soft += 1
        # 5) my_stack 反推 = eff-(pot-5)/2 应 ∈ (0, eff]
        eff = float(r.get("eff_bb", 0))
        pot = float(r.get("pot", 0))
        ms = eff - max(pot - POT0, 0) / 2.0
        if ms <= 0 or ms > eff + 1e-6:
            neg_stack += 1
        if sample is None:
            sample = (r, ms)
    print(f"总行数: {n}")
    print(f"  街≠3: {bad_street}   board≠5张: {bad_board}   缺翻/转历史: {no_hist}   "
          f"seq街道乱序: {bad}   软目标和≠1: {bad_soft}   my_stack异常: {neg_stack}")
    print(f"  seq 里各街动作数: 翻{street_hist[1]} 转{street_hist[2]} 河{street_hist[3]}")
    if sample:
        r, ms = sample
        print("\n样例第一行:")
        print(f"  street={r['street']} board={r['board']} player={r.get('player')}")
        print(f"  seq={r['seq']}")
        print(f"  eff_bb={r.get('eff_bb')} pot={r.get('pot')} → 反推 my_stack={ms:.1f}")
        print(f"  strategy_bucket={r.get('strategy_bucket')}")
    ok = (bad_street == 0 and bad_board == 0 and no_hist == 0 and bad == 0
          and bad_soft == 0 and neg_stack == 0)
    print("\n" + ("✅ 全部通过 —— 河牌数据街标注/动作历史/筹码/软目标都正确"
                  if ok else "❌ 有异常项(见上),把输出发我排查"))
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python3 check_river.py <river_postflop.jsonl>"); sys.exit(2)
    sys.exit(main(sys.argv[1]))
