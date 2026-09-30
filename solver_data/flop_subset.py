"""纹理均衡的翻牌代表子集生成器(替代 Pio/GTOWizard 的专有 flop 子集,免授权、可控数量)。

原理:
  1. 枚举全部 C(52,3)=22100 个翻牌;
  2. 按【花色同构】归并成 1755 个规范翻牌,每个记录它代表多少个具体翻牌(=出现频率权重);
  3. 按纹理(成对情况 × 花色形态)分层,层内按频率加权 + 沿高牌均匀铺开,采样 N 个。
  → turn/river 不用采样(solver 解每个翻牌时已枚举其所有转河发牌)。

用法:
  python3 flop_subset.py --n 25                 # 打印 25 个代表翻牌 + 纹理分布
  python3 flop_subset.py --n 49 --out flops.txt # 存文件(每行一个,如 Qs,Jh,2h)
被 solver_gen.py 通过 --flop-subset N 调用(见 subset())。
"""
import argparse
from itertools import combinations, permutations

RANKC = "23456789TJQKA"     # idx0=2 .. idx12=A
SUITC = "shdc"


def _canon_sig(cards):
    """3 张牌(rank,suit)→ 花色同构规范签名(对4!花色置换取字典序最小)。"""
    best = None
    for perm in permutations(range(4)):
        mapped = sorted(((r, perm[s]) for r, s in cards), key=lambda x: (-x[0], x[1]))
        # 花色按出现顺序重标为 0,1,2...(保证与具体标号无关)
        relabel = {}
        norm = []
        for r, s in mapped:
            if s not in relabel:
                relabel[s] = len(relabel)
            norm.append((r, relabel[s]))
        norm = tuple(norm)
        if best is None or norm < best:
            best = norm
    return best


def canonical_flops():
    """→ {sig: [rep_cards(tuple (rank,suit0..)), weight]}。共 1755 个。"""
    groups = {}
    deck = [(r, s) for r in range(13) for s in range(4)]
    for c in combinations(deck, 3):
        sig = _canon_sig(c)
        if sig not in groups:
            groups[sig] = [sig, 0]      # sig 本身就是规范代表(花色已标 0,1,2)
        groups[sig][1] += 1
    return groups


def _board_str(cards):
    """规范 cards [(rank,suit_idx)] 降序 → 'Qs,Jh,2h'(suit_idx→shdc)。"""
    cs = sorted(cards, key=lambda x: (-x[0], x[1]))
    return ",".join(RANKC[r] + SUITC[s] for r, s in cs)


def _texture(cards):
    ranks = sorted((r for r, _ in cards), reverse=True)
    suits = [s for _, s in cards]
    nd = len(set(suits))
    suit_cls = {1: "mono", 2: "twotone", 3: "rainbow"}[nd]
    if ranks[0] == ranks[1] == ranks[2]:
        pair_cls = "trips"
    elif ranks[0] == ranks[1] or ranks[1] == ranks[2]:
        pair_cls = "paired"
    else:
        pair_cls = "unpaired"
    top = ranks[0]
    hi = "A" if top == 12 else "K" if top == 11 else "Q" if top == 10 else \
         "J" if top == 9 else "T" if top == 8 else "mid" if top >= 5 else "low"
    return {"pair": pair_cls, "suit": suit_cls, "hi": hi,
            "ranks": tuple(ranks), "key": (pair_cls, suit_cls)}


def subset(n, seed=0):
    """返回 n 个纹理均衡的翻牌 board 字符串列表(确定性,seed 仅影响层内取整微调)。"""
    groups = canonical_flops()
    flops = []
    for sig, (rep, w) in groups.items():
        tex = _texture(rep)
        flops.append({"board": _board_str(rep), "w": w, **tex})
    total_w = sum(f["w"] for f in flops)

    # 按 (pair,suit) 分层
    strata = {}
    for f in flops:
        strata.setdefault(f["key"], []).append(f)
    wsum = {k: sum(x["w"] for x in v) for k, v in strata.items()}
    keys = sorted(strata, key=lambda k: -wsum[k])

    # 层配额:大余数法(Hamilton),精确和为 n,允许小层为 0(N 很小时),不会死循环
    raw = {k: n * wsum[k] / total_w for k in strata}
    alloc = {k: int(raw[k]) for k in strata}
    rem = n - sum(alloc.values())
    for k in sorted(strata, key=lambda k: -(raw[k] - int(raw[k])))[:max(rem, 0)]:
        alloc[k] += 1

    # 层内:按高牌降序,均匀取样铺开(覆盖高/中/低)
    picked = []
    for k in keys:
        pool = sorted(strata[k], key=lambda x: (-x["ranks"][0], -x["ranks"][1], -x["ranks"][2]))
        m = min(alloc[k], len(pool))
        if m <= 0:
            continue
        if m >= len(pool):
            picked += pool
        else:
            idx = sorted({round(i * (len(pool) - 1) / (m - 1)) if m > 1 else 0 for i in range(m)})
            picked += [pool[i] for i in idx]
    # dedup 取整可能少几个 → 从剩余(按频率)补足
    if len(picked) < n:
        have = {p["board"] for p in picked}
        for f in sorted(flops, key=lambda x: -x["w"]):
            if f["board"] not in have:
                picked.append(f); have.add(f["board"])
                if len(picked) >= n:
                    break
    picked = picked[:n]
    return [p["board"] for p in picked], picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    boards, meta = subset(args.n)
    import collections
    print(f"=== {len(boards)} 个代表翻牌 ===")
    for b in boards:
        print(" ", b)
    print("=== 纹理分布 ===")
    print("  成对:", dict(collections.Counter(m["pair"] for m in meta)))
    print("  花色:", dict(collections.Counter(m["suit"] for m in meta)))
    print("  高牌:", dict(collections.Counter(m["hi"] for m in meta)))
    if args.out:
        with open(args.out, "w") as f:
            f.write("\n".join(boards) + "\n")
        print(f"[已写] {args.out}")


if __name__ == "__main__":
    main()
