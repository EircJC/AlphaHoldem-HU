"""翻后 solver JSONL → BC 软目标 npz(格式与 preflop_etl 完全一致,可直接喂 preflop_bc.py)。

输入:solver_parse.py 产的 solver_postflop.jsonl,每行 =
  {id, depth_bb, eff_bb, board, street, pot, player,
   seq:[[player,bucket,street],...], hand:"AsAh",
   strategy_raw:{...}, strategy_bucket:{"1":0.6,"6":0.4}}

每行 → 一条 BC 样本:hero(=player)在 (board + 翻后历史 + 筹码) 下,拿 hand 的目标动作分布(软标签)。
复用 encoder.encode(),与 bc_etl / RL 同一口径。历史用 pm=0(hero)/1(opp),street 1=翻/2=转/3=河。

    python3 solver_etl.py --jsonl solver_postflop.jsonl --out solver_ds.npz --norm 500
    python3 solver_etl.py --jsonl solver_postflop.jsonl --out solver_deep.npz --norm 500 --depths deep
    python3 solver_etl.py --selftest        # 不需数据,合成几行自检

--norm 必须与 postflop BC / RL / 评估全程一致(500)。--start-stack 估算当前剩余筹码用。
产出 npz 键:cards, actions, scalars, legal, target_soft, weight, targets, norm —— 与 preflop_etl 一致。
之后训练:python3 preflop_bc.py --preflop solver_ds.npz --resume <某专精.pt> ...(preflop_bc 吃软目标)
"""
import argparse
import json
import os
import random
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import NB, CALL                                   # noqa: E402
from encoder import encode                                    # noqa: E402
from engine.evaluator import evaluate7                        # noqa: E402

RANKC = "23456789TJQKA"
SUITC = "shdc"
# 深度带 → 名字(和 solver_gen 的 DEPTHS 对齐,用于 --depths 过滤)
BANDS = {"short": (0, 55), "mid": (55, 200), "deep": (200, 320), "vdeep": (320, 9999)}


def _card(s):
    return RANKC.index(s[0].upper()) * 4 + SUITC.index(s[1].lower())


def parse_cards_str(s):
    """'AsAh' → [c,c];'Qs,Jh,2h,Ac' → [c,c,c,c]。"""
    s = (s or "").strip()
    if not s:
        return []
    if "," in s:
        return [_card(x.strip()) for x in s.split(",") if x.strip()]
    return [_card(s[i:i + 2]) for i in range(0, len(s), 2)]


def band_of(depth_bb):
    for name, (lo, hi) in BANDS.items():
        if lo <= depth_bb < hi:
            return name
    return "?"


def hand_cat(rec):
    """hero 当前成手类别(0高牌..4顺子..5同花..6葫芦..7四条..8同花顺);牌<5张或解析失败→-1。"""
    try:
        cards = parse_cards_str(rec.get("hand", "")) + parse_cards_str(rec.get("board", ""))
        if len(cards) < 5:
            return -1
        return evaluate7(cards)[0]
    except Exception:
        return -1


def build_row(rec, norm, start_stack):
    """一条 JSONL → (cards, actions, scalars, legal, soft_target)。"""
    hero = int(rec["player"])
    hole = parse_cards_str(rec["hand"])
    board = parse_cards_str(rec["board"])
    # 历史:seq=[[player,bucket,street],...] → history[4],pm=0(hero)/1(opp)
    hist = [[], [], [], []]
    for p, bkt, street in rec["seq"]:
        pm = 0 if int(p) == hero else 1
        st = int(street)
        if 0 <= st <= 3:
            hist[st].append((pm, int(bkt)))
    # 软目标:strategy_bucket → NB 维分布
    soft = np.zeros(NB, dtype=np.float32)
    for k, v in rec["strategy_bucket"].items():
        soft[int(k)] += float(v)
    s = soft.sum()
    if s > 1e-9:
        soft /= s
    # 合法掩码:该节点解算给出的档 ∪ 跟注(保证目标合法)
    legal = [False] * NB
    legal[CALL] = True
    for b in range(NB):
        if soft[b] > 0:
            legal[b] = True
    # 当前筹码/底池(近似,和 bc_etl 同思路):池比翻牌起始多出的部分两人各摊一半
    pot = float(rec["pot"])
    eff = float(rec.get("eff_bb", start_stack))
    pot0 = 5.0                                   # SRP 翻牌起始底池(两人各 2.5bb)
    my_stack = max(eff - max(pot - pot0, 0.0) / 2.0, 1e-3)
    state = {"player": 0, "hole": hole, "board": board, "history": hist,
             "legal": legal, "my_stack": my_stack, "opp_stack": my_stack,
             "pot": pot, "stack_norm": norm}
    c, a, sc, lg = encode(state)
    return c, a, sc, lg, soft


def _selftest():
    rows = [
        {"player": 1, "depth_bb": 250, "eff_bb": 247.5, "board": "Qs,Jh,2h", "street": 1,
         "pot": 5.0, "seq": [], "hand": "AsAd",
         "strategy_bucket": {"1": 0.7, "6": 0.3}},
        {"player": 0, "depth_bb": 250, "eff_bb": 247.5, "board": "Qs,Jh,2h,Ac", "street": 2,
         "pot": 15.0, "seq": [[1, 1, 1], [0, 6, 1]], "hand": "KdKh",
         "strategy_bucket": {"1": 0.4, "12": 0.6}},
    ]
    for r in rows:
        c, a, sc, lg, soft = build_row(r, 500.0, 250.0)
        assert abs(soft.sum() - 1.0) < 1e-5, soft.sum()
        assert c.shape[0] == 6 and lg.shape[0] == NB
    print("[selftest] OK — 编码/软目标/合法掩码 都正常")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", help="solver_parse 产的 solver_postflop.jsonl")
    ap.add_argument("--out", help="输出 npz")
    ap.add_argument("--norm", type=float, default=500.0, help="标量归一化基准,须与训练一致(500)")
    ap.add_argument("--start-stack", type=float, default=250.0, help="缺 eff_bb 时的兜底起始筹码")
    ap.add_argument("--depths", default="", help="只保留这些深度带(逗号,如 deep 或 short,mid);空=全部")
    ap.add_argument("--sample", type=float, default=1.0,
                    help="二次抽样保留比例(0~1)。某档行数太大、ETL 内存吃紧时用(如 0.5 减半),无需重解析")
    ap.add_argument("--passive-keep", type=float, default=1.0,
                    help="动作再平衡:被动样本(GTO 下注质量<5%,即几乎全过牌/跟注/弃牌)的保留比例。"
                         "1.0=不平衡;0.3=被动只留三成,让下注样本别被淹没→BC 不会把侵略压没。治过度被动。")
    ap.add_argument("--strong-weight", type=float, default=1.0,
                    help="强成手(≥--strong-cat,默认同花+)样本的价值加权 + 抽样豁免。1.0=关;3~8=让 BC 学会"
                         "对坚果类【该加/该跟】(治'拿四条只跟、弃坚果同花')。开启会对每行评牌型,ETL 变慢。")
    ap.add_argument("--strong-cat", type=int, default=5,
                    help="判为'强成手'的最低牌型(evaluate7 类别:4顺子 5同花 6葫芦 7四条 8同花顺)。默认5=同花+")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest(); return
    if not args.jsonl or not args.out:
        ap.error("需 --jsonl 和 --out(或 --selftest)")

    want = {x.strip() for x in args.depths.split(",") if x.strip()}
    cards, acts, scals, legals, softs, ws = [], [], [], [], [], []
    n = ok = bad = skip = 0
    n_active = n_passive = n_strong = 0        # 再平衡/强牌统计(保留下来的)
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] 开始:{args.jsonl}")
    with open(args.jsonl) as f:
        for line in f:
            n += 1
            try:
                rec = json.loads(line)
                if want and band_of(float(rec["depth_bb"])) not in want:
                    skip += 1; continue
                # 强成手识别(仅开启加权时才评牌型,省时):强牌豁免抽样 + 价值加权
                strong = args.strong_weight != 1.0 and hand_cat(rec) >= args.strong_cat
                if args.sample < 1.0 and not strong and random.random() > args.sample:
                    skip += 1; continue
                # 动作再平衡:被动样本(GTO 下注质量<5%)按 passive_keep 降采样(强牌天然主动,豁免)
                if args.passive_keep < 1.0:
                    bet_mass = sum(float(v) for k, v in rec["strategy_bucket"].items() if int(k) >= 2)
                    is_active = bet_mass >= 0.05
                    if not is_active and not strong and random.random() > args.passive_keep:
                        skip += 1; continue
                    n_active += 1 if is_active else 0
                    n_passive += 0 if is_active else 1
                c, a, sc, lg, soft = build_row(rec, args.norm, args.start_stack)
                cards.append(c.astype(np.uint8)); acts.append(a.astype(np.uint8))
                scals.append(sc.astype(np.float16)); legals.append(lg.astype(np.uint8))
                softs.append(soft); ws.append(args.strong_weight if strong else 1.0)
                if strong:
                    n_strong += 1
                ok += 1
            except Exception:
                bad += 1
            if n % 200000 == 0:
                gb = ok * 2049 / 1e9      # 每样本各张量合计约 2KB;估算入内存占用
                print(f"[{time.strftime('%H:%M:%S')}]  ...{n} 行 (有效{ok}/跳过{skip}/坏{bad}) "
                      f"~{gb:.1f}GB内存 {time.time()-t0:.0f}s", flush=True)
                if ok > 6_000_000:
                    print("   [警告] 有效样本已 >600万,入内存可能 >12GB。若担心 OOM,"
                          "用更小的 --sample 重跑 solver_parse(或分档 ETL)。", flush=True)

    if ok == 0:
        print("没有有效样本(检查 --jsonl / --depths)"); sys.exit(1)
    soft = np.stack(softs).astype(np.float32)
    np.savez_compressed(args.out,
                        cards=np.stack(cards), actions=np.stack(acts),
                        scalars=np.stack(scals), legal=np.stack(legals),
                        target_soft=soft, weight=np.asarray(ws, np.float32),
                        targets=soft.argmax(1).astype(np.int16), norm=np.float32(args.norm))
    dist = soft.sum(0)
    print(f"[{time.strftime('%H:%M:%S')}] [完成] 有效 {ok}/总 {n} (跳过{skip} 坏{bad}) "
          f"用时 {(time.time()-t0)/60:.1f}分 → {args.out}")
    if args.passive_keep < 1.0:
        tot = n_active + n_passive
        print(f"  [再平衡] 保留:主动(有下注){n_active} / 被动{n_passive}  "
              f"→ 被动占比 {100*n_passive/max(tot,1):.0f}% (passive_keep={args.passive_keep})")
    if args.strong_weight != 1.0:
        _cn = {4: "顺子+", 5: "同花+", 6: "葫芦+", 7: "四条+", 8: "同花顺"}.get(args.strong_cat, f"类别{args.strong_cat}+")
        print(f"  [强牌加权] 强成手(≥{_cn})保留 {n_strong} 条 "
              f"({100*n_strong/max(ok,1):.1f}%),权重×{args.strong_weight}、且豁免抽样 → 治'坚果只跟/弃坚果'")
    print("  软目标质量分布(各档累计概率):")
    for b in range(NB):
        if dist[b] > 0.5:
            print(f"    档{b:2d}: {dist[b]:.0f}  ({dist[b]/ok*100:.1f}%)")
    print("  下一步训练(preflop_bc 吃软目标):")
    print(f"    python3 preflop_bc.py --preflop {args.out} --norm {args.norm:.0f} ...")


if __name__ == "__main__":
    main()
