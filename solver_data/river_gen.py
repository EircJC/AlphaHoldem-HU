"""方案A 收尾:把一个已解 flop(dump-2)的【进河牌点】× 代表性河牌 → 生成河牌单街解输入。

流程:river_range 枚举进河牌点(board4/pot/behind/IP范围/OOP范围)
  → 对每个点采样若干河牌(去阻断)→ 写 TexasSolver 河牌输入(5张牌 + 抽出的范围)
  → 输出 inputs_river/*.txt + manifest_river.json(供 console_solver 解、solver_parse 解析)。

每个河牌解:单街、几MB、秒级 → 24GB 毫无压力。范围是从 flop+turn 解真均衡抽下来的(自洽)。

用法:
  python3 river_gen.py --json outputs/srp_mid_b0.json --input inputs/srp_mid_b0.txt \
      --id-prefix srp_mid_b0 --depth-bb 130 --river-samples 4 \
      --turn-cap 3 --line-cap 40 --out-dir river_out
"""
import argparse
import json
import os

import river_range as rr

RANKS = "23456789TJQKA"
SUITS = "cdhs"
ALL_CARDS = [r + s for r in RANKS for s in SUITS]


def river_bet_lines():
    """河牌下注树(单街):下注75%/加注50%/全下,IP+OOP。"""
    L = []
    for pos in ("oop", "ip"):
        L += [f"set_bet_sizes {pos},river,bet,75",
              f"set_bet_sizes {pos},river,raise,50",
              f"set_bet_sizes {pos},river,allin"]
    return L


def sample_rivers(board4, ip_reach, oop_reach, n):
    """从未在桌/未完全阻断的牌里,均匀取 n 张河牌。"""
    used = set(board4)
    cand = [c for c in ALL_CARDS if c not in used]
    # 只保留"发出后双方仍各有牌"的河牌(极端阻断的跳过)
    ok = []
    for c in cand:
        ip_left = any(c not in (k[:2], k[2:]) for k in ip_reach)
        oop_left = any(c not in (k[:2], k[2:]) for k in oop_reach)
        if ip_left and oop_left:
            ok.append(c)
    if n <= 0 or n >= len(ok):
        return ok
    step = len(ok) / n                                # 均匀取样(跨点数/花色)
    return [ok[int(i * step)] for i in range(n)]


def build_river_input(out_json, board5, pot, eff, ip_str, oop_str, threads, iters, accuracy):
    L = [f"set_pot {pot:.2f}", f"set_effective_stack {eff:.2f}",
         f"set_board {board5}", f"set_range_ip {ip_str}", f"set_range_oop {oop_str}"]
    L += river_bet_lines()
    L += ["set_allin_threshold 0.67", "build_tree",
          f"set_thread_num {threads}", f"set_accuracy {accuracy}",
          f"set_max_iteration {iters}", "set_print_interval 50",
          "set_use_isomorphism 0", "start_solve", "set_dump_rounds 1", f"dump_result {out_json}"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="已解 flop(dump-2) JSON")
    ap.add_argument("--input", required=True, help="对应 solver_gen 输入(读 pot/eff/board/范围)")
    ap.add_argument("--id-prefix", required=True, help="河牌场景 id 前缀(如 srp_mid_b0)")
    ap.add_argument("--depth-bb", type=float, default=0, help="标注深度(供 ETL 分档;0=用 eff)")
    ap.add_argument("--turn-cap", type=int, default=3, help="每翻牌线最多几张转牌")
    ap.add_argument("--line-cap", type=int, default=40, help="进河牌点上限")
    ap.add_argument("--river-samples", type=int, default=4, help="每个进河牌点采样几张河牌")
    ap.add_argument("--min-behind", type=float, default=2.0, help="剩余筹码<此值跳过(全下runout无需解)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--accuracy", type=float, default=0.5)
    ap.add_argument("--out-dir", default="river_out")
    args = ap.parse_args()

    tree = json.load(open(args.json))
    cfg = rr._parse_input(args.input)
    entries = rr.enumerate_river_entries(tree, cfg["ip"], cfg["oop"], cfg["pot"], cfg["eff"],
                                         cfg["board"], turn_cap=args.turn_cap, line_cap=args.line_cap)
    in_dir = os.path.join(args.out_dir, "inputs")
    out_dir = os.path.join(args.out_dir, "outputs")
    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    if os.path.exists(os.path.join(args.out_dir, "manifest.json")):
        manifest = json.load(open(os.path.join(args.out_dir, "manifest.json")))
    seen = {m["id"] for m in manifest}
    n_scn = 0
    for ei, (board4, pot, behind, ip_reach, oop_reach, prefix) in enumerate(entries):
        if behind < args.min_behind:
            continue
        for rc in sample_rivers(board4, ip_reach, oop_reach, args.river_samples):
            board5 = board4 + [rc]
            ip5 = {k: v for k, v in ip_reach.items() if rc not in (k[:2], k[2:])}
            oop5 = {k: v for k, v in oop_reach.items() if rc not in (k[:2], k[2:])}
            ip_str = rr.combos_to_range_str(ip5, board5)
            oop_str = rr.combos_to_range_str(oop5, board5)
            if not ip_str or not oop_str:
                continue
            sid = f"{args.id_prefix}_e{ei}_{rc}"
            if sid in seen:
                continue
            oj = os.path.abspath(os.path.join(out_dir, sid + ".json"))
            txt = build_river_input(oj, ",".join(board5), pot, behind, ip_str, oop_str,
                                    args.threads, args.iters, args.accuracy)
            with open(os.path.join(in_dir, sid + ".txt"), "w") as f:
                f.write(txt)
            # 河牌SOLVE 的 set_effective_stack 用 behind(河牌起始剩余);
            # 但 ETL 的 eff_bb 必须用【完整翻牌 eff】(cfg['eff']),否则 build_row 的
            # my_stack = eff-(pot-5)/2 会把翻+转投入扣两遍(behind 已扣过一次)。
            manifest.append({"id": sid, "out_file": oj, "board": ",".join(board5),
                             "pot": pot, "eff": cfg["eff"],
                             "depth_bb": args.depth_bb or round(cfg["eff"]),
                             "prefix": prefix})          # 翻+转动作历史,solver_parse 拼到每行 seq 前
            seen.add(sid)
            n_scn += 1
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, ensure_ascii=False)
    print(f"[river_gen] 进河牌点 {len(entries)} → 生成河牌场景 {n_scn} 个 "
          f"(manifest 累计 {len(manifest)}) → {args.out_dir}/inputs")
    print(f"  下一步: 逐个 solve inputs/*.txt → solver_parse --manifest {args.out_dir}/manifest.json")


if __name__ == "__main__":
    main()
