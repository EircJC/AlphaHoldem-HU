"""TexasSolver 场景生成器:按【深度 × 翻牌 × 翻前线】生成一批 console_solver 命令脚本 + manifest。

每个场景 = 一个翻后解算局面:给定深度(有效筹码)、翻前打到翻牌的底池与双方范围、翻牌、下注树。
console_solver 吃这些命令脚本,输出每局面的 GTO 混合策略 JSON。之后 solver_parse.py 把 JSON 转成
翻后软目标 JSONL,喂 BC 微调 4 个深度模型。

    python3 solver_gen.py --out-dir . --iters 120 --threads 8
    # 生成 inputs/*.txt 和 manifest.json;dump_result 写到 outputs/*.json(绝对路径)

小样默认:4 深度 × 5 翻牌 × 1 条线(单加注池 SRP)= 20 个解算。跑通后再加线/加翻牌放量。
深度→有效筹码(bb):short=40 / mid=130 / deep=250 / vdeep=400,对齐 4 个专精的训练中心。
"""
import argparse
import json
import os
import time

# ---- 深度档(bb) ----
DEPTHS = {"short": 40.0, "mid": 130.0, "deep": 250.0, "vdeep": 400.0}

# ---- 翻前线(小样只做单加注池 SRP:BTN/SB 开池2.5,BB 跟)----
# 翻后 OOP=BB(先动) / IP=SB(后动)。pot=双方各投2.5=5;eff=depth-2.5。
LINES = {
    "srp": {
        "desc": "SB(BTN)open2.5, BB call",
        "pot": 5.0,
        "invested": 2.5,   # 每人翻前已投,eff = depth - invested
        # 范围用规范式(AA / AKs / AKo / A5s ...),console_solver 可直接吃。
        # ★Layer3:放宽范围治"库覆盖不足→回退NN→输平衡reg A"。IP=top66% / OOP=top56%(重档)。
        #   ★口径:66%/56% 是【组合数占比】(IP 870/1326=65.6%,OOP 738/1326=55.7%);按【类数】是 74%/65%(125/110类)。
        #   因对子=6/同花=4/非同花=12组合权重不同,两口径不等。窄档 40%/34%(组合) → 重档 66%/56%,解算量≈2.67×。
        #   改此处后需【重灌库+重转DB】。
        #   旧窄范围(IP40%/OOP34%)已注释保留,回退把下面两行换回 _range_ip_narrow/_range_oop_narrow 即可。
        "range_ip":  "AA,KK,QQ,99,JJ,TT,77,88,55,66,33,44,22,AKs,AQs,AJs,AKo,ATs,KQs,QJs,AQo,A9s,JTs,KJs,A8s,AJo,QTs,KTs,A7s,KQo,A6s,J9s,Q9s,T9s,ATo,K9s,QJo,KJo,98s,A5s,A9o,K8s,J8s,Q8s,T8s,A4s,JTo,A8o,QTo,K7s,KTo,87s,97s,Q7s,A3s,T7s,A7o,J7s,K6s,T9o,J9o,Q6s,A2s,Q9o,A6o,76s,K5s,J6s,K9o,86s,96s,Q5s,A5o,T6s,K4s,98o,K8o,J5s,T8o,65s,J8o,Q4s,Q8o,A4o,T5s,75s,85s,K3s,K7o,95s,J4s,Q3s,87o,Q7o,T4s,97o,A3o,K2s,T7o,K6o,54s,J3s,J7o,64s,94s,74s,Q2s,Q6o,T3s,84s,A2o,K5o,76o,J2s,J6o,86o,93s,96o,Q5o,43s,T2s,83s,T6o,53s,K4o",
        "range_oop": "AA,KK,QQ,99,JJ,TT,77,88,55,66,33,44,22,AKs,AQs,AJs,AKo,ATs,KQs,QJs,AQo,A9s,JTs,KJs,A8s,AJo,QTs,KTs,A7s,KQo,A6s,J9s,Q9s,T9s,ATo,K9s,QJo,KJo,98s,A5s,A9o,K8s,J8s,Q8s,T8s,A4s,JTo,A8o,QTo,K7s,KTo,87s,97s,Q7s,A3s,T7s,A7o,J7s,K6s,T9o,J9o,Q6s,A2s,Q9o,A6o,76s,K5s,J6s,K9o,86s,96s,Q5s,A5o,T6s,K4s,98o,K8o,J5s,T8o,65s,J8o,Q4s,Q8o,A4o,T5s,75s,85s,K3s,K7o,95s,J4s,Q3s,87o,Q7o,T4s,97o,A3o,K2s,T7o,K6o,54s,J3s,J7o,64s,94s,74s,Q2s,Q6o,T3s,43s",
        # 旧窄范围(回退用):
        "_range_ip_narrow":  "22,33,44,55,66,77,88,99,TT,JJ,QQ,KK,AA,A2s,A3s,A4s,A5s,A6s,A7s,A8s,A9s,ATs,AJs,AQs,AKs,K2s,K3s,K4s,K5s,K6s,K7s,K8s,K9s,KTs,KJs,KQs,Q6s,Q7s,Q8s,Q9s,QTs,QJs,J7s,J8s,J9s,JTs,T7s,T8s,T9s,97s,98s,86s,87s,75s,76s,65s,54s,A2o,A3o,A4o,A5o,A6o,A7o,A8o,A9o,ATo,AJo,AQo,AKo,K9o,KTo,KJo,KQo,Q9o,QTo,QJo,J9o,JTo,T9o,98o",
        "_range_oop_narrow": "22,33,44,55,66,77,88,99,TT,JJ,QQ,A2s,A3s,A4s,A5s,A6s,A7s,A8s,A9s,ATs,AJs,AQs,K5s,K6s,K7s,K8s,K9s,KTs,KJs,KQs,Q7s,Q8s,Q9s,QTs,QJs,J7s,J8s,J9s,JTs,T7s,T8s,T9s,96s,97s,98s,85s,86s,87s,75s,76s,64s,65s,54s,43s,A7o,A8o,A9o,ATo,AJo,AQo,K9o,KTo,KJo,KQo,Q9o,QTo,QJo,J9o,JTo,T8o,T9o,98o",
    },
}

# ---- 代表翻牌(小样5个,覆盖 干/湿/高/连/对)----
BOARDS = [
    "Qs,Jh,2h",   # 两色高张
    "8c,3d,2h",   # 干燥低彩虹
    "Ah,Kd,7c",   # 高张干燥
    "9s,8s,5c",   # 湿润相连两色
    "Kd,Kh,4s",   # 对子面
]


def bet_tree_lines(tree="full"):
    """下注树三档:
      fast = 1档下注+全下、【无raise】→ 树最小,仅【冒烟测试】或【极浅筹】。深筹别用(面对注只能jam,退化)。
      med  = 1档下注 + 1档加注(60%池)+ 全下 → 有合理加注、深筹不退化,比full小,【正式数据推荐】。
      full = 2档下注 + 加注 + 全下 → 最精细最慢最吃内存。
    关键:深筹(mid/deep/vdeep)必须有 raise 档,否则面对下注只能 jam 全部身价或不加注 → 策略坏掉。
    所以 med 保留了 raise,只是把下注收成单档来省体积。"""
    L = []
    for pos in ("oop", "ip"):
        if tree == "fast":
            L += [f"set_bet_sizes {pos},flop,bet,50",
                  f"set_bet_sizes {pos},flop,allin",
                  f"set_bet_sizes {pos},turn,bet,66",
                  f"set_bet_sizes {pos},turn,allin",
                  f"set_bet_sizes {pos},river,bet,75",
                  f"set_bet_sizes {pos},river,allin"]
        elif tree == "med":
            # 单档下注 + 一档加注(60%池,合理大小,非 jam)+ 全下。
            # 【必须有 raise 档】否则深筹面对下注只能 jam 200bb 或不加注 → 策略退化。
            L += [f"set_bet_sizes {pos},flop,bet,50",
                  f"set_bet_sizes {pos},flop,raise,50",
                  f"set_bet_sizes {pos},flop,allin",
                  f"set_bet_sizes {pos},turn,bet,66",
                  f"set_bet_sizes {pos},turn,raise,50",
                  f"set_bet_sizes {pos},turn,allin",
                  f"set_bet_sizes {pos},river,bet,75",
                  f"set_bet_sizes {pos},river,raise,50",
                  f"set_bet_sizes {pos},river,allin"]
        else:  # full
            L += [f"set_bet_sizes {pos},flop,bet,33,75",
                  f"set_bet_sizes {pos},flop,raise,60",
                  f"set_bet_sizes {pos},flop,allin",
                  f"set_bet_sizes {pos},turn,bet,60",
                  f"set_bet_sizes {pos},turn,raise,60",
                  f"set_bet_sizes {pos},turn,allin",
                  f"set_bet_sizes {pos},river,bet,75",
                  f"set_bet_sizes {pos},river,raise,60",
                  f"set_bet_sizes {pos},river,allin"]
    return L


def make_cmd(scn, out_json, iters, threads, accuracy, tree="full", dump_rounds=2):
    L = [f"set_pot {scn['pot']:.2f}",
         f"set_effective_stack {scn['eff']:.2f}",
         f"set_board {scn['board']}",
         f"set_range_ip {scn['range_ip']}",
         f"set_range_oop {scn['range_oop']}"]
    L += bet_tree_lines(tree)
    L += ["set_allin_threshold 0.67",
          "build_tree",
          f"set_thread_num {threads}",
          f"set_accuracy {accuracy}",
          f"set_max_iteration {iters}",
          "set_print_interval 10",
          "set_use_isomorphism 1",
          "start_solve",
          f"set_dump_rounds {dump_rounds}",
          f"dump_result {out_json}"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=".", help="生成到哪(会建 inputs/ outputs/)")
    ap.add_argument("--iters", type=int, default=0, help="每局面最大迭代(0=自动:fast60/med80/full120)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--accuracy", type=float, default=0.5, help="收敛精度(越小越准越慢;0.5=可接受)")
    ap.add_argument("--lines", default="srp", help="逗号分隔的翻前线名(默认只 srp)")
    ap.add_argument("--tree", choices=["fast", "med", "full"], default="full",
                    help="下注树:fast=冒烟测试/med=正式数据推荐/full=最精细最慢")
    ap.add_argument("--fast", action="store_true", help="(等价 --tree fast)")
    ap.add_argument("--depths", default="",
                    help="只生成这些深度(逗号分隔,如 deep,vdeep;空=全部4档)")
    ap.add_argument("--boards", type=int, default=0,
                    help="只用前 N 个内置翻牌(如 2;0=全部5个)。深筹提速用")
    ap.add_argument("--flop-subset", type=int, default=0,
                    help="用纹理均衡代表子集的 N 个翻牌(如 25/49),替代内置5个。正式数据用这个")
    ap.add_argument("--flop-list-file", default="",
                    help="从文件读翻牌(每行一个,格式如 'Ah,Ks,2d'),替代 --flop-subset。用于只解指定/新增的牌面")
    ap.add_argument("--dump-rounds", type=int, default=2,
                    help="dump 到第几街:1=翻/2=翻转(推荐)/3=翻转河/4=全部。越大文件越爆炸(河层~48倍)")
    args = ap.parse_args()

    tree = "fast" if args.fast else args.tree
    iters = args.iters if args.iters > 0 else {"fast": 60, "med": 80, "full": 120}[tree]
    depths = {k: v for k, v in DEPTHS.items()
              if not args.depths or k in [x.strip() for x in args.depths.split(",")]}
    if args.flop_list_file:
        boards = [ln.strip() for ln in open(args.flop_list_file) if ln.strip()]
        print(f"[flop-list] 从 {args.flop_list_file} 读入 {len(boards)} 个翻牌")
    elif args.flop_subset > 0:
        import flop_subset
        boards = flop_subset.subset(args.flop_subset)[0]
    else:
        boards = BOARDS[:args.boards] if args.boards > 0 else BOARDS

    root = os.path.abspath(args.out_dir)
    in_dir = os.path.join(root, "inputs")
    out_dir = os.path.join(root, "outputs")
    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    want_lines = [x.strip() for x in args.lines.split(",") if x.strip()]
    manifest = []
    for line_name in want_lines:
        line = LINES[line_name]
        for depth_name, depth_bb in depths.items():
            eff = depth_bb - line["invested"]
            for bi, board in enumerate(boards):
                sid = f"{line_name}_{depth_name}_b{bi}"
                scn = {"id": sid, "line": line_name, "depth_name": depth_name,
                       "depth_bb": depth_bb, "pot": line["pot"], "eff": eff,
                       "board": board, "range_ip": line["range_ip"],
                       "range_oop": line["range_oop"]}
                out_json = os.path.join(out_dir, sid + ".json")
                cmd = make_cmd(scn, out_json, iters, args.threads, args.accuracy, tree, args.dump_rounds)
                in_path = os.path.join(in_dir, sid + ".txt")
                with open(in_path, "w") as f:
                    f.write(cmd)
                scn["cmd_file"] = in_path
                scn["out_file"] = out_json
                manifest.append(scn)

    # 合并到已有 manifest(按 id 去重累加),避免分多次生成不同深度时把之前的覆盖掉。
    # 想从头开始就先删掉 manifest.json + inputs/ outputs/。
    man_path = os.path.join(root, "manifest.json")
    existing = {}
    if os.path.exists(man_path):
        try:
            for s in json.load(open(man_path)):
                existing[s["id"]] = s
        except Exception:
            pass
    for s in manifest:
        existing[s["id"]] = s          # 同 id 用本次的覆盖(参数变了以新为准)
    merged = list(existing.values())
    with open(man_path, "w") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"[{time.strftime('%H:%M:%S')}] [生成完成] 本次 {len(manifest)} 个场景, "
          f"manifest 累计 {len(merged)} 个 → {in_dir}")
    print(f"           manifest → {man_path}")
    print(f"           本次深度={list(depths)} 翻牌={len(boards)}个 线={want_lines} "
          f"iters={iters} tree={tree}")


if __name__ == "__main__":
    main()
