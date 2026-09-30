"""serve 自测:
  python3 serve/selftest.py                       # 只测局面重建(无 torch)
  python3 serve/selftest.py --model runs/spec_mid_L4A5/model_iter280.pt --db gto_lib_wide.db
                                                  # 额外真跑一次 decide(需 torch)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import engine_state as es
from config import FOLD, CALL


def test_state():
    # 翻前:button=0(SB) raise to 3,hero=1(BB) 面对
    req = {"table_id": "t", "hero_seat": 1, "button": 0, "start_stacks": [200, 200],
           "hero_hole": ["Ah", "Ks"], "board": [], "actions": [{"street": 0, "actor": 0, "type": "raise", "to": 3.0}]}
    env, h = es.build_env(req)
    assert env.to_act == 1 and env.current_bet == 3.0 and env.street == 0
    assert env.legal_mask()[FOLD] and env.legal_mask()[CALL]           # 面对加注:可弃可跟
    assert es.bucket_to_action(env, CALL, h)["add"] == 2.0             # 跟需补 2
    # 翻牌 hero 先手(raise-call 后无动作)
    req2 = dict(req); req2["board"] = ["Qs", "Jh", "2h"]
    req2["actions"] = [{"street": 0, "actor": 0, "type": "raise", "to": 3},
                       {"street": 0, "actor": 1, "type": "call"}]
    env2, _ = es.build_env(req2)
    assert env2.street == 1 and env2.street_bet == [0.0, 0.0] and env2.current_bet == 0.0
    assert env2.legal_mask()[CALL] and not env2.legal_mask()[FOLD]     # 无注:可过牌不可弃
    assert abs(sum(env2.committed) - 6.0) < 1e-9
    # 转牌面对下注
    req3 = dict(req2); req3["board"] = ["Qs", "Jh", "2h", "Td"]
    req3["actions"] = req2["actions"] + [{"street": 1, "actor": 1, "type": "check"},
                                         {"street": 1, "actor": 0, "type": "check"},
                                         {"street": 2, "actor": 0, "type": "bet", "to": 4}]
    env3, _ = es.build_env(req3)
    assert env3.street == 2 and env3.current_bet == 4.0
    assert env3.legal_mask()[FOLD]                                     # 面对下注可弃
    print("  [局面重建] 翻前/翻牌先手/转牌面对下注/动作换算 全部✓")


def test_bot(model, db):
    from bot import PokerBot
    bot = PokerBot(model_path=model, db_path=db, greedy=True, river_solve=False)
    req = {"table_id": "t1", "hero_seat": 1, "button": 0, "start_stacks": [200, 200],
           "hero_hole": ["As", "Ad"], "board": ["Qs", "Jh", "2h"],
           "actions": [{"street": 0, "actor": 0, "type": "raise", "to": 3},
                       {"street": 0, "actor": 1, "type": "call"}]}
    r = bot.decide(req)
    assert r["action"] in ("fold", "check", "call", "raise", "allin")
    print(f"  [模型 decide] AA 翻牌 QsJh2h → {r['action']} to={r['to']} 来源={r['meta']['source']} 护栏={r['meta']['guards']} ✓")
    bot.hand_end("t1", {**req, "actions": req["actions"] + [{"street": 1, "actor": 1, "type": "check"}]})
    print("  [hand_end] 对手读数更新 OK ✓")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--db", default=None)
    args = ap.parse_args()
    print("== serve 自测 ==")
    test_state()
    if args.model:
        print("[真模型 decide]:")
        test_bot(args.model, args.db)
    else:
        print("[模型 decide]:跳过(传 --model 可测,需 torch)")
    print("== 通过 ✓ ==")


if __name__ == "__main__":
    main()
