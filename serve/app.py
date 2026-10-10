"""HTTP 服务:把 PokerBot 暴露成跨语言、多桌可调用的接口。

启动(配置走环境变量):
    MODEL_PATH=runs/spec_mid_L4A6/model_iter330.pt DB_PATH=gto_lib_wide.db \
    uvicorn serve.app:app --host 0.0.0.0 --port 8000 --workers 1

可选环境变量:
    NORM=500  GREEDY=0  EXPLOIT=1  PREFLOP_GUARD=1
    RIVER_SOLVE=0  SOLVER_DIR=TexasSolver-v0.2.0-MacOs  RIVER_ITERS=40
接口:
    POST /decide     body=局面JSON(见 engine_state.py 契约) → {action,to,add,bucket,meta}
    POST /hand_end   body=本手完整动作JSON(需 table_id, hero_seat, actions) → 更新对手读数
    POST /reset_table body={"table_id":...} → 对手换人时清读数
    GET  /health
注意:模型推理是有状态共享(单个 net + 每桌 tracker),用锁串行化,保证多桌并发下安全;
      高并发要更大吞吐用多进程(--workers>1,但那样每进程各自一份 tracker,需保证同一桌打到同一进程)。
"""
from __future__ import annotations
import os
import threading

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bot import PokerBot                                      # noqa: E402


def _b(name, default):
    return os.environ.get(name, str(default)).lower() in ("1", "true", "yes", "on")


app = FastAPI(title="PokerBot serve", version="1.0")
_bot: Optional[PokerBot] = None
_lock = threading.Lock()


@app.on_event("startup")
def _startup():
    global _bot
    mp = os.environ.get("MODEL_PATH")
    if not mp:
        raise RuntimeError("必须设 MODEL_PATH 环境变量(模型 .pt 路径)")
    _bot = PokerBot(
        model_path=mp,
        db_path=os.environ.get("DB_PATH") or None,
        norm=float(os.environ.get("NORM", "500")),
        greedy=_b("GREEDY", False),
        exploit=_b("EXPLOIT", True),
        preflop_guard=_b("PREFLOP_GUARD", True),
        river_solve=_b("RIVER_SOLVE", False),
        solver_dir=os.environ.get("SOLVER_DIR", "TexasSolver-v0.2.0-MacOs"),
        river_iters=int(os.environ.get("RIVER_ITERS", "40")),
    )
    print(f"[serve] 模型={mp} 库={os.environ.get('DB_PATH')} 河解={_b('RIVER_SOLVE', False)} 就绪")


class Action(BaseModel):
    street: int
    actor: int
    type: str
    to: Optional[float] = None


class DecideReq(BaseModel):
    table_id: str                                            # 必填:桌号(多桌隔离对手读数)
    hand_id: str                                             # 必填:当前手牌 id(同桌可多手);出参回传 + /hand_end 去重
    hero_seat: int
    button: int
    sb: float = 0.5
    bb: float = 1.0
    start_stacks: List[float]
    hero_hole: List[str]
    board: List[str] = []
    actions: List[Action] = []


class TableReq(BaseModel):
    table_id: str


@app.get("/health")
def health():
    return {"ok": _bot is not None}


@app.post("/decide")
def decide(req: DecideReq):
    if _bot is None:
        raise HTTPException(503, "bot 未就绪")
    body: Dict[str, Any] = req.dict()
    with _lock:
        try:
            out = _bot.decide(body)
        except Exception as e:
            raise HTTPException(400, f"decide 失败: {type(e).__name__}: {e}")
    out["table_id"] = req.table_id                           # 回传桌号+手牌id 供服务端关联/校验幂等
    out["hand_id"] = req.hand_id
    return out


@app.post("/hand_end")
def hand_end(req: DecideReq):
    if _bot is None:
        raise HTTPException(503, "bot 未就绪")
    with _lock:
        _bot.hand_end(req.table_id, req.dict())
    return {"ok": True}


@app.post("/reset_table")
def reset_table(req: TableReq):
    if _bot is None:
        raise HTTPException(503, "bot 未就绪")
    with _lock:
        _bot.reset_table(req.table_id)
    return {"ok": True}


@app.get("/opp_read")
def opp_read(table_id: str):
    """只读:返回该桌对手当前画像 {n,loose,pfr,af,call_ratio,label,confident}(供对打日志展示)。"""
    if _bot is None:
        raise HTTPException(503, "bot 未就绪")
    with _lock:
        return _bot.opp_read(table_id)
