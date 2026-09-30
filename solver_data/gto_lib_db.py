"""GTO 库 SQLite 查询层:只读单节点,常驻内存几乎为零(替代整板 load_lib)。

配 lib_to_sqlite.py 建出的 gto_lib.db 使用。查库路径与内存库【完全同一套导航逻辑】
(gto_opponent._flopturn_nav),只是节点来源从"整块 dict"换成"DB 单行查询"。

  python3 gto_lib_db.py --show gto_lib.db      # 看 DB 概况(档/板数/节点数)
"""
import argparse
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(__file__))
import gto_lib                                   # noqa: E402
import board_match as bm                         # noqa: E402
import gto_opponent as go                        # noqa: E402
import lib_to_sqlite as l2s                      # noqa: E402  (sig_str/unpack_node)


class LibDB:
    """只读打开 gto_lib.db。select_band→按有效筹码就近选档;match→板匹配+花色重映射;get_node→单节点。"""

    def __init__(self, db_path, mmap_gb=2):
        if not os.path.exists(db_path):
            raise FileNotFoundError(db_path)
        self.con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True,
                                   check_same_thread=False)
        try:
            self.con.execute(f"PRAGMA mmap_size={int(mmap_gb * 1e9)}")
            self.con.execute("PRAGMA query_only=ON")
        except sqlite3.Error:
            pass
        self._cur = self.con.cursor()
        self._bands = list(self.con.execute("SELECT name, depth_bb FROM bands"))
        if not self._bands:
            raise ValueError(f"{db_path} 没有 bands 表内容,可能不是有效的 GTO 库 DB")

    # ---- 档选择 ----
    def select_band(self, eff_bb, lo_ratio=0.5, hi_ratio=1.7):
        """有效筹码 → 最近深度档名;无档或【离最近档太远】返回 None(→ 回退 NN)。
        深度容差:库里每个节点是在该档特定 SPR 下解出的,拿 250bb 策略套 1000bb 局面是错的。
        只有 eff 落在最近档的 [lo_ratio, hi_ratio]×depth 区间内才用库,否则不查(避免错档策略)。
        现有档 40/130/250 → 覆盖约 20~425bb;更深(deep 上半段/super-deep)自动回退 NN+护栏+河解。"""
        if not self._bands:
            return None
        name, depth = min(self._bands, key=lambda b: abs(b[1] - eff_bb))
        if depth <= 0:
            return None
        r = eff_bb / depth
        if r < lo_ratio or r > hi_ratio:
            return None                       # 离最近档太远 → 不查库,回退 NN(见 flopturn_bucket_db)
        return name

    def band_names(self):
        return {n: d for n, d in self._bands}

    # ---- 板匹配(复用 board_match 的花色规范化/置换,行为与内存库一致)----
    def match(self, band, real_board, hole):
        """真实(board,hole)→(solved_flop, 重映射整板, 重映射底牌)或 None(库未覆盖)。"""
        rc = bm._cards(real_board)
        sig = bm.canon_sig([bm._fmt(*c) for c in rc[:3]])
        row = self._cur.execute(
            "SELECT solved_flop FROM boards WHERE band=? AND flop_sig=?",
            (band, l2s.sig_str(sig))).fetchone()
        if row is None:
            return None
        return bm.match({sig: (row[0], None)}, real_board, hole)

    # ---- 单节点(供导航)----
    def get_node(self, band, board_key, path):
        nk = gto_lib.node_key(board_key, path)
        row = self._cur.execute(
            "SELECT node FROM nodes WHERE band=? AND node_key=?", (band, nk)).fetchone()
        return l2s.unpack_node(row[0]) if row is not None else None

    def nav(self, band, real_board, hole, history, cur_street, legal, greedy, rng):
        """板匹配 + 导航采样,返回 (动作档 或 None, 原因)。与内存库 flopturn_action 同结果。"""
        m = self.match(band, real_board, hole)
        if m is None:
            return None, "board_not_covered"
        _solved, rboard, rhole = m
        return go._flopturn_nav(lambda bk, p: self.get_node(band, bk, p),
                                rboard, rhole, history, cur_street, legal, greedy, rng)

    def count(self):
        """总节点数:优先读转换时写好的 meta.n_nodes(O(1));没有再退回 COUNT(*)(全表扫描,慢)。"""
        try:
            row = self.con.execute("SELECT v FROM meta WHERE k='n_nodes'").fetchone()
            if row is not None:
                return int(row[0])
        except sqlite3.Error:
            pass
        return self.con.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]

    def close(self):
        self.con.close()


def flopturn_bucket_db(db, env, seat, greedy=False, rng=None):
    """查 GTO 库(SQLite),给 seat 在翻/转的动作档。签名对齐 gto_opponent.flopturn_bucket,
    可直接替换服务链里的内存库版。返回 (bucket 或 None, reason)。"""
    from engine import card_str
    try:
        board = [card_str(c) for c in env.board]
        hole = [card_str(c) for c in env.holes[seat]]
        band = db.select_band(min(env.start_stacks))
        if band is None:
            return None, "no_band"
        history = {1: list(env.history[1]), 2: list(env.history[2])}
        return db.nav(band, ",".join(board), ",".join(hole), history, env.street,
                      list(env.legal_mask()), greedy, rng)
    except Exception as e:                                  # 单手异常一律回退,别崩整场
        return None, f"exc:{type(e).__name__}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", metavar="DB", help="打印 DB 概况")
    args = ap.parse_args()
    if args.show:
        db = LibDB(args.show)
        nb = db.con.execute("SELECT band, COUNT(*) FROM boards GROUP BY band").fetchall()
        nn = db.con.execute("SELECT band, COUNT(*) FROM nodes GROUP BY band").fetchall()
        print(f"DB: {args.show}  {os.path.getsize(args.show)/1e9:.2f}GB")
        print(f"  档(depth_bb): {db.band_names()}")
        print(f"  每档板数: {dict(nb)}")
        print(f"  每档节点数: {dict(nn)}")
        return
    ap.error("用 --show DB")


if __name__ == "__main__":
    main()
