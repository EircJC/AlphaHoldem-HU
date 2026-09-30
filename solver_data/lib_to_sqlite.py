"""把 GTO 库(per-board .lib.json.gz)转成 per-node SQLite,查库时只读单个节点。

为什么:原来查库要把整块板(60MB JSON→内存 ~600MB)整个 load 进来才能取 1 个节点,
随机翻牌下 LRU 命中率极低 → 几乎每手解析一整块(实测 ~5s/手)+ 反复加载/释放把内存推到几十 GB。
转成 SQLite 后:board→solved_flop 匹配索引和每个节点都按主键存,查一个节点只读那一行(~0.1ms),
常驻内存几乎为零(SQLite mmap),彻底消除整板加载与 OOM。

表结构:
  nodes(band, node_key, node)   node = gzip(json({pot,actions,strat}))   PRIMARY KEY(band,node_key)
  boards(band, flop_sig, solved_flop)   flop_sig=花色无关规范签名字符串   PRIMARY KEY(band,flop_sig)
  bands(name, depth_bb)                 每档代表深度(按有效筹码就近选带)
  done_files(path)                      已入库文件(断点续跑用)

用法:
  python3 lib_to_sqlite.py gto_libdir gto_lib.db          # 全量转换(可中断,重跑自动续)
  python3 lib_to_sqlite.py gto_libdir/deep gto_lib.db --band deep   # 只转一个档
  python3 lib_to_sqlite.py --selftest                     # 合成库→建DB→与内存库对拍
"""
import argparse
import glob
import gzip
import json
import os
import sqlite3
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import gto_lib                                   # noqa: E402
import board_match as bm                         # noqa: E402


def sig_str(sig):
    """canon_sig 返回的 tuple((rank,suit)...) → 稳定字符串,做主键。"""
    return "|".join(f"{r}{s}" for r, s in sig)


def _open_db(db_path):
    con = sqlite3.connect(db_path)
    # WAL+NORMAL:批量插入够快,又能在进程被杀/机器休眠/中断时不损坏库(journal=OFF 一旦中断写就可能损坏→
    # 断点续跑读不回 done_files 就白费)。转换结束会 wal_checkpoint(TRUNCATE) 把 WAL 并回主库,产出单个干净 .db。
    con.executescript(
        "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA temp_store=MEMORY;")
    con.execute("CREATE TABLE IF NOT EXISTS nodes"
                "(band TEXT, node_key TEXT, node BLOB, PRIMARY KEY(band,node_key)) WITHOUT ROWID")
    con.execute("CREATE TABLE IF NOT EXISTS boards"
                "(band TEXT, flop_sig TEXT, solved_flop TEXT, PRIMARY KEY(band,flop_sig))")
    con.execute("CREATE TABLE IF NOT EXISTS bands(name TEXT PRIMARY KEY, depth_bb REAL)")
    con.execute("CREATE TABLE IF NOT EXISTS done_files(path TEXT PRIMARY KEY)")
    return con


def _pack_node(node):
    """只留导航要用的字段(pot/actions/strat),gzip 压缩存 blob。"""
    slim = {"pot": node.get("pot", 1.0), "actions": node["actions"], "strat": node["strat"]}
    return gzip.compress(json.dumps(slim, separators=(",", ":")).encode("utf-8"))


def unpack_node(blob):
    return json.loads(gzip.decompress(blob).decode("utf-8"))


def _band_of(lib_dir, root):
    """档名:root 下的子目录名(short/mid/deep/vdeep);root 本身=扁平库 → 'flat'。"""
    b = os.path.basename(lib_dir.rstrip("/"))
    return b if os.path.abspath(lib_dir) != os.path.abspath(root) else "flat"


def convert_one(con, path, band):
    """把一块 .lib.json.gz 写进 DB(峰值内存=这一块)。返回 (节点数, depth_bb)。"""
    lib = gto_lib.load_lib(path)                       # 一次一块,用完即释放
    meta = lib.get("meta", {})
    flop = meta["flop"]
    dep = meta.get("depth_bb")
    if dep is None:
        dep = bm._NAME_BB.get(band.lower(), 100.0)
    con.execute("INSERT OR REPLACE INTO boards VALUES(?,?,?)",
                (band, sig_str(bm.canon_sig(flop)), flop))
    rows = [(band, k, _pack_node(v)) for k, v in lib["nodes"].items()]
    con.executemany("INSERT OR REPLACE INTO nodes VALUES(?,?,?)", rows)
    con.execute("INSERT OR REPLACE INTO bands VALUES(?,?)", (band, float(dep)))
    con.execute("INSERT OR REPLACE INTO done_files VALUES(?)", (path,))
    con.commit()
    return len(rows), float(dep)


def convert(root, db_path):
    """root 下所有 band 子目录(+可能的扁平库)全部转进 db_path。断点续跑:已入库文件跳过。"""
    con = _open_db(db_path)
    done = {r[0] for r in con.execute("SELECT path FROM done_files")}
    subdirs = [d for d in sorted(glob.glob(os.path.join(root, "*")))
               if os.path.isdir(d) and glob.glob(os.path.join(d, "*.lib.json.gz"))]
    targets = list(subdirs)
    if glob.glob(os.path.join(root, "*.lib.json.gz")):
        targets.append(root)
    if not targets:
        print(f"[转换] {root} 下没找到 .lib.json.gz")
        return
    t0 = time.time()
    total_nodes = 0
    total_files = sum(len(glob.glob(os.path.join(d, "*.lib.json.gz"))) for d in targets)
    seen = 0
    print(f"[转换] 目标档: {[os.path.basename(d.rstrip('/')) or '.' for d in targets]}  "
          f"共 {total_files} 块  已入库 {len(done)} 块(跳过)")
    for d in targets:
        band = _band_of(d, root)
        files = sorted(glob.glob(os.path.join(d, "*.lib.json.gz")))
        for p in files:
            seen += 1
            if p in done:
                continue
            try:
                n, dep = convert_one(con, p, band)
                total_nodes += n
            except Exception as e:
                print(f"[转换][跳过] {os.path.basename(p)} 出错: {type(e).__name__}: {e}")
                continue
            if seen % 50 == 0 or seen == total_files:
                el = time.time() - t0
                rate = (seen - len(done)) / max(el, 1e-9)
                eta = (total_files - seen) / max(rate, 1e-9)
                print(f"[转换] {seen}/{total_files} 块  band={band}  累计节点 {total_nodes}  "
                      f"用时 {el:.0f}s  剩~{eta/60:.1f}分", flush=True)
    con.commit()
    # 存一份总节点数到 meta,免得对战启动时对 2000万+ 行做 COUNT(*) 全表扫描(慢)
    n_nodes = con.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    con.execute("CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT)")
    con.execute("INSERT OR REPLACE INTO meta VALUES('n_nodes', ?)", (str(n_nodes),))
    con.commit()
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)")     # WAL 并回主库 → 产出单个干净 .db(只读打开无残留)
    bands_info = dict(con.execute('SELECT name,depth_bb FROM bands'))
    con.close()
    sz = os.path.getsize(db_path) / 1e9
    print(f"[转换] 完成。DB={db_path}  {sz:.2f}GB  节点 {n_nodes}  用时 {time.time()-t0:.0f}s  档: {bands_info}")


def _selftest():
    import tempfile
    import random
    sys.path.insert(0, os.path.dirname(__file__))
    import gto_lib_db
    # 合成 2 档、各 1 块库,节点带多步动作线
    root = tempfile.mkdtemp()
    combos = ["AhKh", "KhAh", "QsQd", "7c2d"]
    acts = ["CHECK", "BET 3"]
    for band, dep in (("mid", 130), ("deep", 250)):
        d = os.path.join(root, band)
        os.makedirs(d)
        tree = {"node_type": "action_node", "player": 1,
                "strategy": {"actions": acts,
                             "strategy": {c: [round(random.random(), 3), 0] for c in combos}},
                "childrens": {"CHECK": {"node_type": "chance_node", "dealcards": {
                    "Td": {"node_type": "action_node", "player": 0,
                           "strategy": {"actions": acts, "strategy": {c: [0.5, 0.5] for c in combos}},
                           "childrens": {"CHECK": {"node_type": "showdown_node"},
                                         "BET 3": {"node_type": "terminal_node"}}}}},
                              "BET 3": {"node_type": "terminal_node"}}}
        js = os.path.join(d, "b0.json")
        json.dump(tree, open(js, "w"))
        import solver_parse as sp
        sp.dump_lib_one(js, {"id": "b0", "board": "Qs,Jh,2h", "pot": 5,
                             "eff": dep / 2, "depth_bb": dep}, d)
    db_path = os.path.join(root, "t.db")
    convert(root, db_path)
    # 对拍:同一 board/history/hole,DB 导航结果 == 内存库导航结果
    db = gto_lib_db.LibDB(db_path)
    import gto_opponent as go
    mi = bm.build_multi_index(root)
    import random as _r
    n_ok = 0
    for _ in range(50):
        rng1 = _r.Random(7)
        rng2 = _r.Random(7)
        # 内存库路径
        idx = mi.select(130)
        m = bm.match(idx, "Qs,Jh,2h,Td", "AhKh")
        assert m is not None
        solved, rboard, rhole = m
        path = idx[bm.canon_sig(["Qs", "Jh", "2h"])][1]
        lib = gto_lib.load_lib(path)
        hist = {1: [(1, 1), (0, 1)], 2: []}
        b_mem, r_mem = go.flopturn_action(lib, rboard, rhole, hist, 2, None, False, rng1)
        # DB 路径(同一 getter 语义)
        b_db, r_db = db.nav("mid", "Qs,Jh,2h,Td", "AhKh", hist, 2, None, False, rng2)
        assert (b_mem, r_mem) == (b_db, r_db), ((b_mem, r_mem), (b_db, r_db))
        n_ok += 1
    print(f"[selftest] OK —— DB 与内存库逐次对拍一致({n_ok} 次);"
          f"节点表 {db.count()} 行,档 {db.band_names()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", help="库根目录(含 short/mid/deep/… 子目录)或单个档目录")
    ap.add_argument("db", nargs="?", help="输出 SQLite 路径,如 gto_lib.db")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest(); return
    if not (args.root and args.db):
        ap.error("用法: lib_to_sqlite.py <库目录> <输出.db>  或  --selftest")
    convert(args.root, args.db)


if __name__ == "__main__":
    main()
