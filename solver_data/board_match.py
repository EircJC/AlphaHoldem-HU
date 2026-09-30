"""匹配层:真实牌面 → 库里已解板 + 花色置换,并用同一置换重映射底牌/转河。

为什么要:GTO 库只解了有限翻牌(每块是花色同构类的一个代表)。真实一手的翻牌若与某已解板
【花色同构】,就能查它的策略——但必须用【同一套花色置换】把底牌、转牌、河牌一起重映射,
否则会查到错的 combo(阻断/听花全乱)。GTO 策略对花色对称,所以任一合法置换给的策略都对。

实现:
- canon_sig(flop):枚举 24 种花色双射取字典序最小 → 花色无关的规范签名。同构翻牌签名相同,用作索引 key。
- flop_perm(real_flop, solved_flop):枚举 24 种双射,找把 real_flop 精确映成 solved_flop 的那个(全4花色双射)。
- match(index, real_board, hole):真实翻牌签名 → 查到已解板 → 求置换 → 重映射整板+底牌 → 返回可查库的形式。

  python3 board_match.py --selftest
"""
import argparse
from itertools import permutations

RANKC = "23456789TJQKA"
SUITS = "shdc"


def _parse(c):
    return RANKC.index(c[0].upper()), c[1].lower()


def _fmt(rk, su):
    return RANKC[rk] + su


def _cards(board):
    if isinstance(board, str):
        board = [x.strip() for x in board.split(",") if x.strip()]
    return [_parse(c) for c in board]


def canon_sig(flop):
    """花色无关的规范签名(同构翻牌相同)。枚举 24 花色双射,取映射后排序的字典序最小。"""
    cs = _cards(flop)
    best = None
    for perm in permutations(SUITS):
        pi = dict(zip(SUITS, perm))
        mapped = tuple(sorted((rk, pi[su]) for rk, su in cs))
        if best is None or mapped < best:
            best = mapped
    return best


def flop_perm(real_flop, solved_flop):
    """找把 real_flop 精确映成 solved_flop 的花色双射(全 4 花色);不同构返回 None。"""
    rr = _cards(real_flop)
    target = sorted(_cards(solved_flop))
    if sorted(rk for rk, _ in rr) != sorted(rk for rk, _ in target):
        return None
    for perm in permutations(SUITS):
        pi = dict(zip(SUITS, perm))
        if sorted((rk, pi[su]) for rk, su in rr) == target:
            return pi
    return None


def remap(board_or_cards, perm):
    """用花色置换重映射一组牌(保持点数,换花色)。返回逗号分隔字符串。"""
    return ",".join(_fmt(rk, perm[su]) for rk, su in _cards(board_or_cards))


def build_index(lib_dir):
    """扫 lib_dir 下所有 .lib.json.gz,建 {canon_sig: (solved_flop_str, lib_path)}。"""
    import glob
    import os
    import gto_lib
    idx = {}
    files = sorted(glob.glob(os.path.join(lib_dir, "*.lib.json.gz")))
    for i, p in enumerate(files):
        lib = gto_lib.load_lib(p)
        flop = lib["meta"]["flop"]
        idx[canon_sig(flop)] = (flop, p)
        if (i + 1) % 500 == 0:
            print(f"[库索引] 读取 {os.path.basename(lib_dir)}: {i + 1}/{len(files)} ...", flush=True)
    return idx


_NAME_BB = {"short": 40.0, "mid": 130.0, "deep": 250.0, "vdeep": 400.0}


class MultiDepthIndex:
    """多深度库索引:按有效筹码就近选深度带。bands=[(depth_bb, {sig:(flop,path)}, dir)]。"""

    def __init__(self, bands):
        self.bands = sorted(bands, key=lambda b: b[0])

    def __len__(self):
        return sum(len(b[1]) for b in self.bands)

    def select(self, eff_bb, lo_ratio=0.5, hi_ratio=1.7):
        """有效筹码 → 最近深度带的索引(canon_sig→(flop,path));无带或离最近带太远则空(→回退NN)。
        深度容差与 gto_lib_db.select_band 一致:每带策略是该 SPR 下解出的,拿 250bb 套 1000bb 是错的,
        只有 eff 落在最近带 [lo_ratio,hi_ratio]×depth 内才用,否则空(现有 40/130/250 → 覆盖约 20~425bb)。"""
        if not self.bands:
            return {}
        depth, idx = min(self.bands, key=lambda b: abs(b[0] - eff_bb))[0:2]
        if depth <= 0 or eff_bb / depth < lo_ratio or eff_bb / depth > hi_ratio:
            return {}
        return idx

    def summary(self):
        return {int(round(b[0])): len(b[1]) for b in self.bands}


def build_multi_index(root):
    """root 下若有深度子目录(short/mid/deep,各含 .lib.json.gz)→ 多深度索引;
    若 root 直接是扁平库 → 单档(向后兼容)。每档代表深度取库 meta.depth_bb,拿不到按目录名猜。
    带缓存:索引(sig→flop/path)按 文件数+最新mtime 缓存到 root/.mindex_cache.pkl,
    库没变则秒加载(否则每次要解压上千个 .lib.json.gz,启动要几分钟)。"""
    import glob
    import os
    import pickle
    import time
    import gto_lib
    t0 = time.time()
    all_files = glob.glob(os.path.join(root, "**", "*.lib.json.gz"), recursive=True)
    cache_sig = (len(all_files), round(max((os.path.getmtime(f) for f in all_files), default=0.0), 3))
    cache_path = os.path.join(root, ".mindex_cache.pkl")
    if os.path.exists(cache_path):
        try:
            c = pickle.load(open(cache_path, "rb"))
            if c.get("sig") == cache_sig:
                print(f"[库索引] 命中缓存({len(all_files)}块,{time.time()-t0:.1f}s)→ {cache_path}")
                return MultiDepthIndex(c["bands"])
        except Exception:
            pass
    print(f"[库索引] 无缓存/库已变,开始建索引({len(all_files)}块,预计每千块约1-2分钟)...", flush=True)
    subdirs = [d for d in sorted(glob.glob(os.path.join(root, "*")))
               if os.path.isdir(d) and glob.glob(os.path.join(d, "*.lib.json.gz"))]
    # 子目录 + 根目录扁平库(若两者并存,扁平的也当一档,避免未归档的库被静默漏掉)
    targets = list(subdirs)
    if glob.glob(os.path.join(root, "*.lib.json.gz")):
        targets.append(root)
    bands = []
    for d in targets:
        idx = build_index(d)
        if not idx:
            continue
        dep = None
        try:
            one = sorted(glob.glob(os.path.join(d, "*.lib.json.gz")))[0]
            dep = gto_lib.load_lib(one)["meta"].get("depth_bb")
        except Exception:
            dep = None
        if dep is None:
            dep = _NAME_BB.get(os.path.basename(d.rstrip("/")).lower(), 100.0)
        bands.append((float(dep), idx, d))
    try:
        pickle.dump({"sig": cache_sig, "bands": bands}, open(cache_path, "wb"))
        print(f"[库索引] 已建索引并缓存({len(all_files)}块,用时 {time.time()-t0:.1f}s)→ {cache_path}")
    except Exception:
        print(f"[库索引] 建索引完成({len(all_files)}块,用时 {time.time()-t0:.1f}s;缓存写入失败,下次仍需重建)")
    return MultiDepthIndex(bands)


def match(index, real_board, hole):
    """真实(board, hole)→ (solved_flop, 重映射后的整板字符串, 重映射后的底牌combo) 或 None(库未覆盖该翻牌类)。
    real_board 可为 3/4/5 张;hole=两张如 'AhKh'/'Ah,Kh'。"""
    rc = _cards(real_board)
    real_flop = rc[:3]
    sig = canon_sig([_fmt(*c) for c in real_flop])
    hit = index.get(sig)
    if hit is None:
        return None
    solved_flop, _path = hit
    perm = flop_perm([_fmt(*c) for c in real_flop], solved_flop)
    if perm is None:
        return None
    # 翻牌部分【必须用库里存的 solved_flop 顺序】(库节点 key 就是这个顺序);
    # 真实牌面顺序常与库不同,若照真实顺序拼 key 会连根节点都查不到(nav_no_node)。
    # 转/河牌按同一花色置换重映射后接在其后(库转牌节点 key = solved_flop 顺序 + 转牌)。
    extra = rc[3:]
    if extra:
        board_mapped = solved_flop + "," + remap([_fmt(*c) for c in extra], perm)
    else:
        board_mapped = solved_flop
    hole_mapped = remap(hole, perm).replace(",", "")
    return solved_flop, board_mapped, hole_mapped


def _selftest():
    # 1) 同构翻牌签名相同、非同构不同
    assert canon_sig("Qs,Jh,2h") == canon_sig("Qc,Jd,2d"), "两色同构应同签名"
    assert canon_sig("Qs,Jh,2h") == canon_sig("Qh,Js,2s")
    assert canon_sig("Qs,Js,2s") != canon_sig("Qs,Jh,2h"), "单色 vs 两色应不同"
    assert canon_sig("2h,2s,Qd") == canon_sig("2c,2d,Qh"), "对子板同构"

    # 2) flop_perm 精确映射 + 保持花色共享(听花不丢)
    perm = flop_perm("Qc,Jd,2d", "Qs,Jh,2h")
    assert perm is not None
    assert remap("Qc,Jd,2d", perm) == "Qs,Jh,2h", remap("Qc,Jd,2d", perm)
    # real 手牌 AdKd(与翻牌 Jd/2d 同花 d)→ 映射后应与 solved 的 h 同花(Jh/2h)
    hole_m = remap("Ad,Kd", perm)
    assert hole_m == "Ah,Kh", hole_m       # d→h,听同花关系保住

    # 3) 非同构返回 None
    assert flop_perm("Qs,Js,2s", "Qs,Jh,2h") is None

    # 4) match:整板 + 底牌一起重映射,转牌也跟着走
    index = {canon_sig("Qs,Jh,2h"): ("Qs,Jh,2h", "dummy.lib.json.gz")}
    #   真实一手:翻 Qc Jd 2d,转 Td(新花),底牌 Ad Kd
    r = match(index, "Qc,Jd,2d,Td", "Ad,Kd")
    assert r is not None
    solved_flop, board_m, hole_m = r
    assert solved_flop == "Qs,Jh,2h"
    # d→h;Td 的 d→h,底牌 AdKd→AhKh;整板 = Qs,Jh,2h,Th
    assert board_m == "Qs,Jh,2h,Th", board_m
    assert hole_m == "AhKh", hole_m
    # 库没覆盖的翻牌类 → None
    assert match(index, "2s,2h,2d,5c", "Ah,Kh") is None

    # 5) 阻断关系:底牌恰是翻牌一张的花色,映射后仍指向对应 solved 花色
    perm2 = flop_perm("Ac,Kc,Qc", "As,Ks,Qs")   # 单色 c→s
    assert remap("Jc,Tc", perm2) == "Js,Ts", remap("Jc,Tc", perm2)   # 同花听花保住

    # 6) 【关键回归】真实翻牌顺序 ≠ 库里顺序:返回的翻牌部分必须是【库的顺序】,否则连根节点 key 都查不到
    index3 = {canon_sig("As,Kh,Qh"): ("As,Kh,Qh", "d.gz")}
    r3 = match(index3, "Qh,Kh,As,Td", "2c,3d")   # 真实顺序 Q,K,A,转Td;库顺序 A,K,Q
    assert r3 is not None
    _sf, bm3, _hm3 = r3
    assert bm3.split(",")[:3] == ["As", "Kh", "Qh"], bm3    # 翻牌=库顺序(非真实 Q,K,A)
    assert bm3.split(",")[3] == "Td", bm3                   # 转牌接在其后(此处 perm 恒等)

    # 7) 多深度索引就近选带
    mi = MultiDepthIndex([(40.0, {"a": 1}, "short"), (130.0, {"b": 2}, "mid"), (250.0, {"c": 3}, "deep")])
    assert mi.select(30) == {"a": 1}, "30bb→short"
    assert mi.select(120) == {"b": 2}, "120bb→mid"
    assert mi.select(400) == {"c": 3}, "400bb→deep(最近)"
    assert mi.select(100) == {"b": 2}, "100bb→mid(比short近)"
    assert len(mi) == 3 and mi.summary() == {40: 1, 130: 1, 250: 1}
    print("  多深度就近: 30→short 120→mid 400→deep ✓")

    print("[selftest] OK —— 规范签名/精确置换/整板+底牌重映射/听花与阻断关系保持/未覆盖返回None/多深度就近 全部正确")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--lib-dir", help="给个 lib 目录,打印它覆盖的翻牌签名数")
    args = ap.parse_args()
    if args.selftest:
        _selftest(); return
    if args.lib_dir:
        idx = build_index(args.lib_dir)
        print(f"覆盖 {len(idx)} 个花色同构翻牌类")
        return
    ap.error("需 --selftest 或 --lib-dir")


if __name__ == "__main__":
    main()
