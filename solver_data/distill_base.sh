#!/usr/bin/env bash
# ============================================================
# 用【已灌的 GTO 库】蒸馏一个新底座(零重解):
#   ① lib_to_bc   库(.lib.json.gz)→ BC 行(与原始树逐行一致,已 diff 验证)
#   ② solver_etl  BC 行 → 软目标 npz(强牌加权 + 被动再平衡)
#   ③ merge_npz   混入翻前软数据(防精修时翻前漂移)
#   ④ preflop_bc  从 bigflop 热启动 + 小 lr 少轮精修 → 新底座
# 在 alphaholdem-python 根目录跑:bash solver_data/distill_base.sh
# 可调(env):GTO_LIBDIR / PREFLOP_NPZ / INIT / OUT / STRONG_WEIGHT / PREFLOP_W / EPOCHS / LR / DEPTHS
# ============================================================
set -eu
cd "$(dirname "$0")/.."                                    # → alphaholdem-python 根

GTO_LIBDIR="${GTO_LIBDIR:-gto_libdir}"
PREFLOP_NPZ="${PREFLOP_NPZ:-preflop_ds_n500.npz}"          # 训 bigflop 用的翻前软数据
INIT="${INIT:-solver_data/bc_actor_bigflop.pt}"           # 热启动底座
OUT="${OUT:-solver_data/bc_actor_libdistill.pt}"
STRONG_WEIGHT="${STRONG_WEIGHT:-6}"
PREFLOP_W="${PREFLOP_W:-3}"                                # merge 时翻前:库 的权重比(3:1,防翻前被淹)
BC_SAMPLE="${BC_SAMPLE:-0.005}"                            # 【必须抽样】库全量=几十亿行会OOM;0.005→控在几百万行
EPOCHS="${EPOCHS:-6}"
LR="${LR:-3e-4}"
DEPTHS="${DEPTHS:-}"                                       # 只蒸馏某些深度带(如 short,mid),空=全部

BC_JSONL="solver_data/solver_lib_bc.jsonl"
LIB_NPZ="solver_lib_bc.npz"
MIX_NPZ="base_mix.npz"

MAXROWS="${MAXROWS:-6000000}"                             # solver_etl 全量入内存的上限(24GB 机约 600 万行)
if [ -n "${REUSE_JSONL:-}" ] && [ -s "$BC_JSONL" ]; then
  echo "[$(date '+%H:%M:%S')] ① 复用已存在的 $BC_JSONL(REUSE_JSONL=1,跳过重读库)"
else
  echo "[$(date '+%H:%M:%S')] ① 库 → BC 行 (lib=$GTO_LIBDIR depths='${DEPTHS:-全部}' sample=$BC_SAMPLE)"
  python3 solver_data/lib_to_bc.py --lib-dir "$GTO_LIBDIR" --out "$BC_JSONL" --sample "$BC_SAMPLE" ${DEPTHS:+--depths "$DEPTHS"}
fi
bc_rows=$(wc -l < "$BC_JSONL" | tr -d ' ')
echo "    BC 行数=$bc_rows"
# 行数超上限→不中止,在 ETL 阶段【自动抽稀】(jsonl 留磁盘,只是入内存的少)。
# ×0.7 余量:solver_etl 的强牌样本【豁免抽样、全保留】,会额外占量,留头别顶到上限。
etl_sample="1.0"
if [ "$bc_rows" -gt "$MAXROWS" ]; then
  etl_sample=$(python3 -c "print(round(0.7*$MAXROWS/$bc_rows,4))")
  echo "    行数超 $MAXROWS → ETL 自动抽稀 --sample $etl_sample(留 30% 余量给豁免抽样的强牌)"
fi

echo "[$(date '+%H:%M:%S')] ② BC 行 → 软 npz (强牌加权×$STRONG_WEIGHT, etl-sample=$etl_sample)"
python3 solver_data/solver_etl.py --jsonl "$BC_JSONL" --out "$LIB_NPZ" --norm 500 \
    --passive-keep 0.3 --strong-weight "$STRONG_WEIGHT" --sample "$etl_sample"

if [ -f "$PREFLOP_NPZ" ]; then
  echo "[$(date '+%H:%M:%S')] ③ 混入翻前 $PREFLOP_NPZ (权重 翻前:库 = $PREFLOP_W:1)"
  python3 merge_npz.py --in "$PREFLOP_NPZ" "$LIB_NPZ" --weights "${PREFLOP_W},1" --out "$MIX_NPZ"
  TRAIN_NPZ="$MIX_NPZ"
else
  echo "[$(date '+%H:%M:%S')] ③ 未找到 $PREFLOP_NPZ → 跳过混合(仅用库数据;翻前靠热启动权重+对打护栏)"
  TRAIN_NPZ="$LIB_NPZ"
fi

echo "[$(date '+%H:%M:%S')] ④ 精修底座:热启动 $INIT → $OUT (epochs=$EPOCHS lr=$LR)"
python3 preflop_bc.py --preflop "$TRAIN_NPZ" --init "$INIT" --out "$OUT" --epochs "$EPOCHS" --lr "$LR"

echo "[$(date '+%H:%M:%S')] ✅ 完成 → $OUT"
echo "  验证(单看底座提升,先不叠 river-solve/flopturn-lib):"
echo "  python3 play_vs_bot.py --depth mid --model $OUT --preflop-guard --auto --opp bcpoker --hands 100000 --chips 100,100 --diag --big 0.3 --log bighand-distill.log"
