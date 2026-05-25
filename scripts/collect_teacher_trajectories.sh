#!/bin/bash
# Generate Qwen3-30B teacher trajectories on SWE-Gym training set in TWO stages.
#
# STAGE 1 — any-match: retry until each instance has ≥1 lenient gold match
#          at all three levels (file/module/entity). Catches cases where the
#          model omitted module/function output entirely.
# STAGE 2 — half-recall: among Stage-1-passing instances, retry instances whose
#          per-instance recall is below 0.5 at any level. Stops when each
#          level's mean recall reaches 0.5 OR no progress OR max iter.
#
# Lenient matching follows progress_labeler._loc_matches (exact / dot-prefix /
# dot-suffix on the qualified-name part within the same file).
#
# Prereq: serve the teacher model with vLLM (OpenAI-compatible endpoint)
# Usage:
#   nohup bash scripts/collect_teacher_trajectories.sh > logs/teacher.log 2>&1 &

# Don't use set -e: child timeouts/stalls are expected and handled.

# Set OPENAI_API_BASE / OPENAI_API_KEY to your teacher vLLM endpoint
export PYTHONPATH=$PYTHONPATH:$(pwd)
export GRAPH_INDEX_DIR='index_data/SWE-Gym/graph_index_v2.3'
export BM25_INDEX_DIR='index_data/SWE-Gym/BM25_index'
export SUMMARY_INDEX_DIR='index_data/SWE-Gym/file_summaries'

DATASET='SWE-Gym/SWE-Gym'
SPLIT='train'
MODEL='openai/qwen3-30b'
NUM_PROCESSES=64
TIMEOUT=300
EVAL_N_LIMIT=2438
STALL_TIMEOUT=2400
GT_FILE='evaluation/gt_location/SWE-Gym/train/gt_location.jsonl'

OUTPUT_DIR='./toolplan_data/trajectories_swegym_30b/location'
OUTPUT_FILE="${OUTPUT_DIR}/loc_outputs.jsonl"
TRAJ_FILE="${OUTPUT_DIR}/loc_trajs.jsonl"
mkdir -p "${OUTPUT_DIR}" logs

STAGE1_MAX_ITERS=5
STAGE2_MAX_ITERS=8
STAGE2_RECALL_TARGET=0.4

# ── Filter helper ────────────────────────────────────────────────────────────
# Args:
#   $1 = stage  ("any" | "half")
# Removes failing entries from BOTH loc_outputs.jsonl and loc_trajs.jsonl
# so auto_search_main re-runs them.
# Prints "passing failing mean_recall_F mean_recall_M mean_recall_U"
filter_entries() {
    local STAGE=$1
    "${PYTHON_BIN:-python}" -c "
import json, os
OUTPUT_FILE = '${OUTPUT_FILE}'
TRAJ_FILE = '${TRAJ_FILE}'
GT_FILE = '${GT_FILE}'
STAGE = '${STAGE}'
TARGET = float('${STAGE2_RECALL_TARGET}')

if not os.path.exists(OUTPUT_FILE):
    print('0 0 0.000 0.000 0.000'); exit()

gold = {}
with open(GT_FILE) as f:
    for line in f:
        d = json.loads(line)
        files, mods, ents = set(), set(), set()
        for fc in d.get('file_changes') or []:
            if fc.get('file'): files.add(fc['file'])
            chg = fc.get('changes', {}) or {}
            mods |= set(chg.get('edited_modules', []) or [])
            ents |= set(chg.get('edited_entities', []) or [])
        gold[d['instance_id']] = (files, mods, ents)

def name_matches(a, b):
    if a == b: return True
    if a.startswith(b + '.') or b.startswith(a + '.'): return True
    if a.endswith('.' + b) or b.endswith('.' + a): return True
    return False

def loc_matches(p, g):
    if p == g: return True
    pf, _, pn = p.partition(':')
    gf, _, gn = g.partition(':')
    if pf != gf: return False
    if not pn or not gn: return False
    return name_matches(pn, gn)

def lenient_recall(preds, golds):
    # |gold matched by ≥1 pred| / |gold|
    if not golds: return 1.0
    matched = 0
    for g in golds:
        for p in preds:
            if loc_matches(p, g): matched += 1; break
    return matched / len(golds)

passing_lines, passing_ids = [], set()
failing_ids = set()
sum_rf = sum_rm = sum_re = 0.0
n_evaluated = 0
with open(OUTPUT_FILE) as f:
    for line in f:
        if not line.strip(): continue
        d = json.loads(line)
        iid = d['instance_id']
        if iid not in gold:
            passing_lines.append(line); passing_ids.add(iid); continue
        gf, gm, ge = gold[iid]
        ff = set((d.get('found_files', [[]]) or [[]])[0] or [])
        fm = set((d.get('found_modules', [[]]) or [[]])[0] or [])
        fe = set((d.get('found_entities', [[]]) or [[]])[0] or [])
        rf = (len(ff & gf) / len(gf)) if gf else 1.0
        rm = lenient_recall(fm, gm)
        re = lenient_recall(fe, ge)
        if STAGE == 'any':
            ok = (rf > 0) and (rm > 0) and (re > 0)
        else:  # 'half'
            ok = (rf >= TARGET) and (rm >= TARGET) and (re >= TARGET)
        sum_rf += rf; sum_rm += rm; sum_re += re; n_evaluated += 1
        if ok:
            passing_lines.append(line); passing_ids.add(iid)
        else:
            failing_ids.add(iid)

with open(OUTPUT_FILE, 'w') as f:
    for line in passing_lines:
        f.write(line)
if os.path.exists(TRAJ_FILE):
    keep = []
    with open(TRAJ_FILE) as f:
        for line in f:
            if not line.strip(): continue
            d = json.loads(line)
            if d['instance_id'] in passing_ids:
                keep.append(line)
    with open(TRAJ_FILE, 'w') as f:
        for line in keep:
            f.write(line)

n = max(n_evaluated, 1)
print(f'{len(passing_ids)} {len(failing_ids)} {sum_rf/n:.3f} {sum_rm/n:.3f} {sum_re/n:.3f}')
" 2>/dev/null
}

# ── Run one full pass through the dataset ────────────────────────────────────
run_one_pass() {
    while true; do
        local current
        current=$(wc -l < "$OUTPUT_FILE" 2>/dev/null || echo 0)
        if [ "$current" -ge "$EVAL_N_LIMIT" ]; then
            echo "Already at ${current}/${EVAL_N_LIMIT}, skip pass."
            break
        fi
        echo "Starting pass (progress: ${current}/${EVAL_N_LIMIT})"
        "${PYTHON_BIN:-python}" auto_search_main.py \
            --dataset "${DATASET}" \
            --split "${SPLIT}" \
            --model "${MODEL}" \
            --localize \
            --merge \
            --output_folder "${OUTPUT_DIR}" \
            --eval_n_limit "${EVAL_N_LIMIT}" \
            --num_samples 1 \
            --num_processes "${NUM_PROCESSES}" \
            --timeout "${TIMEOUT}" \
            --use_function_calling \
            --native_tool_calling \
            --enable_commit_search \
            --enable_file_summary \
            --exclude_tools examine_commit &
        local main_pid=$!
        local last_mod
        last_mod=$(stat -c %Y "$OUTPUT_FILE" 2>/dev/null || echo 0)
        while kill -0 $main_pid 2>/dev/null; do
            sleep 60
            local now_mod
            local log_mod
            now_mod=$(stat -c %Y "$OUTPUT_FILE" 2>/dev/null || echo 0)
            log_mod=$(stat -c %Y "${OUTPUT_DIR}/localize.log" 2>/dev/null || echo 0)
            local now=$(date +%s)
            local output_age=$((now - now_mod))
            local log_age=$((now - log_mod))
            if [ "$output_age" -gt "$STALL_TIMEOUT" ] && [ "$log_age" -gt "$STALL_TIMEOUT" ]; then
                echo "=== $(date) === STALL detected. Restarting..."
                kill -9 -$main_pid 2>/dev/null
                pkill -9 -P $main_pid 2>/dev/null
                pkill -9 -f "multiprocessing.spawn" 2>/dev/null
                sleep 5; break
            fi
            if [ "$now_mod" -ne "$last_mod" ]; then
                last_mod=$now_mod
                local new_count
                new_count=$(wc -l < "$OUTPUT_FILE" 2>/dev/null || echo 0)
                echo "=== $(date) === ${new_count}/${EVAL_N_LIMIT} ==="
                if [ "$new_count" -ge "$EVAL_N_LIMIT" ]; then
                    kill $main_pid 2>/dev/null; break
                fi
            fi
        done
        wait $main_pid 2>/dev/null
        sleep 5
    done
}

# ────────────────────────────────────────────────────────────────────────────
echo "================================================================"
echo "  Qwen3-30B teacher generation on SWE-Gym (two-stage retry)"
echo "  Output: ${OUTPUT_FILE}"
echo "  Stage 1: any-match (each level ≥1 gold match)"
echo "  Stage 2: half-recall (each level mean recall ≥ ${STAGE2_RECALL_TARGET})"
echo "  $(date)"
echo "================================================================"

# Pass 1: full sweep
echo ""
echo "=== ITER 0 (initial full sweep) ==="
run_one_pass

# ── STAGE 1: any-match retry ─────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  STAGE 1: any-match retry (max ${STAGE1_MAX_ITERS} iters)"
echo "================================================================"
PREV_FAIL=999999
for ITER in $(seq 1 $STAGE1_MAX_ITERS); do
    STATS=$(filter_entries any)
    PASS=$(echo "$STATS" | awk '{print $1}')
    FAIL=$(echo "$STATS" | awk '{print $2}')
    RF=$(echo "$STATS" | awk '{print $3}')
    RM=$(echo "$STATS" | awk '{print $4}')
    RE=$(echo "$STATS" | awk '{print $5}')
    echo ""
    echo "=== STAGE1 iter ${ITER}: pass=${PASS}  fail=${FAIL}  recall(F/M/U)=${RF}/${RM}/${RE} ==="
    if [ "${FAIL:-0}" -eq 0 ]; then
        echo "Stage 1 complete: every instance has ≥1 match at all 3 levels."
        break
    fi
    if [ "${FAIL}" -ge "${PREV_FAIL}" ]; then
        echo "Stage 1: no progress, giving up at iter ${ITER}."
        break
    fi
    PREV_FAIL=$FAIL
    run_one_pass
done

# ── STAGE 2: SKIPPED per user request (decide after seeing Stage 1 result) ──
echo ""
echo "================================================================"
echo "  STAGE 2 SKIPPED. Stage 1 done — review metrics and decide next."
echo "  $(date)"
echo "================================================================"
exit 0

# ── STAGE 2: half-recall retry ───────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  STAGE 2: half-recall retry (target ≥ ${STAGE2_RECALL_TARGET}, max ${STAGE2_MAX_ITERS} iters)"
echo "================================================================"
PREV_FAIL=999999
for ITER in $(seq 1 $STAGE2_MAX_ITERS); do
    STATS=$(filter_entries half)
    PASS=$(echo "$STATS" | awk '{print $1}')
    FAIL=$(echo "$STATS" | awk '{print $2}')
    RF=$(echo "$STATS" | awk '{print $3}')
    RM=$(echo "$STATS" | awk '{print $4}')
    RE=$(echo "$STATS" | awk '{print $5}')
    echo ""
    echo "=== STAGE2 iter ${ITER}: pass=${PASS}  fail=${FAIL}  recall(F/M/U)=${RF}/${RM}/${RE} ==="
    if [ "${FAIL:-0}" -eq 0 ]; then
        echo "Stage 2 complete: every instance hits ${STAGE2_RECALL_TARGET}+ recall at all 3 levels."
        break
    fi
    BELOW=$("${PYTHON_BIN:-python}" -c "
v = [float(x) for x in '${RF} ${RM} ${RE}'.split()]
print('1' if min(v) < ${STAGE2_RECALL_TARGET} else '0')
" 2>/dev/null)
    if [ "${BELOW}" = "0" ]; then
        echo "Stage 2 mean recall reached target on all levels. Done."
        break
    fi
    if [ "${FAIL}" -ge "${PREV_FAIL}" ]; then
        echo "Stage 2: no progress, giving up at iter ${ITER}."
        break
    fi
    PREV_FAIL=$FAIL
    run_one_pass
done

# Final stats
FINAL=$(filter_entries half)
echo ""
echo "================================================================"
echo "  FINAL: pass=$(echo $FINAL | awk '{print $1}')  fail=$(echo $FINAL | awk '{print $2}')"
echo "         mean recall (F/M/U) = $(echo $FINAL | awk '{print $3,$4,$5}')"
echo "  $(date)"
echo "================================================================"
