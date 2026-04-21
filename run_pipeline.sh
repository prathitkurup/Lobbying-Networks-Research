#!/usr/bin/env bash
# run_pipeline.sh — Full pipeline runner for Lobbying Networks Research
# Run from the project root: bash run_pipeline.sh
# Logs per-script pass/fail; continues on failure.
# V13 (BCZ intercentrality) is excluded — times out; run manually on full machine.
# V17 (quarterly dynamics) is excluded from default run — see comment below.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
SRC="$REPO_ROOT/src"
LOG_DIR="$REPO_ROOT/outputs/run_logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SUMMARY_LOG="$LOG_DIR/pipeline_run_$TIMESTAMP.log"

PASS=0
FAIL=0
SKIPPED=0
FAILED_SCRIPTS=()

run_script() {
    local label="$1"
    local script="$2"
    local log="$LOG_DIR/${label}_$TIMESTAMP.log"

    echo -n "  [$label] ... "
    if python "$script" > "$log" 2>&1; then
        echo "PASS"
        ((PASS++))
        echo "PASS: $label" >> "$SUMMARY_LOG"
    else
        echo "FAIL  (see $log)"
        ((FAIL++))
        FAILED_SCRIPTS+=("$label")
        echo "FAIL: $label  →  $log" >> "$SUMMARY_LOG"
    fi
}

hr() { printf '%.0s─' {1..70}; echo; }

echo "" | tee -a "$SUMMARY_LOG"
hr
echo "  Lobbying Networks Pipeline  |  $(date)" | tee -a "$SUMMARY_LOG"
echo "  Logs → $LOG_DIR" | tee -a "$SUMMARY_LOG"
hr

# ── Phase 1: Core extraction and 116th Congress network ──────────────────────
echo ""
echo "Phase 1: Core extraction + directed influence network"
hr

cd "$SRC"
# run_script "01_extraction"         "opensecrets_extraction.py"
run_script "02_rbo_directed"       "rbo_directed_influence.py"
run_script "03_mediated_adoption"  "affiliation_mediated_adoption.py"
run_script "04_viz_mediation"      "visualize_affiliation_mediation.py"
run_script "05_enrich_gml"         "enrich_directed_gml.py"
run_script "06_gephi_export"       "gephi_style_export.py"

# ── Phase 2: Multi-congress pipeline (111th–117th) ───────────────────────────
echo ""
echo "Phase 2: Multi-congress pipeline (111th–117th, ~10–20 min)"
hr

# run_script "07_multi_congress"     "multi_congress_pipeline.py"
run_script "08_xc_stability"       "cross_congressional_stability.py"

# ── Phase 3: Validations ─────────────────────────────────────────────────────
echo ""
echo "Phase 3: Validation scripts"
hr

run_script "V01_extraction_audit"          "validations/01_extraction_audit.py"
run_script "V02_inflation_diagnosis"       "validations/02_inflation_diagnosis.py"
run_script "V03_sparsity_analysis"         "validations/03_sparsity_analysis.py"
run_script "V04_mega_bill_diagnosis"       "validations/04_mega_bill_diagnosis.py"
run_script "V05_ind_filter"                "validations/05_ind_filter_validation.py"
run_script "V06_rbo_cosine_unit_tests"     "validations/06_rbo_cosine_unit_tests.py"
# V07 excluded — imports composite_similarity_network from src/archive/networks/ (archived dependency)
# run_script "V07_composite_validation"      "validations/07_composite_network_validation.py"
run_script "V08_rbo_p_calibration"         "validations/08_rbo_p_calibration.py"
run_script "V10_directed_validation"       "validations/10_rbo_directed_influence_validation.py"
run_script "V11_mediated_adoption"         "validations/11_mediated_adoption_validation.py"
run_script "V12_congress_statistics"       "validations/12_congress_statistics.py"

# V13 excluded — BCZ intercentrality (277 node-removal Katz reruns) times out.
# Run manually: cd src && python validations/13_centrality_vs_agenda_setter.py
echo "  [V13_centrality_bcz] ... SKIPPED (BCZ intercentrality times out; run manually)"
((SKIPPED++))
echo "SKIPPED: V13_centrality_bcz  →  run manually on full machine" >> "$SUMMARY_LOG"

run_script "V14_influencer_regression"     "validations/14_influencer_regression.py"
run_script "V15_cross_sector"              "validations/15_cross_sector_directed_edges.py"
run_script "V16_industry_hierarchy"        "validations/16_industry_influencer_hierarchy.py"

# V17 (quarterly dynamics) not part of primary pipeline; uncomment to include:
# run_script "V17_quarterly_dynamics"      "validations/17_quarterly_dynamics.py"

run_script "V18_payoff_complementarity"    "validations/18_payoff_complementarity.py"
run_script "V19_bill_adoption_diffusion"   "validations/19_bill_adoption_diffusion.py"
run_script "V20_stability_complementarity" "validations/20_influence_stability_complementarity.py"

# ── Phase 4: Publication figures ──────────────────────────────────────────────
echo ""
echo "Phase 4: Publication figures"
hr

run_script "gen_figs_13_19"   "validations/gen_figs_13_19.py"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
hr
echo "  Results: $PASS passed  |  $FAIL failed  |  $SKIPPED skipped"
if [ ${#FAILED_SCRIPTS[@]} -gt 0 ]; then
    echo "  Failed:"
    for s in "${FAILED_SCRIPTS[@]}"; do
        echo "    ✗ $s"
    done
fi
echo "  Full summary → $SUMMARY_LOG"
hr
echo ""
