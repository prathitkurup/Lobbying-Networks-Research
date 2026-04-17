## Directed Influence Network — Full Summary

### Concept
Influence is operationalized as agenda-setting: if Firm A consistently lobbies a bill before Firm B across their shared bill portfolio, A is the agenda-setter and B is the follower. Priority rankings are defined by spend fractions (amount_allocated / total_budget per firm per bill). The proximate mechanism is lobbyist networks but that is out of scope — the network captures the observable outcome.

### What Was Built
Two new scripts: `src/directed_influence_network.py` and `src/utils/visualization.py` (extended).

### Core Logic (directed_influence_network.py)

Input: `opensecrets_lda_reports.csv` (22,860 rows, 305 firms, 2219 bills). Quarter assignment: `base_q = int(report_type[1])`, year_offset = 0 (2019) or 4 (2020), quarter = base_q + year_offset → values 1–8.

Pipeline per quarter q:
1. Filter to quarter q rows only (independent windows, not cumulative)
2. Apply MAX_BILL_DF = 50 prevalence filter (remove mega-bills)
3. Aggregate per (firm, bill) by summing amount_allocated
4. Compute fracs; exclude zero-budget firms
5. Build ranked bill lists (top 30 by spend)
6. Load all RBO pairs from `rbo_edges_q{q}.csv` (pre-computed undirected RBO network)
7. For each RBO pair (A, B): look up `bill_first = {(firm, bill): first_quarter}` using causal window Q1..q only
8. Score over shared top-30 bills: A gets +1 if first_q(A,bill) < first_q(B,bill), B gets +1 if opposite, tie if equal
9. If A_firsts > B_firsts: emit A→B with weight = A_firsts − B_firsts; if equal (all ties or balanced): no edge
10. Build DiGraph, tag nodes with out_strength, in_strength, net_influence = out − in, community
11. Write edges CSV, GML, PNG

Aggregate: Sum source_firsts and target_firsts per canonical pair (min(A,B), max(A,B)) across all 8 quarters, apply net-direction rule to totals.

### Key Design Decisions
- Causal window Q1..q only — no future information leakage (temporal exogeneity)
- Edge weight = source_firsts − target_firsts (net first-mover margin, not RBO weight)
- RBO weight stored separately as `rbo_weight` edge attribute — not incorporated into directed weight
- Ties skipped — simultaneous bill adoption is exogenous co-response, not influence signal
- Q1 always empty — all first_quarters = 1 in Q1 by construction, so every pair ties
- Independent quarterly windows — each quarter uses only that quarter's filings; preserves COVID structural shock detection at Q4→Q5
- Node color in PNG: green = net agenda-setter (out > in), red = net follower (in > out), gray = balanced

### Visualization (utils/visualization.py)
`plot_directed_circular(G, title, path, top_k=20)` — circular layout, top-k nodes by total involvement (out_strength + in_strength), curved edges (arc3,rad=0.15) to prevent A→B and B→A arrows from overlapping, arrow width ∝ edge weight, legend via mpatches. Originally local to `directed_influence_network.py`, moved to `utils/visualization.py`; numpy and matplotlib.pyplot imports removed from main script.

### Empirical Results
- Q1: 0 edges (expected)
- Q2: 303 edges / 2,091 RBO pairs (14.5% decisive), mean net weight 1.39
- Q3: 854 / 2,187 (39.0%)
- Q4: 1,287 / 2,801 (45.9%)
- Q5–Q8: 613–1,039 edges, 40–48% decisive
- Aggregate: 2,822 edges / 6,265 RBO pairs (45.0%), mean net weight 2.50
- Top aggregate agenda-setters: BALL (+132), NORTHROP GRUMMAN (+131), FORD (+112), CUMMINS (+103), BOEING (+99), EXXONMOBIL (+97)
- Top aggregate followers: STATE FARM (−149), ENTERGY (−143), AEP (−117), PPL (−122), AMAZON (−93)

### Validation (src/validations/09_directed_influence_validation.py)
8 checks, all pass:
1. Quarter values exactly 1–8; 2019→Q1-4, 2020→Q5-8
2. Q1 CSV absent (no decisive pairs)
3. weight == source_firsts − target_firsts for all edges Q2–Q8
4. source_firsts > target_firsts for all directed edges
5. rbo_weight > 0 on all edges
6. Coverage rises Q2 (14.5%) → Q8 (48.1%)
7. Aggregate first-mover totals ≥ per-quarter totals for all 5,539 checked pairs
8. Aggregate schema complete (2,822 rows, all required columns including quarters_active)

### Output Files

| File | Contents |
|---|---|
| data/directed_influence_q{2..8}.csv | Per-quarter directed edges (source, target, weight, rbo_weight, source_firsts, target_firsts, tie_count, shared_bills) |
| data/directed_influence_agg.csv | Aggregate directed edges (same columns + quarters_active) |
| visualizations/gml/directed_influence_q{1..8}.gml | Per-quarter directed GMLs for Gephi |
| visualizations/gml/directed_influence_agg.gml | Aggregate GML |
| visualizations/png/directed_influence_q{1..8}.png | Directed circular plots (Q1 skipped, empty) |
| visualizations/png/directed_influence_agg.png | Aggregate circular plot |

### Documentation
- design_decisions.md §0 — conceptual framework (influence = agenda-setting)
- design_decisions.md §20 — full directed influence methodology, scoring rule, design choices, empirical highlights, references (Carpenter et al. 1998; Granger 1969; LaPira & Thomas 2014; Mian et al. 2010)
- README.md — "Conceptual Framework" section at top; §0 in Key Design Decisions list; both new scripts listed with run instructions
