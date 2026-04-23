## RBO Directed Influence Network — Summary

### Concept

Influence is operationalized as agenda-setting: if Firm A consistently lobbies a bill before Firm B across their shared bill portfolio, A is the agenda-setter and B is the follower. Priority rankings are defined by spend fractions (`amount_allocated / total_budget` per firm per bill). The proximate mechanism is shared lobbyists — lobbyists working for both firms transmit bill priorities from the influencer to the follower. The network captures the observable outcome of that mechanism.

---

### Construction (`src/rbo_directed_influence.py`)

**Input:** `data/opensecrets_lda_reports.csv` — one row per (firm, bill, quarter); filtered to Fortune 500 firms, `ind='y'` records, at least one named lobbyist.

**Pipeline:**

1. Build ranked bill lists: for each firm, rank top-30 bills by `amount_allocated / total_budget` share across the full Congress.
2. Filter mega-bills: bills lobbied by >50 firms excluded (`MAX_BILL_DF = 50` in `config.py`) — analogous to stop-word removal.
3. For every pair of firms sharing ≥1 top-30 bill: compute RBO (p=0.85) over their bill-priority rankings.
4. For each shared bill, compare first-quarter adoption: the firm that first appears lobbying the bill gets a first-mover credit. Ties (same quarter) split as 0.5 each.
5. Aggregate credits across all shared bills per pair → `source_firsts`, `target_firsts`, `net_temporal = source_firsts − target_firsts`.
6. Emit two directed edges per pair: weight = `(source_firsts + ties/2) / shared_bills × rbo` (proportional RBO, sums to `rbo` across both edges).

**Key node attribute — `net_strength` (primary):**
`net_strength = Σ_j [rbo(i,j) × net_temporal(i,j)]`
RBO-weighted temporal dominance across all pairings. Positive = net agenda-setter; negative = net follower.

**Supporting attributes:**
- `net_influence`: total first-mover wins minus losses (unweighted bill count; reference only)
- `wc_net_strength`: within-community variant of `net_strength` (same-community neighbors only)

Node color in PNG: green = `net_strength > 0`, red = `net_strength < 0`, gray = neutral/isolated.

---

### Empirical Highlights (116th Congress)

| Statistic | Value |
|---|---|
| Fortune 500 firms with ≥1 lobbying report | 305 |
| Firms in directed network (≥1 edge) | 293 |
| Directed edges | 5,612 |
| Median `net_strength` | −0.01 |

**Top agenda-setters by `net_strength`:**

| Firm | net_strength |
|---|---|
| Cummins | +13.4 |
| IBM | +12.6 |
| DTE Energy | +11.8 |
| Northrop Grumman | +11.2 |
| Boeing | +10.9 |

**Top followers by `net_strength`:**

| Firm | net_strength |
|---|---|
| State Farm | −14.2 |
| Entergy | −13.9 |
| Amazon | −12.7 |
| AEP | −11.4 |
| PPL | −10.8 |

(Exact values from `data/congress/116/node_attributes.csv`.)

---

### Mediation (`src/affiliation_mediated_adoption.py`)

Tests whether directed adoption pairs (A→B, bill) are mediated by shared lobbyists or lobbying firms. Bill-level analysis: ~42% of directed adoption events have a shared lobbyist in the first quarter of bill adoption. Network-level: 89% of directed pairs share at least one lobbyist across their full portfolios. Positive controls (random non-influencer pairs) show significantly lower mediation rates — χ² p < 0.001. See `docs/affiliation_mediated_adoption_summary.md` for full results. Design decision §24.

---

### Enrichment and Export

- `src/enrich_directed_gml.py` — adds `num_bills`, `bill_aff_community`, `within_comm_net_str`, `within_comm_net_inf` to GML node attributes
- `src/gephi_style_export.py` — writes Gephi-ready GEXF (`visualizations/gexf/rbo_directed_influence.gexf`)

---

### Cross-Congressional Stability (111th–117th Congress)

`src/cross_congressional_stability.py` analyzes direction/magnitude/rank stability for 135 firms present in all 7 congresses (2009–2022).

Key findings:
- **Direction consistency:** median 77% of cross-session appearances maintain the same direction for multi-session pairs (Analysis 07, Part C). Mean = 77.3%.
- **Rank stability:** Spearman ρ between adjacent congresses ranges 0.36–0.52 for `net_strength`; stable firms persist in the top tier across congresses.
- **Agenda-setter identity:** top-5 by `net_strength` has ~40–60% overlap between adjacent congresses (Jaccard ≈ 0.25 on top-20 set).

---

### Focused Analyses (`src/analysis/`)

| Script | Question |
|---|---|
| `01_primary_directed_influence.py` | Top-30 agenda-setters, spend comparison, sector leaderboards |
| `02_mediation.py` | Shared-lobbyist mediation test |
| `03_industry_hierarchy.py` | Kendall's W rank stability by sector |
| `04_cross_congressional.py` | Spearman ρ heatmap, RBO list similarity |
| `05_multi_congress.py` | Jaccard top-set overlap, entry/exit transitions |
| `06_centrality_vs_agenda_setters.py` | Centrality vs. net_strength Spearman ρ |
| `07_strategic_complementarity.py` | BCZ payoff complementarity + direction persistence |
| `08_bill_adoption_cascading.py` | Bill adoption diffusion, LPM regression |

All outputs written to `outputs/analysis/`.

---

### Output Files

| File | Contents |
|---|---|
| `data/congress/116/rbo_directed_influence.csv` | Edge list (source, target, weight, rbo, net_temporal, source_firsts, target_firsts, shared_bills) |
| `data/congress/116/node_attributes.csv` | Node attributes (net_strength, net_influence, total_spend, sector, …) |
| `data/congress/116/ranked_bill_lists.csv` | Per-firm top-30 bill rankings |
| `data/affiliation_mediated_adoption.csv` | Bill-level mediated adoption dataset |
| `data/rbo_edges_enriched.csv` | RBO edges with mediation rates and connectivity |
| `visualizations/gml/rbo_directed_influence.gml` | Enriched GML for Gephi |
| `visualizations/gexf/rbo_directed_influence.gexf` | Gephi-ready GEXF |
| `visualizations/png/rbo_directed_influence.png` | Directed circular plot (top-20 firms) |

---

### Design Decision References

- §0 — influence = agenda-setting (conceptual framework)
- §20 — full RBO directed influence methodology (scoring rule, temporal precedence, references)
- §24 — affiliation-mediated adoption findings
- §37 — redesign from quarterly to full-Congress bill rankings
- §38 — multi-congress pipeline and 135-firm stable set
- §39 — analysis scripts (01–08), methods and key findings
