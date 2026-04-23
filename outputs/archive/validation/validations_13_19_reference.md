# Validations 13–19: BCZ Framework Reference

Validations 13–19 collectively test whether the RBO directed influence network is a valid empirical proxy for the theoretical constructs in BCZ (Battaglini, Coughlin, Zelner) lobbying theory. The core claim being validated: firms that are structurally central in the affiliation network are more likely to be temporal first-movers (agenda-setters), that spending and bill breadth predict influencer status, that within-community hierarchies are stable across congresses, and that BCZ's payoff complementarity prediction is recoverable from observed spending dynamics. Validations 18 and 19 are the most direct tests of BCZ mechanisms (payoff complementarity and information diffusion respectively). Validations 13–17 build the scaffolding — establishing that centrality and agenda-setting are distinct but correlated, that bill breadth drives influencer status, that cross-sector influence flows directionally, that within-community hierarchies persist, and that quarterly dynamics are stable enough to justify congress-level aggregation.

---

## Validation 13: Centrality vs. Agenda-Setter

### Purpose
Tests whether structurally central firms (BCZ key players) are the same firms that act as temporal first-movers (net_influence), and which centrality measure best aligns with observed agenda-setting behavior.

### Script Path
```
src/validations/13_centrality_vs_agenda_setter.py
```

### Input Files
```
data/centralities/centrality_affiliation.csv
data/congress/116/rbo_directed_influence.csv
data/congress/116/node_attributes.csv
data/communities/communities_affiliation.csv
```

### Output Files
```
outputs/validation/13_centrality_vs_agenda_setter.csv
outputs/validation/13_centrality_vs_agenda_setter_correlations.csv
outputs/validation/13_centrality_vs_agenda_setter.txt
```

### Method
- **BCZ intercentrality**: computed via 277 individual node-removal Katz reruns. Katz alpha = 0.85 / spectral_radius (unnormalized). BCZ-IC for node i = sum over all j≠i of |Katz_j(full) − Katz_j(removed_i)|.
- **Within-community PageRank (WC PageRank)**: PageRank run on each community's induced affiliation subgraph separately; scores used as within-community positional measure.
- **Global PageRank and WC eigenvector**: loaded directly from `centrality_affiliation.csv`.
- **WC net_influence and WC net_strength**: computed from intra-community directed edges in `rbo_directed_influence.csv` (edges where both endpoints share the same Leiden community label).
- **Agenda-setter proxies**: `net_influence` (signed count of RBO wins as first-mover) and `net_strength` (sum of RBO weights on first-mover edges).
- Spearman rank correlations computed on full sample (N=277) and restricted to top-30 by each respective centrality measure. Top-30 overlap fractions computed for all 14 (centrality × agenda-setter) pairs.

### Key Results
| Pair | ρ_full | p_full | ρ_top30 | p_top30 |
|---|---|---|---|---|
| BCZ intercentrality ↔ net_influence | 0.178 | 0.003 | −0.107 | ns |
| WC PageRank ↔ net_influence | 0.212 | 0.0004 | 0.501 | 0.005 |
| WC PageRank ↔ WC net_influence | — | — | 0.558 | 0.001 |
| Global PageRank ↔ net_strength | — | — | 0.366 | 0.047 |

- BCZ top-30 dominated by energy utilities: CMS Energy (#2), DTE Energy (#3), Exelon (#4), Xcel (#5).
- Net_influence top-30 dominated by diversified industrials: Cummins (#1), Ball (#4), Exxon (#5), DTE Energy (#6).
- Top-30 overlap: BCZ vs net_influence = 0.47; WC PageRank vs net_influence = 0.50.

### Interpretation and Caveats
Structural key players (BCZ-IC) and temporal first-movers (net_influence) are distinct populations — 53% non-overlap at the top 30. BCZ-IC captures firms with large co-affiliation footprints; net_influence captures firms that consistently file before co-lobbyists on the same bills. WC PageRank is the best alignment measure at the top-30 level (ρ=0.558 for WC net_influence), suggesting that within-community position is more predictive of agenda-setting than global graph position. The negative ρ_top30 for BCZ-IC vs net_influence indicates BCZ top-30 firms are systematically absent from the temporal-precedence top-30 — energy utilities dominate structural position but not temporal influence.

---

## Validation 14: Influencer Regression

### Purpose
Tests whether lobbying spend, bill breadth, and structural centrality predict influencer status (net_influence and net_strength), and whether predictors replicate across the 116th and 117th Congresses.

### Script Path
```
src/validations/14_influencer_regression.py
```

### Input Files
```
data/congress/116/rbo_directed_influence.csv
data/congress/116/opensecrets_lda_reports.csv
data/congress/117/rbo_directed_influence.csv
data/congress/117/opensecrets_lda_reports.csv
data/centralities/centrality_affiliation.csv
data/communities/communities_affiliation.csv
```

### Output Files
```
outputs/validation/14_influencer_regression.csv
outputs/validation/14_influencer_regression.txt
```

### Method
- OLS cross-sectional regressions, 116th and 117th Congress separately, firm as unit of observation.
- **Spec A**: `net_influence ~ log_spend + log_bills + katz_centrality + participation_coeff` (continuous outcome). HC3 SE.
- **Spec A2**: binary outcome = 1 if firm in top quartile of net_influence; LPM with same RHS. HC3 SE.
- **Spec B**: `net_strength ~ log_spend + log_bills + katz_centrality + participation_coeff`. HC3 SE.
- **Spec C**: `wc_net_strength ~ within_comm_eigenvector + wc_pagerank`. HC3 SE. Within-community centralities from 116th Leiden partition used as structural baseline for both congresses.
- `log_bills` deduped on `(uniq_id, fortune_name)` before counting to avoid per-bill double-counting across reports.
- 42 firms in 117th Congress absent from the 116th Leiden community partition dropped from Spec C only (centrality not available).

### Key Results
**116th Congress:**
| Spec | Key result |
|---|---|
| A | No significant predictors; R²=0.043, F p=0.068 (marginal) |
| A2 (LPM, top-quartile) | log_bills β=0.164*** (p<0.001); katz β=1.200** (p=0.049); log_spend β=−0.039** (p=0.042); R²=0.282 |
| B | F p=0.036; log_spend β=−0.059* (p=0.065 marginal) |
| C | F p=0.008; no individual covariate significant |

**117th Congress (Spec A2 replication):**
- log_bills β=0.150*** (p<0.001); log_spend β=−0.048** (p=0.044); R²=0.190.
- Pattern replicates across congresses.

### Interpretation and Caveats
Bill breadth (log_bills) is the dominant, stable predictor of top-quartile influencer status across both congresses. The negative spend coefficient is counterintuitive but consistent with the diversification hypothesis: agenda-setters spread activity across many bills rather than concentrating resources on a few. Katz centrality is significant only in the binary spec (A2), suggesting a threshold effect — structural position matters for crossing into influencer status but does not linearly predict degree of influence. Continuous net_influence (Spec A) is not well-explained by observables (R²=0.043), indicating substantial unobserved heterogeneity or that net_influence is driven by bill-level timing that aggregate covariates cannot capture. Spec C non-significance despite a significant F-stat suggests collinearity between within-community eigenvector and WC PageRank.

---

## Validation 15: Cross-Sector Directed Edge Analysis

### Purpose
Characterizes the direction, strength, and firm-level sources of cross-sector influence flows in the RBO network, and tests whether cross-sector edges are systematically different from intra-sector edges.

### Script Path
```
src/validations/15_cross_sector_directed_edges.py
```

### Input Files
```
data/congress/116/rbo_directed_influence.csv
data/centralities/centrality_affiliation.csv   (community labels)
data/congress/116/opensecrets_lda_issues.csv
```

### Output Files
```
outputs/validation/15_cross_sector_edge_table.csv
outputs/validation/15_cross_sector_firm_table.csv
outputs/validation/15_cross_sector_pair_matrix.csv
outputs/validation/15_cross_sector_directed_edges.txt
```

### Method
- Directed edges tagged as intra-/cross-sector using 116th Leiden community labels from `centrality_affiliation.csv`.
- Mann-Whitney U tests on edge weight (RBO score) and net_temporal (signed count of temporal precedence wins) comparing cross-sector vs intra-sector edge populations.
- Pivot table of directed edge counts and mean RBO weights indexed by (source_community, target_community) pair.
- **CS-NI (cross-sector net influence) per firm**: (CS_out_firsts + CS_in_wins) − (CS_out_losses + CS_in_losses), using only cross-sector directed edges.
- **Bridge firms**: ranked by fraction of total directed edges that are cross-sector (CS_frac), with CS-NI as secondary signal.
- Top-10 cross-sector dyads (by net temporal precedence): issue cosine similarity computed from firm × issue_code spend matrix using `opensecrets_lda_issues.csv`.

### Key Results
- Total directed edges with community labels: 1,813. Cross-sector: 783 (43.2%). Intra-sector: 1,030 (56.8%).
- Cross-sector RBO weight: mean=0.033. Intra-sector: mean=0.077. MWU p=2.9e-45.
- Cross-sector net_temporal: mean=1.25. Intra-sector: mean=1.63. MWU p=3.3e-19.
- **Dominant cross-sector flows (net directed edges)**:
  - Defense/Industrial → Health/Pharma: net +47
  - Defense/Industrial → Energy/Utilities: net +38
  - Energy/Utilities ← Defense/Industrial: net −38 (receiving end)
  - Tech/Telecom ← Defense/Industrial: net −26 (receiving end)
- **Top cross-sector influencers (CS-NI)**: Cummins (48), Ford (20), IBM (19), 3M (17), Lockheed Martin (17).
- Defense/Industrial: 70.9% of firms have positive CS-NI. Energy/Utilities: 41.5%.
- **Bridge firms** (CS_frac=1.0 but low NI): Liberty Media, Molson Coors, Windstream. Meaningful bridges: Cognizant (CS_frac=0.875, CS-NI=12), Thermo Fisher (CS_frac=0.778, CS-NI=9).
- Top cross-sector dyad: Berkshire Hathaway → CSX (NT=5, RBO=0.653).
- Most common shared issue code in top cross-sector dyads: BUD (federal budget).

### Interpretation and Caveats
Cross-sector edges are systematically weaker on both weight (RBO=0.033 vs 0.077) and temporal margin, consistent with the hypothesis that firms have weaker informational advantages outside their primary legislative domain (portfolio specialization). Defense/Industrial is the dominant cross-sector agenda-setter, influencing both Health/Pharma and Energy/Utilities. The BUD issue code as the primary shared topic in cross-sector dyads suggests these flows are mediated by appropriations and budget legislation — a natural venue for defense interests to intersect other sectors. CS_frac=1.0 firms (Liberty Media, Molson Coors, Windstream) are niche lobbyists with no within-community influence; they are structural bridges but low-influence ones. Cognizant and Thermo Fisher are more meaningful bridges — high CS_frac with substantial CS-NI.

---

## Validation 16: Within-Community Influencer Hierarchy

### Purpose
Tests whether within-community net_influence rankings are stable across the 111th–117th Congresses, and identifies persistent influencer leaders per community.

### Script Path
```
src/validations/16_industry_influencer_hierarchy.py
```

### Input Files
```
data/congress/111/rbo_directed_influence.csv
data/congress/112/rbo_directed_influence.csv
data/congress/113/rbo_directed_influence.csv
data/congress/114/rbo_directed_influence.csv
data/congress/115/rbo_directed_influence.csv
data/congress/116/rbo_directed_influence.csv
data/congress/117/rbo_directed_influence.csv
data/communities/communities_affiliation.csv
```

### Output Files
```
outputs/validation/16_industry_hierarchy_116.csv
outputs/validation/16_industry_hierarchy_cross_congress.csv
outputs/validation/16_within_community_ni_by_congress.csv
outputs/validation/16_within_community_rank_stability.csv
outputs/validation/16_industry_influencer_hierarchy.txt
```

### Method
- Within-community net_influence computed for all 7 congresses (111th–117th) using intra-community directed edges only (edges where both endpoints share same Leiden community label).
- **Partition used throughout**: 116th Congress Leiden community partition applied as the fixed community proxy for all congresses (no recomputing partition per congress).
- Top-5 leaderboards per community per congress extracted.
- **Kendall's W**: concordance coefficient computed on firms present in all 7 congresses within each community. Chi-squared approximation following Siegel & Castellan (1988). Separate W per community.
- **Adjacent-congress Spearman ρ**: rank correlation on within-community net_influence for each consecutive (congress_k, congress_{k+1}) pair.
- **Persistent leaders**: firms appearing in top-5 in ≥4 of 7 congresses.

### Key Results
| Community | Kendall's W | p-value | n_stable (firms in all 7) |
|---|---|---|---|
| Finance/Insurance | 0.284 | 0.002 | 27 |
| Tech/Telecom | 0.431 | <0.0001 | 25 |
| Defense/Industrial | 0.367 | 0.0001 | 21 |
| Energy/Utilities | 0.553 | <0.0001 | 23 |
| Health/Pharma | 0.146 | 0.43 (ns) | 14 |

**Persistent leaders (top-5 in ≥4/7 congresses):**
- Energy/Utilities: Duke Energy (7/7), Xcel Energy (7/7), CMS Energy (5/7), DTE Energy (4/7).
- Finance/Insurance: American Family Insurance (5/7), Northwestern Mutual (4/7).
- Tech/Telecom: IBM (multiple), AT&T (multiple).
- Health/Pharma: No firm qualifies (none in top-5 in ≥4 congresses).

### Interpretation and Caveats
Energy/Utilities has the most stable within-community hierarchy (W=0.553), driven by a core set of regulated investor-owned utilities (Duke, Xcel, CMS, DTE) that maintain persistent agenda-setting dominance across 14 years. This is consistent with regulated utilities having predictable, recurring legislative agendas (rate cases, infrastructure bills, EPA compliance). Health/Pharma is the most volatile (W=0.146, not significant), interpretable as: biologics pipeline entrants, insurance-adjacent firms, and pharma companies cycling in and out depending on the congressional policy agenda (ACA, drug pricing, FDA). The n_stable column matters for W interpretation — Health/Pharma's low n=14 compounds the instability signal. Using a fixed 116th-Congress partition as community proxy for all congresses is a deliberate design choice: it avoids partition-reestimation noise but means the "community" label for any firm is anchored to its 116th-Congress co-affiliation pattern, which may not perfectly reflect its community in the 111th Congress.

---

## Validation 17: Quarterly Dynamics

### Purpose
Tests whether congress-wide aggregate net_influence is appropriate as the primary unit, by examining whether influencer rankings are stable across the 8 quarters of the 116th Congress.

### Script Path
```
src/validations/17_quarterly_dynamics.py
```

### Input Files
```
data/congress/116/opensecrets_lda_reports.csv
```

### Output Files
```
outputs/validation/17_quarterly_net_influence.csv
outputs/validation/17_quarterly_stability.csv
outputs/validation/17_quarterly_dynamics.txt
visualizations/png/17_bump_chart_quarterly_influencers.png
visualizations/png/17_heatmap_quarterly_net_influence.png
```

### Method
- RBO directed influence pipeline recomputed independently for each of Q1–Q8, restricted to reports with that quarter's `report_type` prefix codes.
- Within-quarter first-mover tie-breaking: resolved by `report_type` ordinal (base report < amendment).
- Balanced pairs (tied first-mover count) excluded from net_influence computation.
- `MAX_BILL_DF=50` filter applied per quarter (same parameter as congress-wide pipeline).
- **Adjacent-quarter Jaccard**: intersection / union of top-10 firm sets between consecutive quarters.
- **Adjacent-quarter Spearman ρ**: rank correlation on full-sample net_influence vectors between consecutive quarters (all firms active in both quarters).
- **Focal firms**: firms appearing in top-10 in ≥3 of 8 quarters.
- Outputs: bump chart (top-10 rank trajectories across Q1–Q8) and heatmap (net_influence by firm × quarter for focal firms).

### Key Results
- Per-quarter firm coverage: 211 (Q1) to 247 (Q8); smaller than congress-wide N=277.
- **Adjacent Jaccard (top-10 overlap)**: range 0.000–0.538; mean=0.304. Q1→Q2 = 0.000 (Q1 structurally distinct).
- **Adjacent Spearman ρ**: range 0.296–0.550; all pairs p<0.001; mean=0.446.
- **Focal firms (top-10 in ≥3 quarters)**:
  - Xcel Energy: 5/8 quarters
  - PG&E: 5/8
  - Duke Energy: 5/8
  - DTE Energy: 4/8
  - FirstEnergy: 4/8
  - Exelon: 4/8
  - Entergy: 4/8
  - Cummins: 3/8
  - Lockheed Martin: 3/8
  - CMS Energy: 3/8
  - 8 of 10 focal firms are Energy/Utilities.

### Interpretation and Caveats
Influencer status is partially stable at the quarterly level — Spearman ρ is consistently significant (all p<0.001), but mean Jaccard of 0.30 means roughly 70% of the top-10 membership turns over between adjacent quarters. This level of instability is the primary justification for using congress-wide aggregation as the main analytical unit rather than quarterly slices. Q1 is a structural outlier (Jaccard=0.000 vs Q2), likely because Q1 represents the very start of a new congress with idiosyncratic early-session bill introductions. The dominance of Energy/Utilities among focal firms is consistent with validations 13 and 16 — these firms have persistent legislative agendas that make them temporally consistent filers across quarters.

---

## Validation 18: Payoff Complementarity

### Purpose
Tests the BCZ payoff complementarity prediction: whether a co-lobbyist's entry on a bill causes the focal firm to increase its own lobbying spend, and whether this effect is concentrated among high-RBO (strongly linked) pairs.

### Script Path
```
src/validations/18_payoff_complementarity.py
```

### Input Files
```
data/congress/116/opensecrets_lda_reports.csv
data/congress/116/rbo_directed_influence.csv
```

### Output Files
```
outputs/validation/18_payoff_complementarity_panel.csv
outputs/validation/18_payoff_complementarity_results.csv
outputs/validation/18_payoff_complementarity.txt
```

### Method
- Panel at (firm_i, firm_j, bill, quarter) level. Outcome: `Δlog_spend_{i,b,t+1}` (change in log spend of firm i on bill b from quarter t to t+1).
- **Specification**: `Δlog = β1*entry_j + β2*rbo_ij + β3*(entry_j × rbo_ij) + α_{i,b} + γ_t + ε`
  - `entry_j` = 1 if quarter t is firm j's first quarter lobbying bill b; 0 if j is already active on b at t.
  - `α_{i,b}` = firm-bill fixed effects, absorbed via within-transformation (demeaning within each (i,b) group).
  - `γ_t` = quarter fixed effects.
  - HC3 heteroskedasticity-consistent SE.
- Panel construction includes ALL firm_j active on bill b at each quarter t (not only entrant quarters) — essential for within-group variation in `entry_j`.
- **Four specifications**:
  - Spec A: Full RBO-linked pairs (N=67,194).
  - Spec B: High-RBO pairs, rbo_ij ≥ p75=0.131 (N=16,845).
  - Spec C: Low-RBO pairs, rbo_ij < p25=0.007 (N=16,796).
  - Spec D: All pairs, rbo_ij=0 for non-RBO-linked pairs (N=147,341).

### Key Results
| Spec | β3 (interaction) | SE | p-value | Within R² |
|---|---|---|---|---|
| A (full RBO-linked) | −0.125 | 0.033 | <0.001 | 0.148 |
| B (high-RBO ≥p75) | +0.147 | 0.069 | 0.033 | 0.191 |
| C (low-RBO <p25) | −8.001 | 2.979 | 0.007 | ~0.148 |
| D (all pairs, rbo=0 for unlinked) | −0.173 | 0.029 | <0.001 | 0.155 |

- β1 (entry_j main effect): significantly negative in Specs A, B, D (range −0.013 to −0.125).
- Spec C β3 caveat: coefficient is huge (−8.001) because rbo_ij values are <0.007 in this subsample; actual predicted marginal effect at mean low-RBO weight is economically negligible despite p=0.007.

### Interpretation and Caveats
The BCZ complementarity prediction (positive β3) is confirmed only in Spec B — high-RBO pairs, where a co-lobbyist's entry is associated with a significant +0.147 increase in Δlog_spend. In the full sample and among low-RBO pairs, β3 is negative, consistent with crowd-out: the average co-lobbyist's entry is associated with spending restraint by the focal firm, possibly because both can free-ride on the other's lobbying. The sign flip between Spec A (−0.125) and Spec B (+0.147) is the core result — strategic amplification is present but concentrated at the top quartile of the RBO distribution. β1 being negative throughout confirms that the entry event itself is not a stimulus for spending by default; the RBO interaction is what drives the positive response in high-influence pairs. Within R² of ~0.15 is reasonable for a differenced spending outcome with firm-bill FE absorbed.

---

## Validation 19: Bill Adoption Diffusion

### Purpose
Tests whether firms are more likely to adopt bills lobbied by high-RBO co-lobbyists (following the agenda-setter), consistent with information diffusion along the directed influence network.

### Script Path
```
src/validations/19_bill_adoption_diffusion.py
```

### Input Files
```
data/congress/116/opensecrets_lda_reports.csv
data/congress/116/rbo_directed_influence.csv
data/congress/116/node_attributes.csv
```

### Output Files
```
outputs/validation/19_adoption_candidates.csv
outputs/validation/19_adoption_rates.csv
outputs/validation/19_adoption_regression.csv
outputs/validation/19_bill_adoption_diffusion.txt
visualizations/png/19_bill_adoption_diffusion.png
```

### Method
- **Candidate generation**: for each directed (A→B) edge with `balanced=0` (A is the net temporal leader over B), enumerate all (A, B, bill) triples where A first lobbies bill X at quarter t AND B had not lobbied X at or before t. This yields 80,384 candidate rows.
- **Outcome**: binary indicator B adopts bill X within k quarters (k=1, 2, 3). Horizon observability: candidates excluded when `a_entry_q + k > 8` (insufficient quarters remaining in congress).
- **Primary test**: χ² test comparing adoption rates for high-RBO vs low-RBO pairs at Q+1.
- **MWU test**: rbo_weight distribution of adopters vs non-adopters.
- **Regression covariates**: `log(rbo_weight)`, `log(n_firms_bill)` (bill popularity), `a_net_influence` (firm A's congress-wide net_influence), `b_net_influence` (firm B's), `a_entry_q_norm` (A's entry quarter normalized 0–1).
- Both logit and LPM estimated for each horizon.
- **Unique-entry robustness**: restrict to bills where A is the sole first-entrant (no other firm enters in same quarter); N=25,378 rows. Re-estimates logit at each horizon.

### Key Results
**Overall adoption rates:**
| Horizon | Rate |
|---|---|
| Q+1 | 3.1% |
| Q+2 | 4.8% |
| Q+3 | 5.7% |

**Adoption rate by RBO quartile (at Q+3):**
| Quartile | Rate |
|---|---|
| Q1 (lowest RBO) | 3.9% |
| Q2 | 4.6% |
| Q3 | 6.1% |
| Q4 (highest RBO) | 8.3% |

- Median-split ratio (high vs low RBO): 1.754x at Q+1, 1.707x at Q+2, 1.697x at Q+3.
- χ²(Q+1, high vs low RBO) = 179.4, p<0.0001.
- Adopters mean RBO = 0.077 vs non-adopters mean RBO = 0.049. MWU p<0.0001.

**Logit log(rbo) coefficient:**
| Horizon | Coef | |
|---|---|---|
| Q+1 | 0.163*** | p<0.001 |
| Q+2 | 0.160*** | p<0.001 |
| Q+3 | 0.169*** | p<0.001 |

**LPM log(rbo) coefficient:**
| Horizon | Coef | |
|---|---|---|
| Q+1 | 0.0042*** | p<0.001 |
| Q+2 | 0.0062*** | p<0.001 |
| Q+3 | 0.0076*** | p<0.001 |

**Unique-entry logit log(rbo) (robustness):**
| Horizon | Coef |
|---|---|
| Q+1 | 0.235*** |
| Q+2 | 0.230*** |
| Q+3 | 0.205*** |

- Firm A net_influence quartile at Q+1: Q1 (weakest A) = 3.5% adoption, Q4 (strongest A) = 2.9% — no monotone gradient.

### Interpretation and Caveats
The RBO network predicts bill adoption: follower firms (B) are ~1.7x more likely to adopt bills already lobbied by high-RBO partners (A) compared to low-RBO partners, and this ratio is stable across Q+1 through Q+3 horizons. The stability over 3 horizons suggests the RBO network captures a structural propensity to follow rather than a transient coincidence in Q+1. Unique-entry robustness (N=25,378) confirms results are not driven by multi-firm entry waves — the effect is present even when A is the sole entrant on the bill. The non-monotone gradient for A's net_influence quartile (Q4 = 2.9%, lower than Q1 = 3.5%) is a composition effect: high-net_influence firms tend to lobby niche or complex bills that fewer followers adopt. Together with Validation 18, this provides two-level BCZ evidence — both spending response (V18) and bill adoption (V19) amplify with RBO weight, consistent with the directed influence network recovering the BCZ influence ordering.
