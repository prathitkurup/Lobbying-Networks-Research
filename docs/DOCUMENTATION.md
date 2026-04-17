# Technical Documentation

Comprehensive reproduction guide for the Corporate Lobbying Networks project. Covers data, pipeline, and methodology for all key scripts, with emphasis on the directed influence network.

---

## Table of Contents

1. [Data Sources](#1-data-sources)
2. [Configuration](#2-configuration)
3. [Extraction Pipeline](#3-extraction-pipeline)
4. [Network Construction](#4-network-construction)
5. [Directed Influence Network (Primary)](#5-directed-influence-network-primary)
6. [GML Enrichment and Gephi Export](#6-gml-enrichment-and-gephi-export)
7. [Utility Modules](#7-utility-modules)
8. [Validation Scripts](#8-validation-scripts)
9. [Key Parameters](#9-key-parameters)
10. [Output Files Reference](#10-output-files-reference)
11. [Affiliation-Mediated Adoption Analysis](#11-affiliation-mediated-adoption-analysis)
12. [Cross-Congressional Stability Analysis](#12-cross-congressional-stability-analysis)

---

## 1. Data Sources

### OpenSecrets CRP Bulk Data (`data/OpenSecrets/`)

All networks derive from four pipe-delimited files (latin-1 encoded):

| File | Key columns | Role |
|---|---|---|
| `lob_lobbying.txt` | `uniq_id`, `client`, `ultorg`, `amount`, `isfirm`, `self`, `ind`, `year`, `report_type` | One row per LDA filing report |
| `lob_lobbyist.txt` | `uniq_id`, `lobbyist`, `lobbyist_id`, `year` | Named lobbyists per report |
| `lob_issue.txt` | `issue_id`, `uniq_id`, `issue_code`, `year` | Issue codes per report |
| `lob_bills.txt` | `issue_id`, `congress`, `bill_number` | Bills per issue entry |

`ind` field: OpenSecrets marks each filing as `ind='y'` (valid, countable) or `ind=''` (superseded original or double-counted subsidiary). Only `ind='y'` records are used.

`report_type` field: `q1`–`q4` are original quarterly filings; `q1a`–`q4a` are amendments. Values `2019:q1–q4` and `2020:q1–q4` give 8 quarters total, mapped to integer quarters 1–8.

### Fortune 500 Name Mapping (`data/manual_opensecrets_name_mapping.json`)

Manually curated JSON: `{canonical_fortune500_name: [crp_variant_1, crp_variant_2, ...]}`. Both the canonical name and each variant are added to the lookup table. Resolution tries `ultorg` first (to catch subsidiaries), then falls back to `client`. Approximately 500 firms, ~1,400 lookup entries total.

---

## 2. Configuration (`src/config.py`)

All paths and shared constants. Scripts import from here rather than hardcoding paths.

| Constant | Value | Description |
|---|---|---|
| `ROOT` | project root | Resolved from `__file__` |
| `DATA_DIR` | `ROOT/data` | All data outputs |
| `MANUAL_OPENSECRETS_NAME_MAPPING` | `data/manual_opensecrets_name_mapping.json` | Active name mapping |
| `OPENSECRETS_OUTPUT_CSV` | `data/opensecrets_lda_reports.csv` | Primary extraction output |
| `OPENSECRETS_ISSUES_CSV` | `data/opensecrets_lda_issues.csv` | Issue-level output |
| `TARGET_CONGRESS` | `116` | Congress number |
| `CONGRESS_FILING_YEARS` | `[2019, 2020]` | Year filter |
| `MAX_BILL_DF` | `50` | Mega-bill filter threshold |
| `MAX_ISSUE_DF` | `None` | Issue filter (disabled) |

---

## 3. Extraction Pipeline (`src/opensecrets_extraction.py`)

The single entry point. All downstream scripts read from its output CSVs.

### Filters applied (in order)

1. **Year filter**: retain only records with `year` in `{2019, 2020}`.
2. **`ind='y'` validity filter**: drops superseded originals (the original quarterly when an amendment exists carries `ind=''`) and double-count subsidiary records. This is OpenSecrets' own deduplication — 29,912 of 30,360 `self='i'` records carry `ind=''` and are correctly excluded.
3. **Active report filter**: of the valid `ind='y'` reports, only those with ≥1 named lobbyist are kept. Reports with no lobbyists are retainer/no-activity filings (e.g. `report_type = q1n`) with no issue codes or bills.

### Key parsing functions

- `load_lobbying_f500()` — reads `lob_lobbying.txt`, applies filters, resolves firm names.
- `load_lobbyist_map()` — reads `lob_lobbyist.txt`, returns `{uniq_id: [lobbyist_names]}`.
- `load_issue_map()` — reads `lob_issue.txt`, returns `{uniq_id_upper: set(issue_ids)}`.
- `load_bill_map()` — reads `lob_bills.txt`, returns `{issue_id: set(bill_numbers)}`.

### Bill expansion

Each active report is expanded to one row per linked bill. `amount_allocated = report_amount / n_bills`. Reports with no linked bills keep one row with `bill_number = NaN` and `amount_allocated = amount`. This one-row-per-bill format is then aggregated at network construction time, not here.

### Outputs

| File | Schema | Description |
|---|---|---|
| `opensecrets_lda_reports.csv` | `uniq_id, fortune_name, bill_number, amount, amount_allocated, is_self_filer, self_type, year, congress, report_type, lobbyists, registrant, ...` | Primary: one row per (report, bill) |
| `opensecrets_lda_issues.csv` | `uniq_id, fortune_name, issue_code, amount, amount_allocated, year, congress` | One row per (report, issue_code) |
| `lobbyist_client_116_opensecrets.csv` | `lobbyist_id, lobbyist, fortune_name` | Deduplicated lobbyist–firm pairs (archived reference) |

The `lobbyists` column in reports CSV is pipe-separated lobbyist names per report. The lobbyist affiliation network reads directly from this column rather than from the archived pairs CSV.

---

## 4. Network Construction

### Shared logic

All bill-level networks follow the same preprocessing sequence before building edges:

1. Load `opensecrets_lda_reports.csv` via `utils/data_loading.load_bills_data()`.
2. Aggregate: `df.groupby(["fortune_name", "bill_number"])["amount_allocated"].sum()` — collapses multiple report rows for the same (firm, bill) to true total spend.
3. Compute fracs: `frac = amount_allocated / total_budget` per firm. Zero-budget firms excluded.
4. Apply prevalence filter: drop bills lobbied by more than `MAX_BILL_DF` firms (default 50). This removes 16 omnibus/appropriations bills (CARES Act, NDAA, etc.) that account for ~97.5% of affiliation edges but carry no strategic alignment signal.
5. Build ranked lists (for RBO): `{firm: [bill, ...]}` sorted by `amount_allocated` descending, top 30 bills per firm.

**Two-stage filtering for cosine/RBO:** fracs are computed on *all* bills (stage 1), then mega-bills are excluded only for building ranked lists / similarity pairs (stage 2). This preserves economically correct frac denominators.

### Bill Affiliation Network (`src/bill_affiliation_network.py`)

- Deduplicates to presence/absence: `df.drop_duplicates(["fortune_name", "bill_number"])`.
- Edge weight = `shared_bills(i,j)` (raw count) and `affil_norm = shared_bills / N_total_bills`.
- Uses `affil_norm` as primary weight for community detection and GML export.
- Canonical pair ordering throughout: `src, tgt = (a,b) if a < b else (b,a)`.

### RBO Similarity Network (`src/rbo_similarity_network.py`)

- Edge weight = RBO score between firm i and firm j's top-30 bill lists.
- RBO (Webber et al. 2010): `RBO(l1, l2, p) = (1−p) × Σ_{d=1}^{min(|l1|,|l2|)} p^{d-1} × |l1[:d] ∩ l2[:d]| / d`.
- `p = 0.85`: calibrated to observed Fortune 500 spend concentration (top-5 bills account for ~40% of spend), placing most weight on the top ~10 positions.
- Edges with weight = 0 (no shared top-30 bills) are excluded.

### Cosine Similarity Network (`src/cosine_similarity_network.py`)

- Edge weight = cosine similarity of frac vectors: `cos(i,j) = (u_i · u_j) / (||u_i|| × ||u_j||)`.
- Pivot to (firms × bills) matrix, compute pairwise cosine via `sklearn.metrics.pairwise.cosine_similarity`.
- Only pairs with cosine > 0 (shared non-zero-frac bills) produce edges.

### Issue Similarity Network (`src/issue_similarity_network.py`)

- Reads `opensecrets_lda_issues.csv` instead of the bill CSV.
- Same cosine construction over issue-code frac vectors (75 issue codes vs 2300+ bills).
- No prevalence filter applied (`MAX_ISSUE_DF = None`).

### Lobby Firm Affiliation Network (`src/lobby_firm_affiliation_network.py`)

- Edge weight = number of unique external registrant firms (K-street lobbying firms) shared by two Fortune 500 companies.
- Reads `opensecrets_lda_reports.csv`, filters to `is_self_filer == False`, deduplicates `(fortune_name, registrant)`.

### Lobbyist Affiliation Network (`src/lobbyist_affiliation_network.py`)

- Edge weight = number of unique named lobbyists shared by two Fortune 500 companies.
- Reads the `lobbyists` column from `opensecrets_lda_reports.csv`, splits on `|`, deduplicates `(fortune_name, lobbyist)`.

### Composite Network (`src/composite_similarity_network.py`)

- Combines bill affiliation, cosine, and RBO signals multiplicatively:
  `composite(i,j) = affil_norm(i,j) × cosine(i,j) × rbo(i,j)`.
- Inner join: edges only where all three signals are nonzero.
- AND logic: a pair must have high breadth AND portfolio alignment AND priority-ranking agreement.
- Resolution `γ = 0.25` (lower than individual networks due to sparser, more unimodal edge weight distribution).

### Community Detection (all networks)

`utils/community.detect_communities()` uses `leidenalg.RBConfigurationVertexPartition` with `seed=42`, `n_iterations=10`. Returns `(partition: {node: community_id}, modularity Q, summary: {community_id: [nodes]})`. Default `γ = 1.0`; the RBO similarity network uses `γ = 0.75`.

### Centrality (all networks)

`utils/centrality.compute_community_centralities()` computes:
- `within_comm_eigenvector`: eigenvector centrality on each community subgraph; falls back to weighted degree for <3 nodes or non-convergence.
- `z_score`: Guimerà-Amaral z-score — normalized within-community weighted degree.
- `participation_coeff` (P): Guimerà-Amaral P — fraction of edge weight crossing community boundaries. P=0: pure within-community; P→1: kinless cross-community bridger.
- `global_pagerank`: PageRank on the full graph.
- `katz_centrality`: Katz-Bonacich centrality with `α = 0.85/λ_max` (auto-calibrated to spectral radius).
- `ga_role`: Guimerà-Amaral 7-role taxonomy combining z and P.

---

## 5. Directed Influence Network (Primary) (`src/rbo_directed_influence.py`)

The primary analytical output. Run after extraction; no other network script needs to run first.

### Conceptual model

Influence is operationalized as **agenda-setting via temporal bill-adoption precedence**. If Firm A lobbied a shared bill before Firm B, A is the first-mover on that bill. Aggregating first-mover credits across all shared top-30 bills for a firm pair determines edge direction.

### Data flow

```
opensecrets_lda_reports.csv
  → assign_quarters()           adds integer quarter 1–8
  → aggregate_per_firm_bill()   collapses to total spend per (firm, bill)
  → compute_zero_budget_fracs() adds frac column
  → filter_bills_by_prevalence(MAX_BILL_DF=50)
  → build_ranked_lists(top_bills=30)   {firm: [bill, ...]} by spend desc
  → build_global_first_quarters()      {(firm, bill): min_quarter}
  → build_edges()               pairwise RBO + first-mover scoring
  → build_graph()               DiGraph with node attributes
```

### Quarter assignment

`assign_quarters()`: derives integer quarter from `report_type` (`q1`→1, `q2`→2, ...) and year offset (2019: +0, 2020: +4). Produces quarters 1–8 across the congress.

### Global first-quarter lookup

`build_global_first_quarters()`: `df.groupby(["fortune_name", "bill_number"])["quarter"].min()` — gives the first quarter each firm lobbied each bill across the full congress. Each bill counts at most once per firm.

### Edge construction (`build_edges()`)

For every pair (A, B) with RBO > 0:
1. Compute `rbo_score(list_A, list_B, p=0.85)` → edge weight.
2. Identify shared top-30 bills: `set(list_A) & set(list_B)`.
3. For each shared bill, compare `first_quarter[(A, bill)]` vs `first_quarter[(B, bill)]`. Increment `a_firsts` or `b_firsts`; ties increment `tie_count`.
4. Direction rule:
   - `a_firsts > b_firsts` → single directed A→B edge, `balanced=0`
   - `b_firsts > a_firsts` → single directed B→A edge, `balanced=0`
   - `a_firsts == b_firsts` → single canonical edge `min(A,B)→max(A,B)`, `balanced=1` (direction is alphabetical/arbitrary; balanced edges are excluded from directional metrics)

Edge columns: `source, target, weight, source_firsts, target_firsts, tie_count, shared_bills, net_temporal, balanced`.

### Node attributes (`build_graph()`)

| Attribute | Description |
|---|---|
| `net_influence` | Total first-mover wins minus losses across all pairings (count-based). Used for Gephi node sizing. |
| `total_firsts` | Total bills this firm lobbied first across all pairings |
| `total_losses` | Total bills this firm lobbied second across all pairings |
| `out_strength` | Sum of RBO weights on outgoing edges |
| `in_strength` | Sum of RBO weights on incoming edges |
| `net_strength` | `out_strength − in_strength` on directed (balanced=0) edges only; balanced edges excluded (arbitrary canonical direction) |
| `color` | `#2ECC71` green (net > 0), `#E74C3C` red (net < 0), `#95A5A6` gray (net = 0) |
| `label` | Firm name string |

### Key parameters

| Parameter | Default | Description |
|---|---|---|
| `RBO_P` | `0.85` | RBO top-weight parameter; higher p = more weight on top-ranked bills |
| `TOP_BILLS` | `30` | Bills per firm for ranking and RBO computation |
| `MAX_BILL_DF` | `50` | From `config.py`; mega-bill prevalence filter |
| `TOP_K` | `20` | Nodes in PNG visualization |

### Outputs

| File | Description |
|---|---|
| `data/rbo_directed_influence.csv` | Edge list with all columns above |
| `data/ranked_bill_lists.csv` | Long-format per-firm top-30 rankings (company, rank, bill_number, total_amount, budget_fraction) |
| `visualizations/gml/rbo_directed_influence.gml` | DiGraph GML for Gephi |
| `visualizations/png/rbo_directed_influence.png` | Circular plot of top-20 firms by involvement |

---

## 6. GML Enrichment and Gephi Export

### Enrichment (`src/enrich_directed_gml.py`)

Reads `rbo_directed_influence.gml` and adds four node attributes in-place. **Requires `bill_affiliation_network.py` to have been run first.**

| Attribute | Source | Description |
|---|---|---|
| `num_bills` | `opensecrets_lda_reports.csv` | Unique bills lobbied per firm (post prevalence filter) |
| `bill_aff_community` | `data/communities/communities_affiliation.csv` | Leiden community label from the bill affiliation network |
| `within_comm_net_str` | Computed from GML edges | Net RBO strength on directed (balanced=0) edges to same-community peers |
| `within_comm_net_inf` | Computed from GML edges | Net first-mover count on directed edges to same-community peers |

Uses `-1` sentinel for firms absent from source datasets. `within_comm_net_*` mirrors the global `net_strength`/`net_influence` convention but restricted to same-community edges.

### Gephi GEXF Export (`src/gephi_style_export.py`)

Produces a filtered, colored GEXF from the enriched GML. Three filtering steps applied in order:
1. Remove all `balanced=1` edges (arbitrary canonical direction, no analytical meaning).
2. Remove directed edges below the median RBO weight of the remaining directed-only edge set.
3. Remove nodes with both recomputed `in_strength` and `out_strength` = 0 (isolated after above steps).

Node colors are recomputed using a red→yellow→green diverging colormap centered at 0 on `net_strength` (not `net_influence`). Written as `viz:color` for automatic Gephi loading. All existing node and edge attributes are preserved as GEXF `attvalues`.

Output: `visualizations/gexf/rbo_directed_influence.gexf`.

---

## 7. Utility Modules (`src/utils/`)

### `data_loading.py`

- `load_bills_data(path)` — loads reports CSV, validates columns, drops null `bill_number` rows.
- `load_issues_data(path)` — loads issues CSV, validates columns.
- `load_lobby_firm_data(path)` — loads reports CSV, filters to external registrants only.
- `check_columns(df, required, path)` — raises `ValueError` on missing columns.

### `similarity.py`

- `aggregate_per_firm_bill(df)` — groups by `(fortune_name, bill_number)`, sums `amount_allocated`.
- `compute_zero_budget_fracs(df)` — adds `frac = amount_allocated / total_budget`; validates fracs sum to 1.0±0.001 per firm.
- `rbo_score(l1, l2, p)` — truncated min RBO estimate; returns float in [0,1].
- `build_ranked_lists(df, top_bills)` — returns `{firm: [bill, ...]}` sorted by spend descending.
- `build_frac_matrix(df)` — returns (firms × bills) pivot for cosine similarity.

### `filtering.py`

- `filter_bills_by_prevalence(df, max_df, unit_col)` — removes rows where unique firm count per bill > `max_df`. Does NOT recompute budget fracs.
- `prevalence_summary(df, unit_col, thresholds)` — diagnostic: prints removal counts at multiple thresholds.

### `network_building.py`

- `build_graph(edge_df)` — undirected NetworkX graph from edge list.
- `build_graph_with_attrs(edge_df, weight_col)` — all numeric columns written as edge attributes (used by composite network).
- `write_gml_with_communities(G, partition, gml_path, node_attrs)` — writes GML with Leiden community and optional centrality attributes. None values → `-1.0` or `"unknown"` (GML type constraints).
- `_cent_df_to_attrs(cent_df)` — converts centrality DataFrame to `node_attrs` dict.
- `top_k_subgraph(G, k)` — subgraph of top-k nodes by weighted degree.

### `community.py`

- `networkx_to_igraph(G_nx, weight_attr)` — converts for leidenalg compatibility; preserves node names as `vs["name"]`.
- `detect_communities(G_nx, resolution, seed, n_iterations)` — returns `(partition, modularity, summary)`.
- `sweep_resolution(G_nx, resolutions)` — runs Leiden at multiple γ values; useful for calibration.
- `print_community_summary(summary, partition, G_nx)` — prints per-community size, density, and members.

### `centrality.py`

- `compute_katz_centrality(G, weight_attr)` — `α = 0.85/λ_max`; fallback chain to `0.50/λ_max` then weighted degree.
- `compute_centralities(G)` — degree, betweenness, closeness, eigenvector (global).
- `compute_within_community_eigenvector(G, partition)` — community-scoped eigenvector.
- `compute_within_community_zscore(G, partition)` — Guimerà-Amaral z-score.
- `compute_participation_coefficient(G, partition)` — Guimerà-Amaral P.
- `classify_ga_role(z, P)` — maps (z, P) to one of 7 GA roles.
- `compute_community_centralities(G, partition)` — runs all above; returns unified DataFrame.

### `visualization.py`

- `plot_circular(H, title, path)` — circular layout PNG; node size ∝ weighted degree; edge width ∝ weight.
- `plot_directed_circular(G, title, path, top_k)` — directed circular plot; node color from `net_influence`; edges curved to separate A→B and B→A.

---

## 8. Validation Scripts (`src/validations/`)

Numbered scripts with increasing specificity. All run from `src/`.

| Script | Tests |
|---|---|
| `01_extraction_audit.py` | Validates bill-expanded CSV structure; quantifies duplication (multiple rows per (firm, bill)) |
| `02_inflation_diagnosis.py` | Shows Cartesian product inflation before/after deduplication fix |
| `03_sparsity_analysis.py` | Null model: expected co-lobbying pairs under random bill selection vs observed; ~27× above chance |
| `04_mega_bill_diagnosis.py` | Prevalence distribution; modularity at various `MAX_BILL_DF` thresholds |
| `05_ind_filter_validation.py` | Confirms `ind='y'` filter removes superseded originals and double-count subsidiaries correctly |
| `06_rbo_cosine_unit_tests.py` | Unit tests for `rbo_score()` and cosine helpers |
| `07_composite_network_validation.py` | Validates composite formula, centrality consistency, and Katz fallback |
| `08_rbo_p_calibration.py` | Plots RBO sensitivity to `p` vs empirical spend concentration; calibrates `p=0.85` |
| `10_rbo_directed_influence_validation.py` | Validates directed influence edge integrity, balanced edge consistency, and node attribute accounting |
| `11_mediated_adoption_validation.py` | 9-section validation report for affiliation-mediated adoption: mediation rates, lag tests, broker identification, alignment test |

Outputs written to `src/validations/outputs/`.

---

## 9. Key Parameters

These are the parameters most likely to affect results. All are set as constants at the top of each script (not as CLI arguments).

| Parameter | Location | Default | Effect |
|---|---|---|---|
| `MAX_BILL_DF` | `config.py` | `50` | Mega-bill filter. Set to `None` to disable. |
| `RBO_P` | `rbo_directed_influence.py`, `rbo_similarity_network.py` | `0.85` | Top-weight decay. Higher p = more weight on rank-1 bill. Range (0,1). |
| `TOP_BILLS` | `rbo_directed_influence.py`, `rbo_similarity_network.py` | `30` | Bills per firm included in ranking and RBO. |
| `LEIDEN_RESOLUTION` | each network script | varies (0.75–1.0) | γ parameter. Higher = more/smaller communities. |
| `TOP_K` | each network script | `20` | Nodes in PNG visualization. |
| `WRITE_GML` | each network script | `True` | Toggle GML output. |

---

## 10. Output Files Reference

### Data (`data/`)

| File | Producer | Description |
|---|---|---|
| `opensecrets_lda_reports.csv` | `opensecrets_extraction.py` | Primary: one row per (report, bill) |
| `opensecrets_lda_issues.csv` | `opensecrets_extraction.py` | One row per (report, issue_code) |
| `rbo_directed_influence.csv` | `rbo_directed_influence.py` | Directed influence edge list |
| `ranked_bill_lists.csv` | `rbo_directed_influence.py` | Per-firm top-30 bill rankings |
| `network_edges/affiliation_edges.csv` | `bill_affiliation_network.py` | Shared-bill edge list |
| `network_edges/rbo_edges.csv` | `rbo_similarity_network.py` | RBO similarity edges |
| `network_edges/cosine_edges.csv` | `cosine_similarity_network.py` | Cosine similarity edges |
| `network_edges/composite_edges.csv` | `composite_similarity_network.py` | Composite edges |
| `network_edges/lobbyist_affiliation_edges.csv` | `lobbyist_affiliation_network.py` | Shared lobbyist edges |
| `communities/communities_*.csv` | each network script | Leiden community assignments |
| `centralities/centrality_*.csv` | each network script | Centrality measures |
| `affiliation_mediated_adoption.csv` | `affiliation_mediated_adoption.py` | 8,173 rows: one per (RBO edge, shared bill) with bill-level and network-level mediation flags |
| `rbo_edges_enriched.csv` | `affiliation_mediated_adoption.py` | 3,712 rows: RBO edge list joined with directed-bill mediation rates and network connectivity |

### Visualizations (`visualizations/`)

| File | Description |
|---|---|
| `gml/rbo_directed_influence.gml` | Primary DiGraph GML for Gephi (enriched in-place by `enrich_directed_gml.py`) |
| `gexf/rbo_directed_influence.gexf` | Filtered, colored GEXF for Gephi |
| `gml/*.gml` | GML for each undirected network |
| `png/*.png` | Top-20 subgraph plots |

---

## 11. Affiliation-Mediated Adoption Analysis

Tests whether directed bill-adoption pairs in the RBO network are explained by shared lobbyists or lobbying firms — the proposed proximate transmission mechanism.

### Scripts

| Script | Role |
|---|---|
| `src/affiliation_mediated_adoption.py` | Core data construction; produces bill-level and edge-level outputs |
| `src/validations/11_mediated_adoption_validation.py` | 9-section statistical validation report |

Run `affiliation_mediated_adoption.py` first; `11_mediated_adoption_validation.py` reads its outputs.

### Pipeline (`affiliation_mediated_adoption.py`)

**Inputs:** `data/opensecrets_lda_reports.csv`, `data/rbo_directed_influence.csv`, `data/ranked_bill_lists.csv`, `data/network_edges/lobbyist_affiliation_edges.csv`

**Key constants:**

| Constant | Default | Description |
|---|---|---|
| `INCLUDE_BALANCED` | `True` | Include balanced (tied) RBO edges as a comparison group |
| `FIRM_EXTERNAL_ONLY` | `True` | Firm channel restricted to non-self-filer registrants |
| `LOB_AFFIL_EDGES` | `data/network_edges/lobbyist_affiliation_edges.csv` | Pre-computed lobbyist adjacency |

**Lookups built:**

- `build_first_quarter_lookup(reports)` → `{(firm, bill): min_quarter}` across all 8 quarters
- `build_first_q_registrants(reports, external_only)` → `{(firm, bill): set(registrant)}` for first-quarter reports only
- `build_first_q_lobbyists(reports)` → `{(firm, bill): set(lobbyist_name)}` for first-quarter reports; pipe-separated `lobbyists` column exploded
- `build_top_bills(ranked)` → `{company: set(bill_number)}` from `ranked_bill_lists.csv`
- `build_network_adjacency(reports, external_only)` → `(lob_adj, firm_adj)` undirected adjacency sets; lobbyist adjacency from pre-computed edges file, firm adjacency built inline from reports

**Core loop (`analyze_edges`):** For each RBO edge (A, B), iterates over shared top-30 bills. Per bill: assigns leader/follower by first-quarter, computes lag, intersects first-quarter registrant and lobbyist sets for bill-level mediation, checks pair membership in network adjacency sets for network-level connectivity.

**Aggregation (`build_edge_summary`):** Groups bill-level records by (rbo_source, rbo_target) on directed bills only (lag > 0); computes directed_bills, per-channel mediation counts, mediation rates, mean lag. Merges back to original RBO edge list on (source, target).

### Output Columns

**`affiliation_mediated_adoption.csv`** (one row per (RBO edge, shared bill)):

| Column | Description |
|---|---|
| `rbo_source`, `rbo_target` | RBO edge endpoints |
| `rbo_balanced` | 0 = directed RBO edge; 1 = balanced |
| `bill` | bill number |
| `leader`, `follower` | bill-level first/second adopter; None if tied quarter |
| `q_leader`, `q_follower` | first-adoption quarters (1–8) |
| `lag_quarters` | q_follower − q_leader; 0 = same quarter |
| `is_bill_directed` | True if lag > 0 |
| `shared_lobbyist_count`, `shared_firm_count` | bill-level shared intermediary counts |
| `shared_lobbyists`, `shared_firms` | pipe-separated names |
| `is_lobbyist_mediated`, `is_firm_mediated`, `is_any_mediated` | bill-level mediation flags |
| `net_lob_connected`, `net_firm_connected`, `net_any_connected` | network-level connectivity flags |

**`rbo_edges_enriched.csv`**: all original `rbo_directed_influence.csv` columns plus `directed_bills`, `lobbyist_mediated`, `firm_mediated`, `any_mediated`, `mean_lag_quarters`, `lobbyist_mediation_rate`, `firm_mediation_rate`, `any_mediation_rate`, `net_lob_connected`, `net_firm_connected`, `net_any_connected`.

### Key Empirical Findings

- **Bill-level mediation: 0.2%** (7 / 3,184 directed bill-adoption pairs)
- **Network-level connectivity: 0.6%** (20 / 3,184)
- **Alignment: 100%** of mediated bills have the bill-level first-adopter as the RBO source (p = 0.008)
- Dominant mediated cases: United Technologies → Raytheon (merger context, 6 bills) and LOEWS → ECOLAB (1 bill via Ogilvy Government Relations)
- The RBO influence signal is not primarily explained by direct shared-affiliation channels

### Methodology Reference

See `docs/design_decisions.md §24` for full design rationale, alternatives considered, and detailed empirical findings. See `docs/affiliation_mediated_adoption_summary.md` for the developer-facing summary.

---

## 12. Cross-Congressional Stability Analysis

Tests whether RBO directed influence edges are stable in direction and magnitude across seven consecutive congressional sessions (111th–117th, 2009–2022). Temporal stability is a necessary condition for causal interpretation.

### Scripts

| Script | Role |
|---|---|
| `src/multi_congress_pipeline.py` | Extraction + RBO directed influence for each congress; writes per-congress outputs to `data/congress/{num}/` and GML/PNG to `visualizations/` |
| `src/cross_congressional_stability.py` | Four stability analyses on the 135-firm stable set; writes `outputs/cross_congressional/` |

### Prerequisites

1. Expand `data/manual_opensecrets_name_mapping.json` to include CRP name variants for each congress era (firms may have filed under predecessor names). This is done manually. Pre-111th congresses use semi-annual HLOGA filing codes incompatible with quarterly assignment and are excluded.
2. Run `multi_congress_pipeline.py`; outputs are written to `data/congress/{111..117}/`. Set `SKIP_EXISTING = True` to skip already-completed congresses.

### Pipeline (`multi_congress_pipeline.py`)

Parameterizes extraction and RBO steps by congress number. Key parameters at top of file:

| Parameter | Default | Description |
|---|---|---|
| `CONGRESSES` | `[111, 112, 113, 114, 115, 116, 117]` | Congressional sessions to process |
| `SKIP_EXISTING` | `True` | Skip if `rbo_directed_influence.csv` already exists |
| `RBO_P` | `0.85` | RBO p-parameter (matches 116th calibration) |
| `TOP_BILLS` | `30` | Top-K bills per firm ranking |

**Congress year formula:** `start = 2009 + 2 * (congress_num - 111)`. Quarter assignment: year1 Q1–4 → quarters 1–4; year2 Q1–4 → quarters 5–8.

**Per-congress outputs (`data/congress/{num}/`):**
- `opensecrets_lda_reports.csv` — one row per (report, bill)
- `opensecrets_lda_issues.csv` — one row per (report, issue_code)
- `ranked_bill_lists.csv` — per-firm top-30 bill rankings
- `rbo_directed_influence.csv` — directed edge list with RBO weights and temporal direction
- `node_attributes.csv` — per-firm net_influence, net_strength, total_firsts, total_losses

**Visualization outputs:**
- `visualizations/gml/rbo_directed_influence_{num}.gml` — GML graph for Gephi
- `visualizations/png/rbo_directed_influence_{num}.png` — directed circular plot

### Stability Analysis (`cross_congressional_stability.py`)

Four analyses run on the 135 firms present in all seven congresses:

**Analysis 1 — Direction Consistency**

For each canonical pair (firm_a < firm_b alphabetically) with ≥2 directed sessions: consistency = max(n_a_leads, n_b_leads) / n_directed_sessions. Binomial test per pair under H0: 50/50 coin flip each session. Individual tests are underpowered (most pairs appear in 2–5 sessions); the aggregate majority-direction rate (73.9% observed vs. 50% null) is the primary evidence.

**Analysis 2 — Magnitude Stability (net_temporal)**

Pairwise Spearman ρ on net_temporal scores for all stable-set pairs sharing edges in both congresses. Kendall's W across all seven sessions is computed but degenerates with large n (3,291 pairs) and NaN imputation; individual Spearman values are reliable.

**Analysis 3 — Firm Rank Stability (net_influence)**

Adjacent-congress Spearman ρ and Kendall's W on firm-level net_influence ranks across the 135-firm stable set.

**Analysis 4 — Firm net_strength Rank Stability**

Mirrors Analysis 3 using net_strength (RBO-weighted directed score) instead of net_influence (raw win/loss count). `run_firm_stability` is parameterized by `metric` and called once for each. The figure uses a 2×2 grid: direction histogram (top-left), temporal Spearman heatmap (top-right), net_influence bar chart (bottom-left), net_strength bar chart (bottom-right).

### Key Empirical Findings

- **Stable set:** 135 firms (all 7 congresses); 6,783 canonical pairs; 277 in all 7 sessions.
- **Direction:** Mean consistency 0.770; 73.9% majority-direction; only 2.4% of pairs reach individual binomial significance (most significant: BALL/IBM, 5 directed sessions, p=0.031).
- **Magnitude:** Spearman ρ range 0.037–0.218; highest: 116–117 (ρ=0.218). Kendall's W=0.125 is degenerate (large n, NaN imputation) and should be discounted.
- **net_influence ranks (Analysis 3):** 5 of 6 adjacent-congress ρ significant; range 0.134–0.310; Kendall's W=0.343 (p<0.0001). Top persistent influencers: Lockheed Martin (+72.1), IBM (+59.4), Xcel Energy (+53.6), CMS Energy (+46.4), Duke Energy (+45.4). Top persistent followers: Consolidated Edison (−55.1), Ameren (−48.3), Centerpoint Energy (−43.0).
- **net_strength ranks (Analysis 4):** 3 of 6 adjacent-congress ρ significant (114–115, 115–116, 116–117); Kendall's W=0.273 (p<0.0001). Top high-strength firms: Xcel Energy (2.214), Lockheed Martin (2.054), Duke Energy (1.803). Top low-strength firms: Ally Financial (−1.108), Centerpoint Energy (−0.950), Cisco Systems (−0.929).

### Methodology Reference

See `docs/design_decisions.md §28` for full design rationale and empirical results. See `outputs/cross_congressional/cross_congressional_stability.docx` for the summary document.
