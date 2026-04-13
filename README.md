# Fortune 500 Lobbying Networks

Analysis of corporate lobbying networks among Fortune 500 firms during the 116th Congress (2019–2020), using OpenSecrets CRP bulk data for all network layers.

## Conceptual Framework: Influence as Agenda-Setting

The research distinguishes three complementary forms of influence, each captured by a different analytical layer:

| Form | Operationalization | Layer |
|---|---|---|
| **Structural influence** | Centrality position in the undirected similarity network — who is architecturally positioned to propagate an agenda | Undirected RBO / composite networks + centrality measures |
| **Observed influence** | Temporal first-mover precedence — who consistently lobbied shared bills before whom across the full Congress | RBO directed influence network (`rbo_directed_influence.py`, §21) |
| **Equilibrium influence** | Nash-stable adoption configurations under the Irfan-Ortiz Linear Influence Game given network topology and firm thresholds | PSNE solver (`influence_simple.cpp`) |

**Agenda-setting as the core mechanism:** A highly influential firm shapes another firm's lobbying priorities — its ranked portfolio of lobbied bills — to more closely match its own. Agenda-setting is inferred from temporal bill-adoption patterns: if Firm A consistently lobbies shared bills before Firm B, A is the agenda-setter. Priority rankings are defined by each firm's spend-fraction allocation across its lobbied bills.

**The RBO directed influence network** (§21 in `design_decisions.md`, `src/rbo_directed_influence.py`) is the primary instrument for observed influence. Edge weight = RBO similarity (priority ranking overlap, p=0.85); edge direction = global first-mover across the 116th Congress. The two signals are separated by design: RBO answers "how aligned are these firms?" and the arrow answers "who set whose agenda?"

**The transmission mechanism** is lobbyist networks (Carpenter et al., 1998): shared human lobbyists carry bill-priority information between firms. The directed network captures the *outcome* of this transmission, not the channel itself. The lobbyist affiliation network (Layer 2) is the empirical proxy for the channel.

**Connection to the threshold model:** The first-mover timing data that determines edge direction also provides the revealed thresholds θ̂_i for the Granovetter-Watts threshold model: a firm's peer pressure at the moment of bill adoption is its estimated threshold. These estimates feed heterogeneous thresholds into the PSNE solver, closing the loop between dynamic diffusion and static equilibrium analysis. See `Threshold_Model_Roadmap.docx` for the full implementation plan.

---

## Overview

This project builds and analyzes a set of company-to-company networks where edges represent different dimensions of lobbying similarity or shared activity. Nodes are Fortune 500 corporations. The project is organized into two layers:

**Layer 1** uses OpenSecrets bill and issue data to construct five networks capturing different aspects of co-lobbying behavior, plus a composite network that combines three of them.

**Layer 2** uses OpenSecrets data to build a lobbyist affiliation network where edge weight is the number of unique human lobbyists two firms share.

Both layers draw from a single extraction pipeline (`opensecrets_extraction.py`) that produces clean, deduplicated Fortune 500 lobbying records. All networks use Leiden community detection to identify lobbying coalitions and compute a suite of centrality measures to characterize each firm's structural role.

## Data Source

All data comes from OpenSecrets CRP bulk lobbying tables (Senate Office of Public Records, standardized by CRP):

| File | Description |
|---|---|
| `lob_lobbying.txt` | Report-level filings: registrant, client, amount, self/external flag, validity flag |
| `lob_lobbyist.txt` | Named lobbyists per report |
| `lob_issue.txt` | Issue codes per report |
| `lob_bills.txt` | Bills per issue entry |

**Name mapping:** `data/manual_opensecrets_name_mapping.json` — manually curated flat JSON `{canonical_Fortune500_name: [CRP_variation, ...]}` mapping ~500 firms to their OpenSecrets CRP name variants.

## Extraction Pipeline

`opensecrets_extraction.py` is the single source of truth for Fortune 500 lobbying records. It applies two filters before producing output:

1. **`ind='y'` validity filter** (OpenSecrets Data User Guide p.13): retains only records OpenSecrets marks as valid and countable. This removes superseded originals (when a quarterly amendment exists, the original carries `ind=''`), double-count subsidiary records, and no-activity filings that OpenSecrets explicitly excludes from totals. See `design_decisions.md §16` for full rationale.

2. **Active report filter**: of the valid `ind='y'` reports, only those with ≥ 1 named lobbyist are retained. Reports with no lobbyists are retainer/no-activity filings with no issue codes or bills.

**Produces:**
- `opensecrets_lda_reports.csv` — one row per (active report, bill); `amount_allocated = amount / n_bills_in_report`
- `opensecrets_lda_issues.csv` — one row per (active report, issue_code); `amount_allocated = amount / n_issue_codes_in_report`
- `lobbyist_client_116_opensecrets.csv` — archived deduplicated lobbyist-firm pairs

## Layer 1 — Bill and Issue Networks

Five networks, each capturing a different signal:

| Network | Script | Edge weight |
|---|---|---|
| Bill Affiliation | `bill_affiliation_network.py` | Shared bill count (mega-bill filtered) |
| Cosine Similarity | `cosine_similarity_network.py` | Cosine similarity of lobbying budget portfolios |
| RBO Similarity | `rbo_similarity_network.py` | Rank-Biased Overlap on bill priority rankings |
| Issue Similarity | `issue_similarity_network.py` | Cosine similarity of issue-code portfolios |
| Lobby Firm Affiliation | `lobby_firm_affiliation_network.py` | Shared external lobbying firm registrant count |

**Composite network** (`composite_similarity_network.py`): multiplies the three bill-level signals — `affil_norm × cosine × rbo` — as an inner join. An edge exists only when all three signals are nonzero. Run `composite_community_comparison.py` afterward to compare partitions across all four bill-level networks using NMI, ARI, and firm-level consensus stability.

### Layer 1 Files

```
src/
  opensecrets_extraction.py           Extract + deduplicate Fortune 500 LDA reports
  bill_affiliation_network.py         Layer 1, Network 1
  cosine_similarity_network.py        Layer 1, Network 2
  rbo_similarity_network.py           Layer 1, Network 3
  issue_similarity_network.py         Layer 1, Network 4
  lobby_firm_affiliation_network.py   Layer 1, Network 5
  composite_similarity_network.py     Layer 1, Composite
  composite_community_comparison.py   4-way community comparison
  rbo_quarterly_networks.py           Quarterly RBO networks + temporal evolution (§19)
  rbo_directed_influence.py           Congress-wide RBO directed influence network (§21)
  enrich_directed_gml.py              Add enriched node attrs to rbo_directed_influence.gml (§22)
  build_bill_company_matrix.py        Bill-company incidence matrix (§17)
  config.py                           Paths and shared constants
  utils/
    data_loading.py                   CSV loaders with validation
    filtering.py                      Prevalence filtering (MAX_BILL_DF)
    similarity.py                     RBO and cosine helpers
    network_building.py               Graph construction and GML export
    centrality.py                     All centrality computations
    community.py                      Leiden detection and resolution sweep
    visualization.py                  Circular layout plots
  validations/
    design_decisions.md               Methodology documentation
    01_extraction_audit.py            Audit bill-expanded CSV structure
    02_inflation_diagnosis.py         Cartesian product inflation check
    03_sparsity_analysis.py           Null model co-lobbying signal
    04_mega_bill_diagnosis.py         Mega-bill prevalence distribution
    05_ind_filter_validation.py       Validate ind='y' deduplication filter
    06_rbo_cosine_unit_tests.py       RBO and cosine helper unit tests
    07_composite_network_validation.py Composite formula and centrality tests
    08_rbo_p_calibration.py            RBO p-parameter calibration vs. empirical spend concentration
    09_directed_influence_validation.py Directed influence network integrity checks (§20) [archived]
    10_rbo_directed_influence_validation.py Congress-wide RBO influence network validation (§21)
```

Data files produced (in `data/`): `opensecrets_lda_reports.csv` (includes `quarter` column 1–8), `opensecrets_lda_issues.csv`, `affiliation_edges.csv`, `cosine_edges.csv`, `rbo_edges.csv`, `rbo_edges_q{1..8}.csv`, `composite_edges.csv`, `communities_*.csv`, `communities_rbo_q{1..8}.csv`, `centrality_*.csv`, `centrality_rbo_q{1..8}.csv`, `community_comparison_composite.csv`, `nmi_ari_matrix.csv`, `bill_company_matrix.csv`, `bill_index.csv`, `company_index.csv`, `rbo_quarterly_stats.csv`, `rbo_quarterly_nmi_ari.csv`, `rbo_quarterly_spearman.csv`, `directed_influence_q{1..8}.csv`, `directed_influence_agg.csv`, `rbo_directed_influence.csv`, `ranked_bill_lists.csv`.

Visualization files produced (in `visualizations/`): `gml/*.gml` (Gephi-compatible, with community and centrality attributes), `png/*.png` (top-K subgraph plots).

## Layer 2 — Lobbyist Affiliation Network

Uses OpenSecrets bulk data (`lob_lobbying.txt` + `lob_lobbyist.txt`), extracted by the same `opensecrets_extraction.py` pipeline.

Edge weight = number of unique human lobbyists retained by both Fortune 500 firms during the 116th Congress. Lobbyist-client relationships are derived inline from the pipe-separated `lobbyists` column in `opensecrets_lda_reports.csv` and deduplicated so each lobbyist counts once per firm regardless of how many quarterly reports were filed.

### Layer 2 Files

```
src/
  opensecrets_extraction.py       Shared extraction pipeline (also feeds Layer 1)
  build_opensecrets_mapping.py    Build/enrich the OpenSecrets Fortune 500 name mapping
  lobbyist_affiliation_network.py Layer 2 network construction + analysis
```

Data files produced (in `data/`): `lobbyist_affiliation_edges.csv`, `communities_lobbyist_affiliation.csv`, `centrality_lobbyist_affiliation.csv`.

Name mapping files (in `data/`):
- `manual_opensecrets_name_mapping.json` — manually curated; active mapping used by `opensecrets_extraction.py`
- `opensecrets_name_mapping.json` — archived; auto-generated by `build_opensecrets_mapping.py`

Archived LobbyView files (in `data/archive/`):
- `lobbyview_lda_reports.csv`, `lobbyview_lda_issues.csv`, `lobbyview_name_mapping.json`
- `lobbyist_client_116_opensecrets.csv` — archived lobbyist-firm pairs CSV (Layer 2 now reads from `opensecrets_lda_reports.csv` directly)

Archived source scripts (in `src/archive/`):
- `lobbyview_extraction.py`, `build_lobbyview_mapping.py`

## How to Run

All scripts run from the `src/` directory. Parameters (resolution, top-K, etc.) are constants at the top of each script — edit them directly rather than using command-line flags.

### Step 1: Extract Fortune 500 lobbying data (required first)

```bash
cd src
python opensecrets_extraction.py
```

This produces `opensecrets_lda_reports.csv` and `opensecrets_lda_issues.csv`, which all Layer 1 and Layer 2 network scripts consume.

### Step 2: Build Layer 1 networks

```bash
python bill_affiliation_network.py
python cosine_similarity_network.py
python rbo_similarity_network.py
python issue_similarity_network.py
python lobby_firm_affiliation_network.py

# Composite (requires bill, cosine, and RBO edge CSVs first)
python composite_similarity_network.py

# 4-way community comparison
python composite_community_comparison.py

# Quarterly RBO networks + temporal evolution analysis
python rbo_quarterly_networks.py

# Congress-wide RBO directed influence network (standalone; no prerequisites)
python rbo_directed_influence.py

# Enrich GML with additional node attrs (requires bill_affiliation_network.py first)
python enrich_directed_gml.py
```

### Step 3: Build Layer 2 network

```bash
python lobbyist_affiliation_network.py
```

### Validations

```bash
cd src
python validations/01_extraction_audit.py
python validations/02_inflation_diagnosis.py
python validations/03_sparsity_analysis.py
python validations/04_mega_bill_diagnosis.py
python validations/05_ind_filter_validation.py
python validations/06_rbo_cosine_unit_tests.py
python validations/07_composite_network_validation.py
python validations/08_rbo_p_calibration.py
python validations/09_directed_influence_validation.py   # archived
python validations/10_congress_influence_validation.py
```

Outputs are written to `src/validations/outputs/`.

## Key Design Decisions

All significant methodological choices are documented in `src/validations/design_decisions.md`. Key decisions include:

- **§0** — Influence operationalized as agenda-setting via bill-priority rankings; temporal precedence as the observable signal; lobbyist networks as the proximate mechanism (out of scope)
- **§1** — One row per bill per report; aggregate at network construction time
- **§4** — Mega-bill filtering (`MAX_BILL_DF = 50`) to prevent modularity collapse
- **§6** — Leiden community detection with resolution γ = 1.0
- **§12** — Composite network: multiplicative `affil_norm × cosine × rbo`
- **§15** — No lower threshold on edge formation (`DEFAULT_MIN_WEIGHT = 0.0`)
- **§16** — `ind='y'` validity filter replacing the prior Self-field type filter
- **§18** — RBO parameter recalibration: `p=0.85`, `top_bills=30` (empirically grounded in observed Fortune 500 spend concentration)
- **§19** — Quarterly RBO networks: independent windows, quarter assignment from `report_type`, temporal evolution via NMI+ARI, metric trajectories, and Spearman ρ of PageRank
- **§20** — Directed influence network: temporal bill-adoption precedence scores a directed edge A→B where A lobbied shared bills before B; causal window restricted to Q1..q; ties skipped; net-weight directed edges
- **§21** — Congress-wide directed RBO network: edge weight = RBO similarity, direction = global first-mover (no causal windowing), no double counting per bill, balanced pairs → bidirectional edges, node `net_influence` = count-based net first-mover score across all pairings
- **§22** — Enriched node attributes on the directed GML: `num_bills` (post-filter bill count), `bill_aff_community` (Leiden label from bill affiliation network), `within_comm_net_str` and `within_comm_net_inf` (directed, non-balanced edges to same-community peers only)

## Dependencies

```
pandas numpy networkx python-igraph leidenalg scikit-learn scipy matplotlib requests beautifulsoup4
```

Install: `pip install pandas numpy networkx python-igraph leidenalg scikit-learn scipy matplotlib requests beautifulsoup4`
