# Fortune 500 Lobbying Networks

Analysis of corporate lobbying networks among Fortune 500 firms during the 116th Congress (2019–2020), using OpenSecrets CRP bulk data for all network layers.

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
```

Data files produced (in `data/`): `opensecrets_lda_reports.csv`, `opensecrets_lda_issues.csv`, `affiliation_edges.csv`, `cosine_edges.csv`, `rbo_edges.csv`, `composite_edges.csv`, `communities_*.csv`, `centrality_*.csv`, `community_comparison_composite.csv`, `nmi_ari_matrix.csv`.

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
```

Outputs are written to `src/validations/outputs/`.

## Key Design Decisions

All significant methodological choices are documented in `src/validations/design_decisions.md`. Key decisions include:

- **§1** — One row per bill per report; aggregate at network construction time
- **§4** — Mega-bill filtering (`MAX_BILL_DF = 50`) to prevent modularity collapse
- **§6** — Leiden community detection with resolution γ = 1.0
- **§12** — Composite network: multiplicative `affil_norm × cosine × rbo`
- **§15** — No lower threshold on edge formation (`DEFAULT_MIN_WEIGHT = 0.0`)
- **§16** — `ind='y'` validity filter replacing the prior Self-field type filter

## Dependencies

```
pandas numpy networkx python-igraph leidenalg scikit-learn scipy matplotlib requests beautifulsoup4
```

Install: `pip install pandas numpy networkx python-igraph leidenalg scikit-learn scipy matplotlib requests beautifulsoup4`
