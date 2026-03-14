# Fortune 500 Lobbying Networks Analysis

**Project:** Layer 1 network construction and analysis
**Data:** LobbyView, 116th Congress (2019–2021)
**Focus:** Four company-to-company co-lobbying networks

## Overview

This project reconstructs the network of lobbying relationships among Fortune 500 firms during the 116th Congress. We build four distinct networks based on different aspects of lobbying coordination:

1. **Bill Affiliation Network** — shared bill co-lobbying
2. **Bill Similarity Network** — Bray-Curtis portfolio alignment on bills
3. **Issue Similarity Network** — Bray-Curtis portfolio alignment on issue codes
4. **Lobby Firm Affiliation Network** — shared lobbying firm retention

Each network is analyzed using Leiden community detection to identify lobbying coalitions, with centrality metrics to find industry leaders and cross-sector connectors.

## Repository Structure

```
├── src/
│   ├── config.py                                    # Configuration (MAX_BILL_DF, etc.)
│   ├── bill_affiliation_network.py                  # Network 1
│   ├── bill_similarity_network.py                   # Network 2
│   ├── issue_similarity_network.py                  # Network 3
│   ├── lobby_firm_affiliation_network.py            # Network 4
│   ├── utils/
│   │   ├── data_loading.py                          # CSV loaders
│   │   ├── filtering.py                             # Prevalence filtering
│   │   ├── network_building.py                      # Graph construction, GML export
│   │   ├── visualization.py                         # Matplotlib plots
│   │   ├── centrality.py                            # Eigenvector, PageRank, Guimerà-Amaral
│   │   └── community.py                             # Leiden detection, modularity analysis
│   └── validations/
│       ├── 01_extraction_audit.py                   # Raw data structure audit
│       ├── 02_inflation_diagnosis.py                # Cartesian product inflation check
│       ├── 03_sparsity_analysis.py                  # Null model comparison
│       ├── 04_mega_bill_diagnosis.py                # Prevalence filtering justification
│       ├── 05_issue_score_range.py                  # Weight range proof
│       ├── design_decisions.md                      # Full methodology documentation
│       └── outputs/                                 # Validation reports (text files)
│
├── data/
│   ├── LobbyView/                                   # Raw LobbyView data (not committed)
│   ├── fortune500_*.csv                             # Extracted and aggregated data
│   ├── communities_*.csv                            # Community assignments per network
│   └── centrality_*.csv                             # Centrality metrics per network
│
└── visualizations/
    ├── gml/          # GML files (Gephi-compatible, with community attributes)
    ├── png/          # PNG plots (top-K subgraph layouts)
    ├── gephi/        # Gephi project files (optional)
    └── pdf/          # PDF renders (optional)
```

## Networks

### 1. Bill Affiliation Network
- **Source:** `src/bill_affiliation_network.py`
- **Edges:** Firms sharing co-lobbying on bills
- **Weight:** Number of shared bills
- **Interpretation:** Binary co-lobbying presence; ignores portfolio composition
- **GML output:** `visualizations/gml/bill_affiliation_network.gml`

### 2. Bill Similarity Network
- **Source:** `src/bill_similarity_network.py`
- **Edges:** Firms with aligned bill portfolios
- **Weight:** mean(Bray-Curtis per bill) × breadth term (bounded [0,1])
- **Interpretation:** Portfolio alignment weighted by breadth of co-lobbying
- **GML output:** `visualizations/gml/bill_similarity_network.gml`

### 3. Issue Similarity Network
- **Source:** `src/issue_similarity_network.py`
- **Edges:** Firms with aligned issue-code portfolios
- **Weight:** sum(BC per issue) / sqrt(shared_issue_count) (NOT bounded [0,1])
- **Interpretation:** Policy-area alignment across multiple issue domains
- **Note:** Weights can exceed 1.0; see `design_decisions.md §11`
- **GML output:** `visualizations/gml/issue_similarity_network.gml`

### 4. Lobby Firm Affiliation Network
- **Source:** `src/lobby_firm_affiliation_network.py`
- **Edges:** Firms sharing lobbying firm registrants
- **Weight:** Number of shared lobbying firms
- **Interpretation:** Relational access through common representatives
- **GML output:** `visualizations/gml/lobby_firm_affiliation_network.gml`

## How to Run

### Build a network (all CLI options shown)

```bash
# Default: plot top 20 nodes, show centrality for top 10, write GML, run resolution sweep
python bill_affiliation_network.py

# Custom resolution for coarser communities
python bill_affiliation_network.py --resolution 0.5

# Skip expensive operations
python bill_affiliation_network.py --no-gml --no-sweep --top-k 0 --centrality-k 0

# Fast: skip plot and sweep, keep GML
python bill_affiliation_network.py --top-k 0 --no-sweep

# Run with full diagnostics (BC networks only)
python bill_similarity_network.py --diagnostics

# All networks follow the same CLI interface:
python issue_similarity_network.py --top-k 15 --centrality-k 5 --resolution 1.0
```

### Run validation scripts

```bash
# Audit raw data structure (one row per bill per report)
python src/validations/01_extraction_audit.py

# Check for cartesian product inflation in edge counts
python src/validations/02_inflation_diagnosis.py

# Verify above-null co-lobbying signal with null model
python src/validations/03_sparsity_analysis.py

# Justify mega-bill prevalence filtering (MAX_BILL_DF=50)
python src/validations/04_mega_bill_diagnosis.py

# Prove issue similarity weight range (not bounded [0,1])
python src/validations/05_issue_score_range.py
```

All validation outputs are written to `src/validations/outputs/`.

## Key Preprocessing Steps

1. **Aggregation fix (§1):**
   - Raw data: one row per (firm, bill, report) triple
   - Aggregation: `groupby(["client_name","bill_id"]).sum()` for BC; `drop_duplicates()` for affiliation

2. **Cartesian product fix (§2):**
   - Deduplication before clique-building loop
   - Canonical pair ordering: `(a,b)` if a < b, else (b,a)

3. **Zero-budget exclusion (§8):**
   - Firms with $0 total spend excluded from BC computations (would cause NaN)
   - Remain in affiliation networks

4. **Mega-bill filtering (§4):**
   - Bills lobbied by >50 firms excluded (MAX_BILL_DF=50)
   - These 16 bills account for 97.5% of edges and collapse modularity
   - Fractions computed on all bills; pairing loop uses filtered bills only

5. **Prevalence filtering for issues (§4):**
   - Disabled by default (MAX_ISSUE_DF=None)
   - Issue codes are broader than bills; enable if modularity collapses

## Dependencies

```
pandas numpy networkx python-igraph leidenalg scikit-learn scipy matplotlib
```

Install with: `pip install pandas numpy networkx python-igraph leidenalg scikit-learn scipy matplotlib`

## Design Decisions

Full methodology documented in `src/validations/design_decisions.md`:

- **§1:** Data structure (one row per bill per report) and aggregation strategy
- **§2:** Cartesian product inflation fix and canonical pair ordering
- **§3:** Sparsity and above-null signal (27× above chance for median pair)
- **§4:** Mega-bill prevalence filtering and modularity impact
- **§5:** Similarity metric choice (Bray-Curtis + breadth term)
- **§6:** Leiden vs. Louvain, resolution parameter selection
- **§7:** Three-tier centrality framework (within-community, cross-community, global)
- **§8:** Zero-budget firm exclusion
- **§9:** Community comparison (NMI, ARI, Hungarian alignment)
- **§10:** GML export with community node attributes
- **§11:** Issue similarity weight range (unbounded, why sqrt normalization chosen)

## Validation Scripts

All five validation scripts are self-contained, runnable, and document key analytical choices:

- `05_issue_score_range.py`: Mathematical proof + empirical distribution of issue weights
- `04_mega_bill_diagnosis.py`: Distribution of bill prevalence, modularity collapse
- `03_sparsity_analysis.py`: Null model comparison, above-null quantification
- `02_inflation_diagnosis.py`: Side-by-side comparison of buggy vs. fixed edges
- `01_extraction_audit.py`: Raw data duplication audit

Run any script to see both proof and empirical evidence. Outputs saved to `src/validations/outputs/`.

## Output Files

- **GML files:** `visualizations/gml/*.gml` (Gephi-compatible with community node attributes)
- **CSV exports:** `data/communities_*.csv`, `data/centrality_*.csv`
- **PNG plots:** `visualizations/png/*.png` (top-K subgraphs)
- **Validation reports:** `src/validations/outputs/*.txt`

---

**Authors:** Prathit Kurup, Victoria Figueroa
**License:** Internal research project
**Last updated:** 2026-03-12
