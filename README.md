# Corporate Lobbying Networks

Analysis of corporate lobbying behavior among Fortune 500 firms during the 116th Congress (2019–2020), using OpenSecrets CRP bulk data. The project constructs several company-to-company networks and uses the **RBO directed influence network** as its primary analytical instrument.

---

## Core Idea

Two companies are "similar" if they lobby the same bills and rank those bills similarly by spend. A firm "influences" another if it consistently lobbied their shared bills *first* — i.e., it set the other firm's legislative agenda before the follower adopted those bills.

This agenda-setting signal is operationalized via temporal bill-adoption precedence. Firms are ranked by the fraction of their total lobbying budget allocated to each bill. If Firm A lobbied a bill in Q1 and Firm B adopted it in Q3, A gets a first-mover credit. Aggregating these credits across all shared bills for every firm pair yields the directed influence network.

The proximate mechanism through which agenda-setting propagates is shared human lobbyists — lobbyists who work for both firms create information bridges that transmit bill priorities from influencer to follower.

---

## Data

All data comes from **OpenSecrets CRP bulk lobbying tables** for the 116th Congress.

| File | Contents |
|---|---|
| `lob_lobbying.txt` | Report-level filings: registrant, client, amount |
| `lob_lobbyist.txt` | Named lobbyists per report |
| `lob_issue.txt` | Issue codes per report |
| `lob_bills.txt` | Bills per issue entry |

**Name mapping:** `data/manual_opensecrets_name_mapping.json` — manually curated JSON mapping ~500 Fortune 500 firm names to their OpenSecrets CRP name variants.

---

## Extraction Pipeline

`src/opensecrets_extraction.py` is the single entry point for all data. It filters to `ind='y'` records (OpenSecrets' own validity flag that excludes superseded originals and double-counted subsidiaries) then retains only reports with ≥1 named lobbyist.

**Run first:**
```bash
cd src
python opensecrets_extraction.py
```

**Produces:**
- `data/opensecrets_lda_reports.csv` — one row per (report, bill); `amount_allocated = amount / n_bills`
- `data/opensecrets_lda_issues.csv` — one row per (report, issue_code); `amount_allocated = amount / n_issue_codes`
- `data/lobbyist_client_116_opensecrets.csv` — deduplicated lobbyist–firm pairs (archived reference)

---

## Networks

### RBO Directed Influence Network *(primary)*

**Script:** `src/rbo_directed_influence.py`

The main analytical output. Each node is a Fortune 500 firm. For every pair of firms that share at least one top-30 bill (by spend):

- **Edge weight** = Rank-Biased Overlap (RBO, p=0.85) of their bill-priority rankings — how aligned are their lobbying agendas?
- **Edge direction** = global first-mover over the full 116th Congress — who set whose agenda?

A directed edge A→B means A lobbied more of their shared top-30 bills before B across the whole congress. The two signals are intentionally separated: RBO answers "how similar?", the arrow answers "who influenced whom?".

Each node carries `net_influence` (total first-mover wins minus losses across all pairings), `net_strength` (RBO-weighted directed score on non-balanced edges), and community color (green = net influencer, red = net follower, gray = neutral).

**Outputs:** `data/rbo_directed_influence.csv`, `data/ranked_bill_lists.csv`, `visualizations/gml/rbo_directed_influence.gml`, `visualizations/png/rbo_directed_influence.png`

**Enrichment:** `src/enrich_directed_gml.py` adds `num_bills`, `bill_aff_community`, `within_comm_net_str`, and `within_comm_net_inf` to the GML node attributes. Requires `bill_affiliation_network.py` to run first.

**Gephi export:** `src/gephi_style_export.py` reads the enriched GML, removes balanced and low-weight edges, recolors nodes by net_strength on a red→yellow→green diverging scale, and writes a Gephi-ready GEXF to `visualizations/gexf/rbo_directed_influence.gexf`.

**Affiliation-mediated adoption:** `src/affiliation_mediated_adoption.py` tests whether directed adoption pairs (A→B, bill) are mediated by shared lobbyists or lobbying firms — the proposed transmission mechanism. Operates at two levels: bill-level co-affiliation (same intermediary on the specific bill's first-quarter reports) and network-level connectivity (any shared intermediary across full portfolios). Produces `data/affiliation_mediated_adoption.csv` and `data/rbo_edges_enriched.csv`. See §24 in `docs/design_decisions.md` for findings.

---

### Supporting Networks

These networks characterize the co-lobbying structure from different angles. All run from `src/` after the extraction step. Each produces edge CSVs, Leiden community assignments, centrality measures, a GML file, and a PNG plot.

| Network | Script | Edge weight |
|---|---|---|
| Bill Affiliation | `bill_affiliation_network.py` | Shared bill count (mega-bill filtered, normalized) |
| RBO Similarity | `rbo_similarity_network.py` | Rank-Biased Overlap on bill-priority rankings |
| Cosine Similarity | `cosine_similarity_network.py` | Cosine similarity of lobbying budget portfolios |
| Issue Similarity | `issue_similarity_network.py` | Cosine similarity of issue-code portfolios |
| Lobby Firm Affiliation | `lobby_firm_affiliation_network.py` | Shared external lobbying firm count |
| Lobbyist Affiliation | `lobbyist_affiliation_network.py` | Shared human lobbyist count |
| Composite | `composite_similarity_network.py` | `affil_norm × cosine × rbo` (inner join) |

**Mega-bill filtering** (`MAX_BILL_DF = 50` in `config.py`): bills lobbied by more than 50 firms — omnibus legislation like the CARES Act — are excluded before computing similarity. These bills create edges between every firm regardless of strategic alignment, analogous to stop-word removal in text analysis.

**Community detection:** all networks use the Leiden algorithm (`γ = 1.0` default) which guarantees internally connected communities. Run `src/composite_community_comparison.py` to compare partitions across the four bill-level networks using NMI and ARI.

---

## How to Run

All scripts run from the `src/` directory. Parameters (resolution, top-K, RBO p, etc.) are constants at the top of each script.

```bash
cd src

# 1. Extract data (required first)
python opensecrets_extraction.py

# 2. Build networks (any order after extraction)
python bill_affiliation_network.py
python rbo_similarity_network.py
python cosine_similarity_network.py
python issue_similarity_network.py
python lobby_firm_affiliation_network.py
python lobbyist_affiliation_network.py
python composite_similarity_network.py
python composite_community_comparison.py

# 3. Build the directed influence network (standalone; reads opensecrets_lda_reports.csv)
python rbo_directed_influence.py

# 4. Enrich GML with community and bill-count attributes (requires bill_affiliation first)
python enrich_directed_gml.py

# 5. Export Gephi-ready GEXF (requires enrich_directed_gml.py first)
python gephi_style_export.py
```

### Validations

```bash
python validations/01_extraction_audit.py         # audit bill-expanded CSV structure
python validations/02_inflation_diagnosis.py       # cartesian product inflation check
python validations/03_sparsity_analysis.py         # null model co-lobbying signal
python validations/04_mega_bill_diagnosis.py       # mega-bill prevalence distribution
python validations/05_ind_filter_validation.py     # ind='y' deduplication filter
python validations/06_rbo_cosine_unit_tests.py     # RBO and cosine unit tests
python validations/07_composite_network_validation.py
python validations/08_rbo_p_calibration.py         # RBO p-parameter calibration
python validations/10_rbo_directed_influence_validation.py

# 6. Affiliation-mediated adoption (requires rbo_directed_influence.py and lobbyist_affiliation_network.py first)
python affiliation_mediated_adoption.py
python validations/11_mediated_adoption_validation.py
python visualize_affiliation_mediation.py

# 7. Issue RBO similarity network (standalone; reads opensecrets_lda_issues.csv)
python issue_rbo_similarity_network.py

# 8. Channel tests — mechanism testing for directed influence (run after steps 6-7)
python channel_tests/test_channel1_monitoring_capacity.py
python channel_tests/test_channel3_issue_overlap.py

# 9. Cross-congressional stability (111th–117th Congress, 2009–2022)
#    Step 1: expand manual_opensecrets_name_mapping.json to cover CRP names for each
#            congress era (corporate rebrands, predecessor entities). Manual curation only.
#            Pre-111th congresses are excluded (semi-annual HLOGA filing codes incompatible
#            with quarterly assign_quarters).
#    Step 2: run extraction + RBO for all seven congresses (writes to data/congress/{num}/)
#            Also writes GML and PNG to visualizations/gml/ and visualizations/png/
python multi_congress_pipeline.py
#    Step 3: four stability analyses on the 135-firm stable set
#            (direction consistency, magnitude, net_influence ranks, net_strength ranks)
python cross_congressional_stability.py
```

---

## Project Layout

```
src/
  opensecrets_extraction.py       Extraction pipeline (run first)
  config.py                       Paths and shared constants
  rbo_directed_influence.py       Primary: directed influence network
  enrich_directed_gml.py          Add enriched node attrs to directed GML
  gephi_style_export.py           Export filtered GEXF for Gephi
  bill_affiliation_network.py     Shared-bill affiliation network
  rbo_similarity_network.py       RBO similarity network
  cosine_similarity_network.py    Cosine similarity network
  issue_similarity_network.py     Issue-code similarity network
  lobby_firm_affiliation_network.py  Shared lobby firm network
  lobbyist_affiliation_network.py    Shared lobbyist network
  composite_similarity_network.py    Composite (affil × cosine × rbo)
  composite_community_comparison.py  4-way community comparison (NMI/ARI)
  build_bill_company_matrix.py    Bill-company incidence matrix
  build_opensecrets_mapping.py    Auto-generate OpenSecrets name mapping (archived utility)
  utils/
    data_loading.py               CSV loaders with column validation
    filtering.py                  Mega-bill prevalence filtering
    similarity.py                 RBO and cosine helpers
    network_building.py           Graph construction and GML export
    centrality.py                 All centrality computations (Guimerà-Amaral)
    community.py                  Leiden detection and resolution sweep
    visualization.py              Circular layout plots
  validations/
    01_extraction_audit.py        ...
    ...
    10_rbo_directed_influence_validation.py
    11_mediated_adoption_validation.py
  affiliation_mediated_adoption.py       Bill-level affiliation-mediated adoption analysis
  visualize_affiliation_mediation.py     Visualization suite (3 figures) for mediation analysis
  issue_rbo_similarity_network.py        RBO-based issue-code similarity network
  channel_tests/
    __init__.py
    test_channel1_monitoring_capacity.py  Channel 1: capacity gap as driver of directed influence
    test_channel3_issue_overlap.py        Channel 3: issue-space correlated response
  multi_congress_pipeline.py      Per-congress extraction + RBO influence (111th–117th) + GML/PNG
  cross_congressional_stability.py  Direction/magnitude/rank stability across 7 congresses
  archive/                        Archived scripts (not part of active pipeline)
data/
  OpenSecrets/                    Raw CRP bulk files
  network_edges/                  Edge CSVs for all networks
  communities/                    Leiden community assignments
  centralities/                   Centrality measure CSVs
  manual_opensecrets_name_mapping.json   Active Fortune 500 → CRP name mapping
  opensecrets_lda_reports.csv     Primary extraction output (report × bill)
  opensecrets_lda_issues.csv      Issue-code extraction output
  rbo_directed_influence.csv      Directed influence edge list
  ranked_bill_lists.csv           Per-firm top-30 bill rankings
  affiliation_mediated_adoption.csv  Bill-level mediated adoption dataset
  rbo_edges_enriched.csv          RBO edges with mediation rates and network connectivity
  network_edges/issue_rbo_edges.csv  RBO-based issue-code similarity edges
visualizations/
  gml/                            Gephi-compatible GML files (all networks)
  gexf/                           Gephi-ready GEXF (filtered directed network)
  png/                            Network plots (top-20 subgraphs)
  archive/                        Archived visualizations
data/
  congress/
    111/  Per-congress extraction + RBO outputs (opensecrets_lda_reports.csv,
    112/    rbo_directed_influence.csv, node_attributes.csv, ranked_bill_lists.csv,
    113/    opensecrets_lda_issues.csv)
    114/
    115/
    116/
    117/
outputs/
  validation/                         All validation script text outputs + calibration PNG
  channel_tests/                      Channel test results, figures, revolving door note
  cross_congressional/                Stability report (.txt), figure (.png), summary (.docx)
docs/
  design_decisions.md                     Methodology and design decision log (§0–§28)
  DOCUMENTATION.md                        Full reproduction guide
  directed_influence_summary.md           Summary of directed influence network analysis
  affiliation_mediated_adoption_summary.md  Summary of affiliation-mediated adoption analysis
```

---

## Dependencies

```
pandas numpy networkx python-igraph leidenalg scikit-learn scipy matplotlib
```

Install: `pip install pandas numpy networkx python-igraph leidenalg scikit-learn scipy matplotlib`
