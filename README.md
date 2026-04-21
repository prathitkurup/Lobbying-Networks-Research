# Corporate Lobbying Networks

Analysis of corporate lobbying behavior among Fortune 500 firms during the 116th Congress (2019–2020), using OpenSecrets CRP bulk data. The primary analytical instrument is the **RBO directed influence network**.

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

---

## Networks

### RBO Directed Influence Network *(primary)*

**Script:** `src/rbo_directed_influence.py`

The main analytical output. Each node is a Fortune 500 firm. For every pair of firms that share at least one top-30 bill (by spend), **two directed edges** are produced:

- **Edge weight** = proportional RBO: `[(source_firsts + ties/2) / shared_bills] × RBO` — source's fractional first-mover contribution weighted by alignment
- **Edge rbo** = full Rank-Biased Overlap (RBO, p=0.85) between the two firms' bill-priority rankings — same value for both edges of a pair; weights of both edges sum to `rbo`
- **Edge net_temporal** = `source_firsts − target_firsts` (signed; positive when source is the net first-mover)

For decisive pairs (A leads B): edge A→B has net_temporal > 0, edge B→A has net_temporal < 0. For balanced pairs (equal first-mover counts): both edges have net_temporal = 0 and weight = 0.5 × RBO.

Key node attributes:
- `net_strength` **(primary)**: `Σ_j [RBO(i,j) × net_temporal(i,j)]` — RBO-weighted temporal dominance; positive = net agenda-setter, negative = net follower
- `net_influence` (reference): total first-mover wins minus losses across all pairings (unweighted bill count)
- `wc_net_strength`: within-community variant of net_strength (same-community neighbors only)

Node color: green = net_strength > 0 (agenda-setter), red = net_strength < 0 (follower), gray = neutral/isolated.

**Outputs:** `data/rbo_directed_influence.csv`, `data/ranked_bill_lists.csv`, `visualizations/gml/rbo_directed_influence.gml`, `visualizations/png/rbo_directed_influence.png`

**Enrichment:** `src/enrich_directed_gml.py` adds `num_bills`, `bill_aff_community`, `within_comm_net_str`, and `within_comm_net_inf` to the GML node attributes. Reads community partition from `data/archive/communities/communities_affiliation.csv` (requires `src/archive/networks/bill_affiliation_network.py` in archive to have produced that file).

**Gephi export:** `src/gephi_style_export.py` reads the enriched GML and writes a Gephi-ready GEXF to `visualizations/gexf/rbo_directed_influence.gexf`.

**Affiliation-mediated adoption:** `src/affiliation_mediated_adoption.py` tests whether directed adoption pairs (A→B, bill) are mediated by shared lobbyists or lobbying firms. Operates at bill-level (first-quarter shared intermediaries) and network-level (any shared intermediary across full portfolios). Produces `data/affiliation_mediated_adoption.csv` and `data/rbo_edges_enriched.csv`. See §24 in `docs/design_decisions.md` for findings.

---

### Supporting Networks (Archived)

The supporting networks — bill affiliation, RBO similarity, cosine similarity, issue similarity, lobby firm affiliation, and lobbyist affiliation — characterize the co-lobbying structure from different angles and serve as structural inputs. Their scripts are in `src/archive/networks/` and their output data in `data/archive/`. See `docs/design_decisions.md §5–§11` for methodology and `docs/DOCUMENTATION.md §4` for construction details.

**Mega-bill filtering** (`MAX_BILL_DF = 50` in `config.py`): bills lobbied by more than 50 firms are excluded before computing similarity. These omnibus bills create spurious edges regardless of strategic alignment, analogous to stop-word removal in text analysis.

---

## How to Run

All scripts run from the `src/` directory. Parameters (resolution, top-K, RBO p, etc.) are constants at the top of each script.

```bash
cd src

# 1. Extract data (required first)
python opensecrets_extraction.py

# 2. Build the directed influence network (standalone)
python rbo_directed_influence.py

# 3. Mechanism test for directed influence
python affiliation_mediated_adoption.py
python visualize_affiliation_mediation.py

# 4. Enrich GML and export for Gephi
#    (requires bill_affiliation_network.py in archive to have produced communities_affiliation.csv)
python enrich_directed_gml.py
python gephi_style_export.py

# 5. Cross-congressional stability (111th–117th Congress, 2009–2022)
#    Step 1: expand manual_opensecrets_name_mapping.json to cover each congress era (manual).
#            Pre-111th congresses excluded (HLOGA semi-annual codes incompatible with quarterly assign_quarters).
#    Step 2: run extraction + RBO for all seven congresses (writes to data/congress/{num}/)
python multi_congress_pipeline.py
#    Step 3: four stability analyses on the 135-firm stable set
python cross_congressional_stability.py
```

### Validations

```bash
# Core pipeline validations (V01–V11)
python validations/01_extraction_audit.py         # bill-expanded CSV structure audit
python validations/02_inflation_diagnosis.py       # cartesian product inflation check
python validations/03_sparsity_analysis.py         # null model co-lobbying signal
python validations/04_mega_bill_diagnosis.py       # mega-bill prevalence distribution
python validations/05_ind_filter_validation.py     # ind='y' deduplication filter
python validations/06_rbo_cosine_unit_tests.py     # RBO and cosine unit tests
python validations/07_composite_network_validation.py
python validations/08_rbo_p_calibration.py         # RBO p-parameter calibration
python validations/10_rbo_directed_influence_validation.py
python validations/11_mediated_adoption_validation.py

# Congress statistics (V12)
python validations/12_congress_statistics.py

# Directed network analyses (V13–V16, V18–V19; require multi_congress_pipeline.py first)
python validations/13_centrality_vs_agenda_setter.py
python validations/14_influencer_regression.py
python validations/15_cross_sector_directed_edges.py
python validations/16_industry_influencer_hierarchy.py
python validations/18_payoff_complementarity.py
python validations/19_bill_adoption_diffusion.py
```

All validation outputs are written to `outputs/validation/`.

---

## Project Layout

```
src/
  opensecrets_extraction.py         Extraction pipeline (run first)
  config.py                         Paths and shared constants
  rbo_directed_influence.py         PRIMARY: directed influence network
  enrich_directed_gml.py            Add enriched node attributes to directed GML
  gephi_style_export.py             Export filtered GEXF for Gephi
  affiliation_mediated_adoption.py  Bill-level affiliation-mediated adoption analysis
  visualize_affiliation_mediation.py  Visualization suite for mediation analysis
  multi_congress_pipeline.py        Per-congress extraction + RBO (111th–117th)
  cross_congressional_stability.py  Direction/magnitude/rank stability across 7 congresses
  utils/
    data_loading.py                 CSV loaders with column validation
    filtering.py                    Mega-bill prevalence filtering
    similarity.py                   RBO and cosine helpers
    network_building.py             Graph construction and GML export
    centrality.py                   Centrality computations (within-community eigenvector, PageRank, Katz)
    community.py                    Leiden detection and resolution sweep
    visualization.py                Circular layout plots
    bc_diagnostics.py               Betweenness-centrality diagnostics
  validations/
    01_extraction_audit.py          ...
    19_bill_adoption_diffusion.py   (V17 not present; all outputs → outputs/validation/)
  archive/
    networks/                       Archived supporting network scripts
      bill_affiliation_network.py
      rbo_similarity_network.py
      cosine_similarity_network.py
      issue_similarity_network.py
      lobby_firm_affiliation_network.py
      lobbyist_affiliation_network.py
      issue_rbo_similarity_network.py
      build_bill_company_matrix.py
    build_lobbyview_mapping.py      (other archived legacy scripts)
    lobbyview_extraction.py
    rbo_quarterly_networks.py
    fortune_20/                     Fortune 20 subset legacy scripts
    psne/                           PSNE game theory legacy code

data/
  OpenSecrets/                      Raw CRP bulk files (lob_*.txt)
  congress/
    111/–117/                       Per-congress extraction + RBO outputs
  network_edges/
    lobbyist_affiliation_edges.csv  Shared lobbyist edges (used by mediation analysis)
  manual_opensecrets_name_mapping.json  Active Fortune 500 → CRP name mapping
  opensecrets_lda_reports.csv       Primary extraction output (report × bill)
  opensecrets_lda_issues.csv        Issue-code extraction output
  rbo_directed_influence.csv        Directed influence edge list
  ranked_bill_lists.csv             Per-firm top-30 bill rankings
  affiliation_mediated_adoption.csv Bill-level mediated adoption dataset
  rbo_edges_enriched.csv            RBO edges with mediation rates and connectivity
  archive/                          Supporting network outputs (regeneratable)
    network_edges/                  Affiliation, RBO, cosine, composite, issue edge CSVs
    communities/                    Leiden community assignments
    centralities/                   Centrality measure CSVs
    LobbyView/                      Alternative data source (not used in pipeline)
    cleaning/                       Intermediate cleaning files

visualizations/
  gml/
    rbo_directed_influence.gml      PRIMARY: enriched directed influence GML for Gephi
    rbo_directed_influence_111.gml  111th Congress directed network
    rbo_directed_influence_117.gml  117th Congress directed network
  gexf/
    rbo_directed_influence.gexf     Filtered, colored GEXF for Gephi
  png/
    rbo_directed_influence.png      Directed circular plot (top-20 firms)
    rbo_directed_influence_111.png  111th Congress directed plot
    rbo_directed_influence_117.png  117th Congress directed plot
    affiliation_mediation_*.png     Mediation analysis figures (3)
  pdf/
    filtered_rbo_influence.pdf      Filtered directed network (publication figure)
    rbo_influence.pdf               Full directed network
  gephi/
    lobbying_networks.gephi         Gephi project file
  archive/
    undirected/                     Supporting network GMLs, PNGs, and PDFs

outputs/
  validation/                       All validation script outputs (.txt, .csv, .png)
  cross_congressional/              Stability report (.txt, .png, .docx)
  archive/
    channel_tests/                  Archived channel test outputs (to be manually deleted from outputs/channel_tests/)

docs/
  design_decisions.md               Methodology and design decision log (§0–§35+)
  DOCUMENTATION.md                  Full reproduction guide
  directed_influence_summary.md     Summary of directed influence network analysis
  affiliation_mediated_adoption_summary.md  Summary of mediation analysis
  validations_13_19_reference.md    Reference doc for validations 13–19
```

---

## Dependencies

```
pandas numpy networkx python-igraph leidenalg scikit-learn scipy matplotlib statsmodels
```

Install: `pip install pandas numpy networkx python-igraph leidenalg scikit-learn scipy matplotlib statsmodels`
