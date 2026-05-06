# Corporate Lobbying Networks

Analysis of corporate lobbying behavior among Fortune 500 firms spanning the 111th through 117th Congresses (2009–2022), with the 116th Congress (2019–2020) as the primary analytical window. Uses OpenSecrets CRP bulk data. The primary analytical instrument is the **RBO directed influence network**.

---

## Core Idea

Two firms are "similar" if they lobby the same bills and rank those bills similarly by spend. A firm "influences" another if it consistently lobbied their shared bills *first* — setting the other firm's legislative agenda before the follower adopted those bills.

This agenda-setting signal is operationalized via temporal bill-adoption precedence. For each firm, bills are ranked in descending order of lobbying spend. For every pair of firms sharing at least one top-30 bill, we compute the Rank-Biased Overlap (RBO) of their ranked bill portfolios — capturing both which bills two firms co-lobby and how similarly they prioritize them, weighting agreement at the top of the ranking more heavily. If Firm A lobbied a bill in Q1 and Firm B adopted it in Q3, A receives a first-mover credit. Aggregating across all shared bills, RBO-weighted by portfolio alignment, yields the directed influence network.

A firm's **net\_strength** — the RBO-weighted sum of first-mover advantages over all pairings — is the primary measure of influence. Positive net\_strength identifies a net agenda-setter; negative identifies a net follower.

The proximate mechanism through which agenda-setting propagates is shared human lobbyists: lobbyists who work for both firms create information bridges that transmit bill priorities from influencer to follower.

---

## Data

All data comes from **OpenSecrets CRP bulk lobbying tables**.

| File | Contents |
|---|---|
| `lob_lobbying.txt` | Report-level filings: registrant, client, amount |
| `lob_lobbyist.txt` | Named lobbyists per report |
| `lob_issue.txt` | Issue codes per report |
| `lob_bills.txt` | Bills per issue entry |

**Name mapping:** `data/manual_opensecrets_name_mapping.json` — manually curated JSON mapping Fortune 500 firm names to their OpenSecrets CRP name variants.

---

## Extraction Pipeline

`src/opensecrets_extraction.py` is the single entry point for all data. It filters to `ind='y'` records (OpenSecrets' validity flag excluding superseded filings and double-counted subsidiaries), retains only reports with at least one named lobbyist, and disaggregates reported dollar amounts evenly across all bills named in each report.

**Produces:**
- `data/opensecrets_lda_reports.csv` — one row per (report, bill); `amount_allocated = amount / n_bills`
- `data/opensecrets_lda_issues.csv` — one row per (report, issue_code)

---

## Networks

### RBO Directed Influence Network *(primary)*

**Script:** `src/rbo_directed_influence.py`

For every pair of Fortune 500 firms sharing at least one top-30 bill, two directed edges are produced:

- **Edge weight** = `[(source_firsts + ties/2) / shared_bills] × RBO` — source's fractional first-mover contribution weighted by alignment
- **Edge rbo** = full RBO (p=0.85) between the two firms' bill-priority rankings; same value for both edges of a pair; both edge weights sum to `rbo`
- **Edge net\_temporal** = `source_firsts − target_firsts` (signed; positive when source is the net first-mover)

Key node attributes:
- `net_strength` **(primary)**: `Σ_j [RBO(i,j) × net_temporal(i,j)]` — RBO-weighted temporal dominance
- `net_influence` (reference): total first-mover wins minus losses (unweighted)
- `wc_net_strength`: within-community variant, restricted to same-community neighbors

**Mega-bill filtering** (`MAX_BILL_DF = 50` in `config.py`): bills lobbied by more than 50 firms are excluded before computing similarity. These high-participation omnibus bills inflate similarity regardless of strategic alignment.

**Outputs:** `data/rbo_directed_influence.csv`, `data/ranked_bill_lists.csv`, `visualizations/gml/rbo_directed_influence.gml`, `visualizations/png/rbo_directed_influence.png`

**Enrichment:** `src/enrich_directed_gml.py` adds `num_bills`, `bill_aff_community`, `within_comm_net_str`, and `within_comm_net_inf` to the GML node attributes. Community labels (six Leiden communities: Finance/Insurance, Tech/Telecom, Energy/Utilities, Defense/Industrial, Health/Pharma, Consumer/Manufacturing) come from `src/bill_affiliation_network.py`.

**Gephi export:** `src/gephi_style_export.py` reads the enriched GML and writes a filtered, colored GEXF to `visualizations/gexf/rbo_directed_influence.gexf`.

**Affiliation-mediated adoption:** `src/affiliation_mediated_adoption.py` tests whether directed adoption pairs (A→B, bill) are mediated by shared lobbyists or lobbying firms — at bill-level (first-quarter shared intermediaries) and network-level (any shared intermediary across full portfolios). Produces `data/affiliation_mediated_adoption.csv` and `data/rbo_edges_enriched.csv`.

---

### Bill Affiliation Network

**Script:** `src/bill_affiliation_network.py`

Undirected weighted network where edge weight = number of distinct co-lobbied bills. Used for Leiden community detection (six communities, γ=1.0) and as the structural basis for centrality comparisons. Outputs to `data/archive/` and `visualizations/gml/bill_affiliation_network.gml`.

---

## How to Run

The full pipeline is managed by `run_pipeline.sh` at the repository root:

```bash
bash run_pipeline.sh
```

This runs five phases in sequence: (1) core extraction and 116th Congress networks, (2) cross-congressional stability, (3) validation scripts, (4) publication figures, and (5) the eight focused analyses. Each script is logged individually to `outputs/run_logs/`. The pipeline continues on failure and prints a pass/fail summary at the end.

**Notes:**
- `multi_congress_pipeline.py` must be run manually before the first full pipeline run — it takes 10–20 minutes and is commented out of the default script: `cd src && python multi_congress_pipeline.py`
- All parameters (resolution, top-K, RBO p, etc.) are constants at the top of each script; no CLI arguments

---

## Analyses

```bash
cd src/analysis
python 01_primary_directed_influence.py   # Top global and within-community agenda-setters, spend comparison
python 02_mediation.py                    # Shared-lobbyist mediation channels
python 03_industry_hierarchy.py           # Kendall's W rank stability by sector, 111th–117th Congress
python 04_cross_congressional.py          # Spearman ρ heatmap, RBO list similarity across sessions
python 05_multi_congress.py               # Jaccard top-set overlap, entry/exit transitions
python 06_centrality_vs_agenda_setters.py # Centrality vs. net_strength Spearman ρ
python 07_strategic_complementarity.py    # BCZ payoff complementarity + direction persistence
python 08_bill_adoption_cascading.py      # Bill adoption diffusion, LPM regression
```

All analysis outputs are written to `outputs/analysis/` (CSVs, TXT summaries, PNGs).

Together, these eight analyses address the paper's core empirical questions: who sets the legislative agenda among Fortune 500 lobbying firms, whether that leadership is stable across industries and congressional sessions, whether shared lobbyists are the proximate transmission mechanism, and whether the agenda-setting structure is consistent with strategic complementarity — a model in which followers amplify the bills their influencers prioritize, reinforcing the influence hierarchy over time.

---

## Project Layout

```
src/
  config.py                             Paths and shared constants
  opensecrets_extraction.py             Extraction pipeline (run first)
  build_bill_company_matrix.py          Bill-company incidence matrices for Fortune tiers
  bill_affiliation_network.py           Shared-bill affiliation network + Leiden community detection
  rbo_directed_influence.py             PRIMARY: directed influence network (116th Congress)
  enrich_directed_gml.py                Enrich GML with community and within-community node attributes
  gephi_style_export.py                 Export filtered GEXF for Gephi
  affiliation_mediated_adoption.py      Bill-level affiliation-mediated adoption analysis
  visualize_affiliation_mediation.py    Visualization suite for mediation analysis
  multi_congress_pipeline.py            Per-congress extraction + RBO (111th–117th)
  cross_congressional_stability.py      Direction/magnitude/rank stability across 7 congresses
  utils/
    data_loading.py                     Congress/year range helpers
    filtering.py                        Mega-bill prevalence filtering
    similarity.py                       RBO and cosine helpers
    network_building.py                 Graph construction and GML export
    centrality.py                       Centrality computations (global and within-community)
    community.py                        Leiden detection and resolution sweep
    visualization.py                    Circular layout plots
  analysis/
    01_primary_directed_influence.py
    02_mediation.py
    03_industry_hierarchy.py
    04_cross_congressional.py
    05_multi_congress.py
    06_centrality_vs_agenda_setters.py
    07_strategic_complementarity.py
    08_bill_adoption_cascading.py
  archive/
    validations/                        Archived V01–V20 pipeline validation scripts
    networks/                           Archived supporting network scripts
    extraction/                         Archived legacy extraction scripts
    psne/                               Legacy PSNE game theory code
    fortune_20/                         Legacy Fortune 20 subset scripts

data/
  OpenSecrets/                          Raw CRP bulk files (lob_*.txt)
  congress/
    111/–117/                           Per-congress: opensecrets_lda_reports.csv, rbo_directed_influence.csv,
                                        node_attributes.csv, ranked_bill_lists.csv
  affiliation/
    f30/ f50/ f100/ f500/               Bill-company incidence matrices (with/without singleton bills)
  manual_opensecrets_name_mapping.json  Active Fortune 500 → CRP name mapping (manually curated)
  opensecrets_lda_reports.csv           116th Congress extraction output (report × bill)
  opensecrets_lda_issues.csv            116th Congress issue-code extraction output
  lobbyist_client_116_opensecrets.csv   Lobbyist-to-client mapping (116th Congress)
  fortune_primary_industry_116.csv      Primary industry labels for Fortune 500 firms
  rbo_directed_influence.csv            Directed influence edge list (116th Congress)
  ranked_bill_lists.csv                 Per-firm top-30 bill rankings (116th Congress)
  affiliation_mediated_adoption.csv     Bill-level mediated adoption dataset
  rbo_edges_enriched.csv                RBO edges with mediation rates and connectivity flags
  archive/                              Supporting network outputs (regeneratable)
    network_edges/                      Affiliation, RBO, cosine, composite edge CSVs
    communities/                        Leiden community assignments
    centralities/                       Centrality measure CSVs
    LobbyView/                          Alternative data source (not used in pipeline)
    cleaning/                           Intermediate cleaning files

visualizations/
  gml/
    rbo_directed_influence.gml          PRIMARY: enriched directed influence GML
    bill_affiliation_network.gml        Undirected bill affiliation network with Leiden communities
    rbo_directed_influence_111.gml      Per-congress directed networks (111th–117th)
    …
    rbo_directed_influence_117.gml
  gexf/
    rbo_directed_influence.gexf         Filtered, colored GEXF for Gephi (filtered network only)
  png/
    rbo_directed_influence.png          Directed plot, top-20 firms (116th Congress)
    rbo_directed_influence_111.png      Per-congress directed plots (111th–117th)
    …
    rbo_directed_influence_117.png
    bill_affiliation_network.png        Bill affiliation plot, top-20 firms
    affiliation_mediation_*.png         Mediation analysis figures (3)
  pdf/
    bill_affiliation.pdf
    complete_influence_network.pdf
    filtered_influence_network_net_strength.pdf
    filtered_influence_network_wc_strength.pdf
  publication/
    bill_affiliation.pdf/.png/.svg
    complete_influence_network.pdf/.png/.svg
    filtered_influence_network_net_strength.pdf/.png/.svg
    filtered_influence_network_wc_strength.png/.svg
  gephi/
    final_network_visuals.gephi         Gephi project: bill affiliation, directed, and filtered directed networks
  archive/
    undirected/                         Archived undirected similarity network GMLs, PNGs, PDFs
    fortune_20/                         Legacy Fortune 20 subset visualizations

outputs/
  analysis/                             All analysis script outputs (.csv, .txt, .png)
  validation/                           Validation script outputs (current run)
  cross_congressional/                  Cross-congressional stability outputs
  run_logs/                             Timestamped pipeline run logs
  archive/                              Prior validation and cross-congressional runs

docs/
  DOCUMENTATION.md                      Full reproduction guide
  design_decisions.md                   Methodology and design decision log
  directed_influence_summary.md         Summary of directed influence network analysis
  affiliation_mediated_adoption_summary.md  Summary of mediation analysis
```

---

## Dependencies

```
pandas numpy networkx python-igraph leidenalg scikit-learn scipy matplotlib statsmodels
```

Install: `pip install pandas numpy networkx python-igraph leidenalg scikit-learn scipy matplotlib statsmodels`
