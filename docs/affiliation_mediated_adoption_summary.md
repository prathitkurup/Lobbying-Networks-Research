## Affiliation-Mediated Adoption — Full Summary

### Concept

The RBO directed influence network identifies *which* firm set whose agenda, but not *how* the signal traveled. The proposed proximate mechanism is shared lobbyists and lobbying firms: an intermediary working for both Firm A and Firm B on the same bill creates a natural information channel through which A's bill priority can reach B. This analysis tests that mechanism directly.

**Operationalization:** For each (A, B, bill) triple where A's first-quarter adoption predates B's, check whether A and B share a lobbyist or external registrant — either on the first-quarter reports for that specific bill (bill-level) or across their full lobbying portfolios (network-level). The bill-level test is the strictest causal story; the network-level test asks whether the institutional connection exists at all.

### What Was Built

Two scripts:
- `src/affiliation_mediated_adoption.py` — core data construction
- `src/validations/11_mediated_adoption_validation.py` — 9-section statistical validation report

### Core Logic (`affiliation_mediated_adoption.py`)

**Inputs:** `data/opensecrets_lda_reports.csv`, `data/rbo_directed_influence.csv`, `data/ranked_bill_lists.csv`, `data/network_edges/lobbyist_affiliation_edges.csv`.

**Pipeline:**

1. Build `{(firm, bill): min_quarter}` — global first-adoption quarter for every (firm, bill) pair across all 8 congress quarters.
2. Build `{(firm, bill): set(registrant)}` — external registrants (non-self-filers) on each firm's first-quarter reports for each bill. First-quarter only: captures the affiliation at the moment of adoption.
3. Build `{(firm, bill): set(lobbyist_name)}` — lobbyists from pipe-separated column on each firm's first-quarter reports, exploded to one row per name.
4. Build `{company: set(bill_number)}` from ranked bill lists — top-30 bill sets per firm.
5. Build undirected lobbyist-network adjacency set from `lobbyist_affiliation_edges.csv`; build firm-network adjacency inline from `opensecrets_lda_reports.csv` (external registrants only, mirrors `lobby_firm_affiliation_network.py`).
6. For each RBO edge (A, B), iterate over shared top-30 bills: assign bill-level leader/follower, compute lag, check bill-level and network-level affiliation.

**Output columns per (edge, bill) record:**

| Column | Description |
|---|---|
| `rbo_source`, `rbo_target` | RBO edge endpoints |
| `rbo_balanced` | 0 = directed RBO edge, 1 = balanced (tied) |
| `bill` | specific bill number |
| `leader`, `follower` | bill-level first/second adopter; None if tied |
| `q_leader`, `q_follower` | first-adoption quarters |
| `lag_quarters` | `q_follower − q_leader` (0 = same quarter) |
| `is_bill_directed` | True if lag > 0 |
| `shared_lobbyist_count` | bill-level shared lobbyists on first-quarter reports |
| `shared_firm_count` | bill-level shared external registrants on first-quarter reports |
| `shared_lobbyists`, `shared_firms` | pipe-separated names |
| `is_lobbyist_mediated`, `is_firm_mediated`, `is_any_mediated` | bill-level flags |
| `net_lob_connected`, `net_firm_connected`, `net_any_connected` | network-level connectivity flags |

### Key Design Decisions

- **First-quarter reports only** for the bill-level test: the adoption moment is where the transmission signal should be observable. Using all-quarter reports produces the same result (shared names remain rare) but loses temporal precision.
- **External registrants only** for the firm channel: self-filers are their own registrant — overlap on a self-filer is not evidence of a bridging intermediary. Consistent with `lobby_firm_affiliation_network.py`.
- **Both directed and balanced RBO edges included** in the dataset: balanced pairs serve as a comparison group (section 2 of the validation report).
- **Two-level design**: bill-level co-affiliation (mechanistic, strict) and network-level connectivity (institutional, broad). The two levels answer different questions and should not be collapsed.

### Empirical Results

**Scale:** 8,173 (edge, bill) records total; 3,184 directed (lag > 0); 4,989 tied (same quarter).

**Bill-level mediation:**

| Channel | Count (directed pairs) | Rate |
|---|---|---|
| Lobbyist-mediated | 6 | 0.2% |
| Firm-mediated | 3 | 0.1% |
| Any-mediated | 7 | 0.2% |

**Network-level connectivity:**

| Channel | Count (directed pairs) | Rate |
|---|---|---|
| Lobbyist-network connected | 19 | 0.6% |
| Firm-network connected | 13 | 0.4% |
| Any connected | 20 | 0.6% |

**Substantive finding:** Direct affiliation-mediated adoption is extremely rare. The two non-trivial mediated dyads are:

- **United Technologies → Raytheon** (6 mediated bills): contextually explained by their April 2020 merger into Raytheon Technologies — by 2020 they shared a large shared lobbying team (Coffin, Holladay, Jimenez, McBride, Peterson, Thompson, Thornblad via AJW Group).
- **LOEWS → ECOLAB** (1 bill, S.4178): shared via Ogilvy Government Relations.

**Lag compression (section 3):** No statistically significant difference in adoption lag between mediated and non-mediated pairs (Mann-Whitney U, p = 0.57). Underpowered given n = 7.

**Alignment test (section 8):** All 7 bill-level-mediated bills have the bill-level leader as the RBO edge source — 100% alignment, significant at p = 0.008 (binomial test). For non-mediated directed bills the alignment rate is 94.9% (p ≈ 0). Affiliation-mediated bills are fully consistent with the aggregate RBO direction.

**Edge-level distribution (section 6):** 99.9% of directed RBO edges have any_mediation_rate = 0. Only two edges (LOEWS → ECOLAB; UNITED TECHNOLOGIES → RAYTHEON) have rate > 0.

### Interpretation

The RBO network's directed influence signal is not primarily explained by direct shared-affiliation channels. The agenda-setting coordination captured by the network operates through a broader mechanism: likely a combination of industry-structural co-response (firms in the same sector track the same legislative landscape independently), issue-portfolio similarity driving correlated bill adoption, and possibly multi-step information diffusion not detectable at the direct tie level. The handful of directly affiliation-mediated cases are structurally anomalous (merger context). This is consistent with Carpenter et al. (1998)'s finding that indirect ties matter as much as direct ones in lobbying coalition formation.

### Validation (`11_mediated_adoption_validation.py`)

9 sections:

| Section | Test |
|---|---|
| 1 | Overall mediation rates: directed vs tied pairs |
| 2 | Mediation rate by RBO edge type (directed vs balanced) |
| 3 | Lag distribution: mediated vs non-mediated (Mann-Whitney U) |
| 4 | Top broker lobbyists and firms by mediated-adoption count |
| 5 | Per-bill mediation frequency |
| 6 | Edge-level mediation rate distribution and top mediated edges |
| 7 | Network-level vs bill-level connectivity comparison table |
| 8 | Alignment test: do mediated bills confirm RBO edge direction? (binomial test) |
| 9 | Co-mediation: lobbyist and firm channel overlap |

### Output Files

| File | Description |
|---|---|
| `data/affiliation_mediated_adoption.csv` | 8,173 rows: one per (RBO edge, shared bill) with bill-level and network-level flags |
| `data/rbo_edges_enriched.csv` | 3,712 rows: original RBO edge list joined with directed-bill mediation rates and network connectivity |

### Documentation

- `docs/design_decisions.md §24` — full methodology, two-level design rationale, empirical findings, alternatives considered
- `docs/DOCUMENTATION.md §11` — reproduction steps and output file reference
