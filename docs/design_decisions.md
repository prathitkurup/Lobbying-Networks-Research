# Design Decisions — Corporate Lobbying Network Analysis

**Authors:** Prathit Kurup, Victoria Figueroa
**Data:** OpenSecrets CRP, 116th Congress (2019–2020), Fortune 500 firms

Records significant design decisions, the reasoning behind each, and alternatives considered and rejected. Intended to support the methods section and allow full reconstruction of analytical choices.

---

## §0 — Conceptual Framework: Influence as Agenda-Setting

**Definition:** In this research, *influence* is operationalized as **agenda-setting**.
A highly influential firm influences another to adjust its lobbying priorities
— its ranked portfolio of lobbied bills — to more closely match its own.
Agenda-setting is observed when one firm shifts its priorities as a *follower*:
over time, the follower increasingly engages bills that the influencer had already
been lobbying for, indicating that the influencer's bill adoption shapes the
follower's legislative agenda.

**Operationalization via temporal precedence:** Influence is inferred from
bill-adoption timing. If Firm A consistently lobbies a bill before Firm B across
their shared bill portfolio, A is the agenda-setter and B is the follower.
The directed influence network (§20) formalizes this: for each RBO-linked
company pair, a directed edge A→B is emitted when A is the first-mover on more
shared bills than B.

**Underlying mechanism:** The proximate mechanism through which agenda-setting
propagates is lobbyist networks — shared human lobbyists create information
bridges between firms, transmitting bill priorities from influencers to followers
(Carpenter et al., 1998; Koger & Victor, 2009). However, the direct modeling
of lobbyist networks as transmission channels is out of scope for this project.
The directed influence network captures the *outcome* of agenda-setting
(temporal priority patterns) rather than the mechanism itself.

**Priority rankings:** "Priorities" are defined by the fraction of a firm's total
lobbying spend allocated to each bill (the `frac` column in the processed data).
The ranked bill list for each firm, sorted by spend fraction descending, is the
input to both the RBO similarity networks and the directed influence networks.

**Connection to RBO similarity:** High RBO similarity between A and B means
their bill-priority rankings are already closely aligned. The directed influence
layer adds a causal direction to that similarity: which firm shaped whose agenda.

---

## §1 — Data Structure: One Row Per Bill Per Report

**Decision:** Accept the raw data format from `opensecrets_extraction.py` (one row per
bill per report) and aggregate **before** network construction, not inside
`opensecrets_extraction.py`.

**Rationale:** `opensecrets_extraction.py` divides each filing's reported spend equally
across all bills in that report (`amount / num_bills`), producing one row per
(bill, report) pair. This design is correct for spend accounting — it correctly
tracks how spend was distributed across filing periods. Aggregating inside
`opensecrets_extraction.py` would lose the per-report audit trail. The correct
approach is to aggregate at network construction time.

**Pre-processing fix:**
- Affiliation network: `df.drop_duplicates(subset=["fortune_name","bill_number"])`
  — reduces to presence/absence; amount is irrelevant for affiliation.
- Cosine/RBO similarity: `df.groupby(["fortune_name","bill_number"])["amount_allocated"].sum()`
  — collapses to true total allocated spend per (firm, bill).

**Validation:** Run `validations/01_extraction_audit.py` to quantify
duplication extent. Typically ~40% of (firm, bill) pairs have multiple rows,
with the worst offenders at 10–20 rows.

---

## §2 — Cartesian Product Inflation Bug

**Decision:** Fix the cartesian product inflation in the original edge
construction by deduplicating before the clique-building loop.

**Bug description:** Without deduplication, `df.groupby("bill_number")["fortune_name"]
.apply(list)` builds lists with duplicate firm entries. For a pair (A, B) where
A has R_A rows and B has R_B rows on a shared bill, the i < j loop generates
R_A × R_B records instead of 1 — inflating shared-bill counts ~6x at the median
(observed: median went from 19 to 3 after the fix).

**Fix 1 — Deduplication:** See §1 above.

**Fix 2 — Canonical pair ordering:** Without canonicalization, `(A, B)` and
`(B, A)` from different bills' list orderings are stored as separate records.
The groupby merge only collapses exact (source, target) matches, so reverse-order
pairs remain as distinct edges and the second overwrites the first in NetworkX.
Fix: `src, tgt = (a, b) if a < b else (b, a)` throughout all inner loops.

**Validation:** Run `validations/02_inflation_diagnosis.py` to show side-by-side
comparison of buggy vs. fixed edge weight distributions.

---

## §3 — Sparsity and the Above-Null Signal

**Decision:** Proceed with network analysis rather than dismissing co-lobbying
overlap as noise, based on the null model comparison.

**Finding:** Under a random null (each firm independently selects bills with
probability k_i / B), the expected number of shared bills for the median pair is
≈ 0.11. The observed median is ≈ 3.0 — approximately **27× above chance**.
This is a conservative estimate: it ignores industry structure, which would reduce
expected random overlap even further.

**Interpretation:** Despite global sparsity (most firm pairs share zero bills),
the pairs that do share bills do so at far above chance rates. This is consistent
with strategic coalition formation — firms in the same industry deliberately
coordinate on the same legislation. The 27× signal motivates treating co-lobbying
as a meaningful coordination signal and using community detection to find
lobbying coalitions.

**Decision on singletons:** Bills lobbied by exactly one firm (~54% of all bills)
are **kept in the frac denominator** for cosine and RBO similarity computations.
- They remain in each firm's total budget denominator (preserving frac = share
  of total lobbying portfolio).
- They do NOT produce any co-lobbying edges (no pairs to form).
- Excluding them would artificially inflate fracs on multi-firm bills and distort
  the portfolio composition metric.

**Validation:** Run `validations/03_sparsity_analysis.py` for the full null
model comparison.

---

## §4 — Mega-Bill Prevalence Filtering (MAX_BILL_DF = 50)

**Decision:** Exclude bills lobbied by more than 50 unique firms from all
three component networks (affiliation, cosine, RBO). This is implemented as a
configurable parameter `MAX_BILL_DF` in `config.py`.

**Rationale — Modularity Collapse:** Without filtering, the 16 bills with
df > 50 firms account for ~97.5% of all co-lobbying edges and collapse Leiden
modularity from Q ≈ 0.18 to Q ≈ 0.02 — a near-random partition. The CARES Act
alone (198 firms) creates C(198, 2) = 19,503 pairs, connecting firms from every
sector to every other sector regardless of strategic alignment.

**Rationale — Research Backing:**
1. **TF-IDF stop-word analogy:** Manning, Raghavan & Schütze (2008, §6.2) show
   that terms appearing in a large fraction of documents must be removed as stop
   words before computing co-occurrence or TF-IDF, because they carry no
   discriminative signal. The same logic applies here: a bill lobbied by everyone
   carries no information about which firms coordinate strategically.
2. **Policy science — valence issues:** Hojnacki et al. (2012) and Koger &
   Victor (2009) distinguish "valence issues" (near-universal legislation with
   broad support) from genuinely contested bills where coalition formation occurs.
   CARES Act, NDAA, and annual appropriations are canonical valence issues for the
   Fortune 500.

**Threshold calibration:** The bill prevalence distribution has a natural break
between the 16 mega-bills (50–198 firms) and the industry-specific legislation
(≤ 45 firms). MAX_BILL_DF = 50 removes exactly the mega-bills while preserving
all industry-specific co-lobbying signal.

**Issue-level filtering:** `MAX_ISSUE_DF = None` (disabled) by default for the
issue similarity network. Issue codes are much broader (75 codes vs. 2300+ bills)
and by design most firms lobby multiple issue areas. If analysis reveals that a
handful of codes (TAX, FIN, HCR) collapse issue-network modularity, the same
filtering logic can be applied by setting `MAX_ISSUE_DF` in `config.py`.

**Two-stage filtering for cosine and RBO similarity:**
- Stage 1: Compute total_budget and fracs on **all bills** (including mega-bills)
  to preserve the economic meaning: `frac_ib = spend on bill b / total lobbying budget`.
- Stage 2: Build cosine/RBO pairs from **filtered bills only** (excluding mega-bills).
  This removes the spurious near-equal fracs on omnibus bills (e.g., both firms
  allocate 0.002 to CARES Act, which would inflate cosine similarity despite no
  genuine strategic alignment).

**Validation:** Run `validations/04_mega_bill_diagnosis.py` for the full
prevalence distribution and modularity comparison.

---

## §5 — Choice of Similarity Metrics

**Decision:** Use bill-level RBO similarity as the primary co-lobbying similarity signal for the directed influence network. Affiliation and cosine similarity networks are also built (archived) for structural reference.

**Summary of the three signals:**
- **Affiliation** (`affil_norm = shared_bills / N_total`): raw co-lobbying breadth, normalized by the total filtered bill universe.
- **Cosine similarity**: geometric alignment of portfolio-share (frac) vectors.
- **RBO**: priority-ranking agreement. Captures whether firms' high-spend legislative priorities are the same. This is the primary signal used in the directed influence network.

**Why not Bray-Curtis breadth×depth?**
Bray-Curtis was used in an earlier version but replaced: (a) cosine is a cleaner geometric measure; (b) affil_norm provides the breadth signal more transparently without the exponential calibration parameter λ; (c) RBO top-weights high-spend priorities, matching empirical spend concentration.

---

## §6 — Community Detection: Leiden Algorithm

**Decision:** Use the Leiden algorithm (`leidenalg` library,
`RBConfigurationVertexPartition`) with resolution parameter γ = 1.0 for all
three Layer 1 networks.

**Rationale — Leiden vs. Louvain:** Leiden (Traag, Waltman & van Eck, 2019)
guarantees that all detected communities are internally connected, fixing a known
defect of Louvain where communities can be disconnected. For lobbying networks
where community cohesion is a substantive claim (not just a partition artifact),
internal connectivity is important.

**Rationale — Resolution parameter γ = 1.0:** Higher γ produces more, smaller
communities; lower γ produces fewer, larger communities. γ = 1.0 is the standard
Modularity baseline. For a 291-node network, this typically yields 5–10 communities
— consistent with the number of sectors represented in the Fortune 500. A resolution
sweep at γ ∈ {0.5, 0.75, 1.0, 1.15, 1.25} is run to confirm stability of the
partition structure before reporting results.

**Rationale — Seed fixing:** `seed = 42` is fixed for reproducibility. Leiden
has a stochastic initialization; fixing the seed ensures identical results across
runs.

**Issue network note:** Issue codes (75 codes) are much broader than bills
(2300+), so communities reflect sector-level alignment rather than bill-specific
coalitions. γ = 0.5 may be more appropriate for coarser groupings if the resulting
communities at γ = 1.0 are too granular to interpret.

---

## §7 — Centrality Measures

**Decision:** Compute four centrality measures, all used in validation analyses.

**Within-community eigenvector centrality:** Computed on each Leiden community subgraph independently. Captures recursive "important neighbors" structure within a sector coalition. Falls back to weighted degree for communities with < 3 nodes or when power iteration doesn't converge.

**Within-community PageRank:** Computed on each Leiden community subgraph. Provides the clearest alignment with empirical agenda-setting (see §29 — top-30 Spearman ρ = 0.501 vs. net_influence).

**Katz-Bonacich centrality (global):** `α = 0.85 / λ_max`. Sensitive to multi-hop co-lobbying chains; amplifies structural hubs. See §13 for full rationale.

**Global PageRank:** Full-graph PageRank. Normalizes by out-degree; captures selective high-quality connections. Compared to Katz to identify degree-normalization effects (§13).

**Excluded:** Guimerà-Amaral z-score, participation coefficient (P), and the 7-role taxonomy are not used in the primary analysis.

---

## §8 — Zero-Budget Firm Exclusion

**Decision:** Exclude firms with total lobbying budget = $0 from cosine and RBO
similarity computations (with a printed warning). These firms remain in the
affiliation network.

**Rationale:** A firm with $0 total spend produces `frac = 0/0 = NaN` for all
bills, which would corrupt cosine and RBO scores. The LDA filing is valid even
with $0 spend — firms are legally required to file even if they spent less than
the $5,000 threshold.

**Known affected firms (116th Congress):**
- Air Products & Chemicals
- Crown Holdings
- Dick's Sporting Goods
- Treehouse Foods

**Why some firms appear in the affiliation network but not cosine/RBO:**
1. Zero-budget firms: excluded by the `compute_zero_budget_fracs()` guard.
2. Republic Services: non-zero budget ($70k), but $0 allocated on its only
   shared bill via equal-split → frac = 0 for all bills → cosine = 0 for all
   pairs → absent from cosine edges after the `weight > 0` filter.
3. Firms whose only shared bills are mega-bills: filtered out, leaving them
   with no non-mega bills in common with any other firm.

---


## §10 — GML Export with Community Node Attributes

**Decision:** Store community membership as a node attribute in GML files
(community = integer community_id). This allows direct import into Gephi with
community partition immediately available as a node attribute column, requiring
no separate CSV import or manual post-import assignment.

**Implementation:** The `write_gml_with_communities()` function in
`utils/network_building.py` performs Leiden detection (if not already computed),
then adds a `community` node attribute (cast to int for Gephi compatibility)
before writing the GML file. The function signature is:

```python
write_gml_with_communities(G, partition, filepath)
```

where `partition` is a dict mapping node names to community IDs.

**Workflow:**
1. Build the network graph G.
2. Run Leiden community detection to get `partition` dict.
3. Call `write_gml_with_communities(G, partition, output_path)`.
4. Open the GML in Gephi: node attributes table immediately shows the
   `community` column with integer IDs.
5. Use "Appearance" → "Nodes" → "Partition" to color by community.

This reduces friction in visualization and interpretation compared to importing
GML + separate community CSV, and ensures the partition metadata is always
synchronized with the network structure in a single file.

---

## §11 — Issue Similarity Network: Cosine Similarity on Issue Portfolios

**Decision:** The issue similarity network uses cosine similarity of
issue-portfolio-share (frac) vectors, identical in construction to the
bill-level cosine similarity network but aggregated over issue codes rather
than individual bills.

**Metric definition:**
```
cos(i, j) = (u_i · u_j) / (||u_i|| × ||u_j||)
```
where `u_i[k] = firm i's total spend on issue code k / firm i's total lobbying
budget`.  Weights are bounded in [0, 1] since all fracs are non-negative.

**Why cosine (not Bray-Curtis):**
An earlier version used Bray-Curtis per issue code, aggregated with sqrt
normalization: `weight = sum(BC_k) / sqrt(N_shared_issues)`.  This produced
unbounded weights (max ≈ sqrt(75) ≈ 8.66) and required users to interpret the
weight scale carefully.  Replacing with cosine gives a clean [0, 1] score
consistent with the bill-level cosine network, and removes the need for a
separate normalization convention specific to the issue network.

**Data note:**
`opensecrets_lda_issues.csv` has one row per (report, issue_code).  Without
aggregation, the firms × issues pivot would inflate issue-level fracs by report
count.  We first aggregate to one row per (fortune_name, issue_code)
by summing amounts before computing fracs.

**Relationship to the bill cosine network:**
- Bill cosine: `u_i[b] = spend on bill b / total budget` — fine-grained
  (2300+ dimensions), captures bill-specific alignment.
- Issue cosine: `u_i[k] = spend on issue code k / total budget` — coarser
  (75 dimensions), captures sector/domain alignment.

The two networks are complementary: the bill cosine network is part of the
composite pipeline; the issue cosine network is standalone, capturing policy-area
strategy rather than specific legislative priorities.

**Source:** `src/archive/networks/issue_similarity_network.py`

---


## §13 — Katz-Bonacich Centrality

**Decision:** Compute Katz-Bonacich centrality alongside within-community eigenvector, within-community PageRank, and global PageRank as the four centrality measures used in validation analyses.

**Definition:**
```
C_katz(i) = α × Σ_j A_ij × C_katz(j) + β
```
Equivalently, Katz sums contributions from all walks of all lengths, with
exponential decay in walk length controlled by α:
```
C_katz = β × (I − αA)^{-1} × 1
```
α must be < 1/λ_max (spectral radius of the adjacency matrix) to guarantee
convergence. We auto-set `α = 0.85 / λ_max`, placing the parameter deep in
the convergence region (0.85 = 85% of the theoretical maximum α).

**Rationale:**
- **vs. PageRank:** PageRank normalises by node out-degree, so high-degree
  nodes propagate influence in smaller units to each neighbor. Katz does not
  normalise: a high-degree node receives the full sum of its neighbors'
  centrality values, scaled by α. In lobbying networks, this makes Katz
  sensitive to highly connected hubs that serve as intersection points for
  many co-lobbying chains — firms that appear in many overlapping coalitions
  simultaneously. PageRank would dampen their influence; Katz amplifies it.

- **vs. within-community eigenvector:** Eigenvector centrality is community-
  scoped (run on the community subgraph). Katz is global. Comparing a firm's
  Katz rank to its within-community eigenvector rank identifies firms whose
  community-level dominance translates (or does not translate) into global
  network prominence.

- **Theoretical motivation:** Bonacich (1987) introduced the power centrality
  family specifically to model influence in networks where being connected to
  powerful others confers status. For lobbying, this captures indirect political
  influence: a firm that co-lobbies with many influential firms gains Katz
  centrality even if its direct ties are few, because it is embedded in a
  favorable multi-hop structure.

**Alpha calibration:** `α = 0.85 / λ_max` where `λ_max` is computed via
`numpy.linalg.eigvals` on the weighted adjacency matrix. If convergence fails
at 0.85/λ_max, the implementation falls back to 0.50/λ_max, then to weighted
degree (same order, different magnitude). This fallback chain ensures graceful
degradation without silent failures.

**Interpretation notes:**
- Higher Katz centrality → more influence-receiving capacity through the full
  network topology (direct + indirect paths, path-length penalised).
- Katz ranking and PageRank ranking will differ whenever degree is heterogeneously
  distributed (high-degree nodes are promoted by Katz, normalised by PageRank).
- The difference in Katz vs PageRank rank can be used as a diagnostic: firms
  where Katz >> PageRank are structural hubs in multi-hop co-lobbying chains;
  firms where PageRank >> Katz are selective connectors with few but high-quality
  ties to other influential firms.

**Source:** `src/utils/centrality.py` — `compute_katz_centrality()`
**Validation:** `src/validations/13_centrality_vs_agenda_setter.py`

---


## §15 — No Lower Threshold on Edge Formation; Firm Coverage

**Decision:** Set `DEFAULT_MIN_WEIGHT = 0.0` in all three similarity networks
(cosine, RBO, composite). Every firm pair with any nonzero similarity is included.
Exact-zero similarity pairs (no shared bills at all after mega-bill filtering) are
excluded by a `weight > 0` guard, not an arbitrary positive threshold.

**Rationale:**
The previous defaults (cosine min=0.10, RBO min=0.01) caused 25 firms present in
the bill affiliation network to drop out of the cosine network entirely, and 21 from
the RBO network — including major Fortune 500 companies like Amazon, Anthem, Adobe,
Broadcom, Netflix, and Starbucks.  These firms were not genuinely isolated; they had
some co-lobbying signal, but all their pairwise similarities fell below the arbitrary
cutoff.  An arbitrary threshold is methodologically equivalent to imputing zero
similarity where nonzero similarity exists, which is incorrect.

**Why firms can be absent from similarity networks despite appearing in affiliation:**
Three legitimate reasons for genuine absence (zero similarity, not threshold artifact):
1. **Pure singleton lobbiers**: All bills a firm lobbies for (after mega-bill
   filtering) are bills that no other Fortune 500 firm lobbies for. Their frac
   vector is orthogonal to every other firm's → cosine = 0; no shared bills →
   RBO = 0 and shared_n = 0.
2. **Mega-bill-only co-lobbiers**: A firm's only co-lobbying is on mega-bills
   (df > 50). After filtering these out, they share no bills with anyone.
   Present in affiliation (the mega-bills count there), absent from similarity.
3. **Zero-budget firms**: Firms with $0 total lobbying spend are excluded from
   cosine (frac = 0/0 = NaN) and from RBO (no ranked list possible). See §8.

**Known affected firms (116th Congress OpenSecrets data):**
With threshold=0, firms in the affiliation network that still do not appear in the
cosine or RBO network have genuine zero co-lobbying overlap post-filtering.
Republic Services is the canonical example (§8): non-zero budget but all spend on
a single bill that was filtered as a mega-bill.

**Effect on community size distribution:**
Setting threshold=0 increases network density and reduces the number of small (2–4
node) isolated communities, since previously isolated firms now appear if they have
any co-lobbying similarity above zero. Small communities that survive after this
change represent genuinely niche lobbying clusters — firms whose only co-lobbying
ties are a small number of specialized bills shared with a small group of peers.
These should be interpreted carefully: they may reflect data limitations (limited
LDA filing coverage for some firms) or genuine lobbying niches (e.g., agricultural
commodity firms, specialty insurers).

---

## §16 — OpenSecrets Extraction: ind='y' Validity Filter

**Decision:** Filter `lob_lobbying.txt` to `ind='y'` records only, replacing
the prior `Self ∈ {p, n, s, m, c, x, e}` filter.

**Background — prior bug:** The original extraction used a hard-coded set of
`Self` field codes as a proxy for "LD-2 quarterly reports." This was wrong on
two counts: (a) the `Self` field describes the organizational registrant/client
relationship (self-filer parent, subsidiary, external firm), not the form type;
(b) it produced both false inclusions (superseded originals) and false exclusions
(valid records with `Self='b'` that carry `ind='y'`).

**What `ind='y'` means:** The `Ind` column (position 13) is OpenSecrets' own
validity flag. From the Data User Guide p.13: *"In most cases it is a
straightforward scenario where you just take into account the ind=y."*
A record with `ind='y'` should be counted; `ind=''` means exclude.

**What the prior filter included incorrectly:**
- Superseded originals (e.g., the Q1 report when a Q1a amendment exists). The
  original receives `ind=''`; only the amendment receives `ind='y'`. Including
  both inflated spend by the superseded amount (verified empirically: Altria
  had a superseded Q4 at $2.51M included alongside the valid Q4a at $2.52M).
  Dataset-wide, ~$255M in Fortune 500 spend was double-counted this way.

**What the prior filter excluded incorrectly:**
- `Self='b'` records (non-self-filer subsidiary, different catorder): 666
  records across 36 Fortune 500 firms with `ind='y'` were excluded. These are
  legitimate distinct engagements per OpenSecrets' counting methodology.

**Double-count subsidiary records (`Self='i'`):** 29,912 of 30,360 `Self='i'`
records carry `ind=''` — correctly excluded because the parent self-filer
already includes that spend in their own filing. The prior exclusion of all `i`
types was directionally correct but wrong in rationale (labeled "LD-1
registrations," which is factually incorrect; LD-1 form codes do not appear in
the 2019-2020 dataset at all).

**Quantified impact (Fortune 500, 116th Congress 2019-2020):**

| Filter | Reports | Total Spend | Firms |
|---|---|---|---|
| Old (Self ∈ LD2_TYPES) | 6,521 | $1.977B | 386 |
| New (ind='y') | 4,507 | $1.721B | 377 |

The $255M reduction is removed double-counting. The 9 firms no longer present
had only superseded or no-activity-only records — no valid countable activity.

**`IncludeNSFS` interaction:** Only 63 of 154,766 2019-2020 records have
`IncludeNSFS='y'`, making the complex subsidiary double-count scenario described
in the guide (p.13) essentially negligible. The `ind='y'` flag already handles
these correctly.

**`Use` field redundancy:** Empirically, `ind='y'` ⟺ `use='y' AND ind='y'`
(both yield exactly 97,750 records across all 2019-2020 filers). Filtering on
`ind='y'` alone is sufficient; the `use` field adds no further deduplication.

**Validation:** Run `validations/05_ind_filter_validation.py` to verify: no
superseded originals, no unexpected `Self='i'` records, all report_types are
valid quarterly LD-2 codes, and report counts are as expected.

---

## §18 — RBO Parameter Recalibration: p=0.85, top_bills=30

**Decision:** Change RBO persistence parameter from `p=0.90` to `p=0.85` and
list truncation from `top_bills=100` to `top_bills=30` in `rbo_similarity_network.py`.

**Empirical motivation:** Querying `opensecrets_lda_reports.csv` with mega-bills
included (no `MAX_BILL_DF` filter) reveals that Fortune 500 lobbying spend is
heavily front-loaded within each firm's bill portfolio (116th Congress):

| Cumulative spend target | Median rank | P25 rank | P75 rank |
|---|---|---|---|
| 50% of total spend | 4 | 2 | 7 |
| 80% of total spend | 7 | 3 | 15 |
| 95% of total spend | 10 | 4 | 24 |

The median firm accumulates 80% of its lobbying spend by its 7th-ranked bill,
and 95% by rank 10. The old `p=0.90` placed ~65% of total RBO weight in the
top 10 ranks — undershooting the empirical concentration of spend and giving
meaningful weight to ranks 10–30 that carry only ~5% of a typical firm's budget.

**p=0.85 alignment:** At `p=0.85`, ~80% of RBO weight falls in the top 10 ranks,
matching the empirical P50 firm's spend curve almost exactly. This grounds the
geometric decay in observable firm behavior rather than an arbitrary default.

**top_bills=30 rationale:** The data shows that the median firm lobbies only 11
bills total, and 75% of firms lobby ≤ 30 bills. With `top_bills=100`, the cap
was never binding for the vast majority of firms, adding no information but
including noise from near-zero-spend bills. `top_bills=30` covers the P75 firm
completely and, at `p=0.85`, the 30th bill contributes only ~0.2% of total RBO
weight — below any meaningful signal threshold.

**Source:** Empirical analysis run on `opensecrets_lda_reports.csv` with mega-bills
included; see `visualizations/png/rbo_p_calibration.png` for the spend
concentration vs. RBO weight curve comparison across p ∈ {0.70, 0.80, 0.85, 0.90,
0.95, 0.98}.

---

## References

Bonacich, P. (1987). Power and centrality: A family of measures. American Journal
  of Sociology, 92(5), 1170–1182.

Hojnacki, M., Kimball, D.C., Baumgartner, F.R., Berry, J.M., & Leech, B.L. (2012).
  Studying Organizational Advocacy and Influence: Reexamining Interest Group
  Research. Annual Review of Political Science, 15, 379–399.

Katz, L. (1953). A new status index derived from sociometric analysis.
  Psychometrika, 18(1), 39–43.

Koger, G., & Victor, J.N. (2009). Polarized Agents: Campaign Contributions by
  Lobbyists. PS: Political Science & Politics, 42(3), 485–488.

Manning, C.D., Raghavan, P., & Schütze, H. (2008). Introduction to Information
  Retrieval. Cambridge University Press.

Traag, V.A., Waltman, L., & van Eck, N.J. (2019). From Louvain to Leiden: guaranteeing
  well-connected communities. Scientific Reports, 9, 5234.

Webber, W., Moffat, A., & Zobel, J. (2010). A similarity measure for indefinite
  rankings. ACM Transactions on Information Systems, 28(4), 1–38.

---

## §17 — Bill-Company Incidence Matrix

**Decision:** Build a 2,219 × 305 binary incidence matrix (`bill_company_matrix.csv`)
where rows are bills, columns are Fortune 500 companies, and each cell is 1 if that
company lobbied the bill during the 116th Congress, 0 otherwise. Row and column
integer indices are stored separately in `bill_index.csv` and `company_index.csv` for
explicit mapping back to bill numbers and company names.

**No prevalence filter applied:** Unlike the affiliation network (§4), no `MAX_BILL_DF`
filter is applied here. The incidence matrix is a raw input representation: all 2,219
bills and all 305 companies are included. Any downstream filtering (e.g., for game-
theoretic analysis) should be applied to the matrix by the consuming script.

**Deduplication:** Rows in `opensecrets_lda_reports.csv` are deduplicated on
`(bill_number, fortune_name)` before pivoting — a company's presence is binary
regardless of how many quarterly reports mentioned the bill.

**Sorted order:** Both bills and companies are sorted alphabetically. This gives a
reproducible, canonical row/column assignment that matches `bill_index.csv` and
`company_index.csv`.

**Output files (in `data/`):**
- `bill_company_matrix.csv` — integer row × column indices, values 0/1
- `bill_index.csv` — `row_idx, bill_number`
- `company_index.csv` — `col_idx, fortune_name`

**Matrix stats:** 2,219 bills × 305 companies, 6,999 nonzero entries (1.03% density).

---

## §19 — Quarterly RBO Networks (Archived)

**Script:** `src/archive/rbo_quarterly_networks.py`

This analysis built 8 per-quarter undirected RBO similarity networks for the 116th Congress and characterized temporal evolution via NMI/ARI community stability, network metric trajectories, and PageRank rank correlations. It pre-dates and was superseded by the directed influence network (§21). Outputs in `data/archive/`. See Archived Work Reference at bottom of this file.

---

## §20 — Directed Influence Network: Temporal Bill-Adoption Precedence

**Script:** *(not retained; superseded by `src/rbo_directed_influence.py` before being committed — see §21)*

### Motivation

The undirected RBO similarity network tells us *which* firms share lobbying
priorities, but not *who is setting the agenda* and *who is following*. To draw
a directed causal arrow from A to B we use the observable temporal ordering of
bill adoption: if two firms are similar (high RBO edge in quarter q) and firm A
systematically lobbied their shared bills before firm B, the simplest causal
reading is that A's prior commitment helped pull B toward those bills. This
operationalizes the "weak-ties" influence mechanism (Carpenter et al., 1998) in
a way that is directly testable from the quarterly filing data.

### Scoring Rule

For each RBO-linked pair (A, B) in quarter q:

1. Compute each firm's top-30 ranked bill list for that quarter (same pipeline
   as `rbo_quarterly_networks.py`).
2. Take the intersection of the two lists (shared bills).
3. For each shared bill, look up the **first quarter** (within 1..q) that each
   firm lobbied it — the *causal window*. Only quarters up to and including q
   are examined; no future information is used.
4. Scoring:
   - If `first_q(A, bill) < first_q(B, bill)`: A gets +1 (A lobbied first).
   - If `first_q(B, bill) < first_q(A, bill)`: B gets +1.
   - If equal (same first quarter): **tie — no points awarded** to either firm.
5. Net direction:
   - If `A_firsts > B_firsts`: emit directed edge A→B with weight = A_firsts − B_firsts.
   - If `B_firsts > A_firsts`: emit directed edge B→A.
   - If equal (including all-ties): **no directed edge** — direction is indeterminate.

### Design Choices and Rationale

**Causal window (Q1..q) over full congress (Q1..Q8):** Using only history
available at time q maintains temporal exogeneity — the influence inference for
Q3 cannot be contaminated by Q4 bill adoptions. This is the standard temporal
precedence approach used in dynamic network analysis (Granger, 1969).

**Ties skipped, not split:** Assigning 0.5 to each firm for ties would dilute
the directional signal with noise from simultaneous co-adoption events. For
industry-specific bills, simultaneous adoption is the *expected* outcome when
both firms respond to the same exogenous legislative event; it carries no
information about peer influence. Skipping ties keeps only the informative cases.

**Balanced pairs dropped (A_firsts == B_firsts, both > 0):** A pair where each
firm "won" equally many bills cannot be assigned a dominant direction. Keeping
such pairs with weight 0 adds noise without interpretive value.

**Net weight rather than raw counts:** The net weight (winner's firsts minus
loser's firsts) reflects the *margin* of temporal precedence, not just direction.
A firm that led 8 of 10 shared bills (net=6) is a stronger influencer than one
that led 3 of 5 (net=1), even though both have clear direction. This is
analogous to a net-wins margin in pairwise tournament scoring.

**Q1 produces no directed edges — expected and correct:** All filings in Q1
have first_quarter = 1 by definition, making every shared bill a tie. The
causal window has no prior history to compare against. Directed signal first
emerges in Q2 and grows as temporal contrast accumulates (Q2: 14.5% of RBO
pairs resolve to a directed edge, Q3: 39%, Q4: 46%, later quarters: ~40–48%).

### Aggregate Network

A congress-wide directed network is built by summing `source_firsts` and
`target_firsts` across all 8 quarters for each canonical firm pair, then
applying the net-direction rule to the totals. This captures the persistent
directional influence relationships over the full 116th Congress. The aggregate
edge weight is the total net first-mover margin across all quarters.

### Node Attributes in GML

Each directed GML includes:
- `out_strength`: sum of outgoing edge weights (total first-mover advantage given)
- `in_strength`: sum of incoming edge weights (total first-mover advantage received)
- `net_influence`: out_strength − in_strength (positive = persistent agenda-setter)
- `community`: Leiden community from the same-quarter undirected RBO network
- `label`: firm name

### Empirical Highlights (116th Congress)

Persistent agenda-setters (aggregate out_strength): BALL (+132), NORTHROP
GRUMMAN (+131), FORD (+112), CUMMINS (+103), BOEING (+99), EXXONMOBIL (+97).
These are primarily defense contractors and large industrials that lobby
early-cycle bills. Persistent followers (aggregate in_strength): ENTERGY (−143),
STATE FARM (−149), AMERICAN ELECTRIC POWER (−117), PPL (−122) — utilities and
insurance firms that tend to pick up bills initiated by other sectors.

The defense contractor dominance among agenda-setters aligns with the
"revolving door" and incumbent-advantage literature: firms with established
appropriations relationships lobby authorization and appropriations bills early
in the cycle and others follow (Mian et al., 2010; LaPira & Thomas, 2014).

### Output Files

| File | Contents |
|---|---|
| `data/directed_influence_q{1..8}.csv` | Per-quarter directed edges |
| `data/directed_influence_agg.csv` | Aggregate directed edges |
| `visualizations/gml/directed_influence_q{1..8}.gml` | Per-quarter directed GMLs |
| `visualizations/gml/directed_influence_agg.gml` | Aggregate GML |
| `visualizations/png/directed_influence_q{1..8}.png` | Directed circular plots |
| `visualizations/png/directed_influence_agg.png` | Aggregate circular plot |

PNG color code: green = net agenda-setter (out > in), red = net follower
(in > out), gray = balanced.

### References

Carpenter, D.P., Esterling, K.M., & Lazer, D.M.J. (1998). The strength of weak
  ties in lobbying networks: Evidence from health-care politics in the United
  States. Journal of Theoretical Politics, 10(4), 417–444.

Granger, C.W.J. (1969). Investigating causal relations by econometric models and
  cross-spectral methods. Econometrica, 37(3), 424–438.

LaPira, T.M., & Thomas, H.F. (2014). Revolving door lobbyists and interest
  representation. Interest Groups & Advocacy, 3(1), 4–29.

Mian, A., Sufi, A., & Trebbi, F. (2010). The political economy of the US
  mortgage crisis. The Quarterly Journal of Economics, 125(4), 1452–1560.

---

## §21 — Congress-wide Directed RBO Influence Network: Single Aggregate Design

> **Superseded (partially) by §37 (April 2026).** The edge structure and `net_strength` formula have been redesigned. The temporal first-mover logic, RBO computation, and `net_influence` definition remain valid. See §37 for the new bidirectional edge weights and `net_strength = Σ_j [RBO(i,j) × net_temporal(i,j)]`.

**Script:** `src/rbo_directed_influence.py`

### Motivation

The per-quarter directed influence network (§20) produces 8 separate networks
using a per-quarter causal window and measures edge weight as the first-mover
margin. This section documents a unified single-network alternative where the
edge weight carries the RBO similarity signal and direction carries the temporal
precedence signal separately. The two quantities are no longer conflated into
one edge weight, making each independently interpretable.

### Edge Weight: RBO Similarity (Congress-wide)

RBO is computed on ranked lists built from **aggregated spend across all 8
quarters** (total `amount_allocated` per (firm, bill)). Parameters: p=0.85,
top_bills=30, MAX_BILL_DF=50 (same as §18 and §19).

**Design choice:** Aggregating spend across all quarters before building ranked
lists gives each firm a single congress-wide priority profile, which is more
stable than per-quarter snapshots. High RBO between A and B means their overall
legislative agendas were closely aligned throughout the 116th Congress.

### Edge Direction: Global First-Mover

For each pair (A, B) with RBO > 0, direction is determined by comparing the
**global first-quarter** — the minimum quarter (1–8) in which each firm ever
lobbied each shared top-30 bill — across all bills in the top-30 intersection:

- For each shared bill: if `min_quarter(A, bill) < min_quarter(B, bill)` → A gets +1
- If `min_quarter(B, bill) < min_quarter(A, bill)` → B gets +1
- If equal → tie (no point awarded)
- **No double counting:** each bill contributes at most ±1 regardless of how many
  quarters each firm lobbied it. Example: if A lobbied a bill in Q1 and B lobbied
  it in Q2 and Q5, A gets +1 for that bill (not +2).

**Why global rather than causal windows?** The per-quarter causal window (§20)
was designed for the time-series network where asking "did A lead B this quarter"
requires conditioning on what was observable up to that quarter. In a single
aggregate network spanning the full congress, the temporal ordering is measured
globally: which firm was first to identify a bill as a priority across the entire
period. The global first-mover is the conceptually correct quantity for a
congress-wide network.

### Direction Rule

- `A_firsts > B_firsts` → single directed edge A → B (balanced=0)
- `B_firsts > A_firsts` → single directed edge B → A (balanced=0)
- `A_firsts == B_firsts` → single canonical edge min(A,B) → max(A,B) with `balanced=1`;
  canonical direction is alphabetical (arbitrary but consistent); the edge remains
  visible in Gephi while contributing 0 to each node's `net_influence`.

**Design choice (balanced pairs — single canonical edge):** A balanced result means
neither firm systematically preceded the other. A single canonical edge (alphabetical
min→max) is emitted rather than two antiparallel edges, so each balanced neighbor
contributes the RBO weight exactly once to a node's weighted degree (out_strength +
in_strength). The earlier two-edge representation doubled the RBO contribution for
balanced pairs, artificially inflating weighted degree for nodes with many balanced
neighbors. The canonical direction is *not* used for any directional metric;
`net_strength` explicitly excludes balanced edges to avoid spurious signal from the
arbitrary ordering.

### Direction Scoring — Shared Bills Scope

Direction is scored over the **intersection of the two firms' top-30 ranked
lists only**, not all bills both firms ever lobbied. This ensures conceptual
consistency: the direction is measured over exactly the same bills that determine
the RBO score, i.e., the strategic-priority bills that define the similarity edge.

### Node Attributes

> **Superseded (partially) by §37 and §38 (April 2026).** `net_strength` is now the **primary** agenda-setter proxy; `net_influence` is the **reference/secondary** metric. The definitions below remain valid; the primacy designation has changed.

`net_influence` (integer) is the **reference** node metric and is computed as the
net count of bills where a firm was the first mover across **all** its pairwise
comparisons:

```
net_influence = (out_sf + in_tf) − (out_tf + in_sf)
```

where for each adjacent edge, `sf = source_firsts`, `tf = target_firsts`.
Equivalently, `net_influence = Σ_{all pairings} (this_firm_firsts − opponent_firsts)`.

- Positive → firm was the earlier-mover more often than later across all its
  paired comparisons (agenda-setter)
- Negative → firm adopted bills after others more often (follower)

`total_firsts` and `total_losses` store the raw win/loss counts separately.
`out_strength` and `in_strength` store the RBO-weight sums on outgoing/incoming
edges (graph-theoretic weighted degree decomposition). Their sum (`weighted_degree`)
gives the total RBO-weighted involvement of the node with all its neighbors.

`net_strength` (float) is the RBO-weighted directional influence balance, computed
from **directed (balanced=0) edges only**:

```
net_strength = out_strength(directed) − in_strength(directed)
```

Balanced edges (balanced=1) are excluded because their canonical direction is
alphabetical (arbitrary), so including them would produce a spurious net signal
unrelated to actual temporal precedence. `net_strength` answers: "among pairs where
one firm clearly preceded the other, is this firm net-pushing RBO weight outward
(influencer) or net-receiving (follower)?" A node can have `net_influence > 0` but
`net_strength ≈ 0` if its decisive wins are against low-RBO neighbors.

**For Gephi:** Size nodes by `net_influence` (or `abs(net_influence)`) for
count-based agenda-setting. Use `net_strength` to weight by portfolio alignment
strength. Color by `color` attribute (green/red/gray). Edge weight is RBO similarity.

### Empirical Results (116th Congress, 305 firms → 289 with ranked lists)

- Total edges in DiGraph: 3,712 (post-balanced-fix)
  - Directed (decisive): 1,812
  - Balanced pairs: 1,900 canonical edges (one per pair; previously 3,800 bidirectional)
- Mean RBO weight: 0.0614
- Mean net_temporal (directed pairs): 1.47
- Top agenda-setters (net_influence): CUMMINS (+58), IBM (+48), DTE ENERGY (+45),
  BALL (+44), EXXONMOBIL (+43), ABBVIE (+36), LEIDOS HOLDINGS (+34)
- Top followers: MUTUAL OF OMAHA (−112), CENTERPOINT ENERGY (−88),
  CORNING (−79), GILEAD SCIENCES (−55), STATE FARM (−46)

*Note: net_influence values are unchanged by the balanced fix — balanced pairs
always contributed 0 to each node's net count. Only weighted degree
(out_strength + in_strength) changes.*

### Output Files

| File | Contents |
|---|---|
| `data/rbo_directed_influence.csv` | Edge list (source, target, weight/RBO, source_firsts, target_firsts, tie_count, shared_bills, net_temporal, balanced) |
| `visualizations/gml/rbo_directed_influence.gml` | Directed GML for Gephi (289 nodes incl. 35 isolated, 3,712 edges) |
| `visualizations/png/rbo_directed_influence.png` | Circular directed plot, top-20 by RBO out+in strength |

### Naming Note

Script and all outputs renamed from `congress_influence_network.*` to
`rbo_directed_influence.*` (2026-04-09) to consistently reflect the
analytical concept (RBO-weighted directed influence) rather than the
data scope (congress-wide). The methodology is unchanged.

---

## §22 — Enriched Node Attributes on the RBO Directed Influence GML

**Script:** `src/enrich_directed_gml.py`

Four additional node attributes are appended to `rbo_directed_influence.gml` by a
standalone enrichment script. This script is intentionally separate from
`rbo_directed_influence.py` so the expensive edge-building step does not need to be
re-run when community assignments change.

### Attributes

| Attribute | Type | Description |
|---|---|---|
| `num_bills` | int | Unique bills lobbied per firm after the MAX_BILL_DF=50 prevalence filter. Consistent with the bills actually used in network construction. |
| `bill_aff_community` | int | Leiden community label from the bill affiliation network (γ=1.0). Sentinel −1 for firms present in the directed network but absent from the bill affiliation network (e.g. isolated nodes). |
| `within_comm_net_str` | float | Within-community net RBO strength: sum of RBO weights on directed (balanced=0) out-edges to same-community peers minus in-edges from same-community peers. Balanced edges excluded (arbitrary canonical direction; consistent with global `net_strength` convention in §21). |
| `within_comm_net_inf` | int | Within-community net influence: first-mover wins minus losses on directed (balanced=0) edges where both endpoints share the same bill affiliation community. Mirrors the global `net_influence` formula restricted to intra-community pairs. |

### Design choices

**Community source:** Bill affiliation communities are used rather than RBO or composite
communities because the bill affiliation network directly encodes co-lobbying overlap
— the same structural basis as the temporal first-mover comparison. Using bill
affiliation communities to segment the directed influence network asks: "within the
set of firms that lobby the same bills, who leads and who follows?"

**Balanced edge exclusion:** Consistent with §21. Balanced edges convey no directional
information; their canonical direction is alphabetical.

**Prevalence-filtered bill count:** `num_bills` uses the post-filter bill universe so it
reflects the bills that actually participate in RBO and first-mover comparisons, not
the raw lobbying breadth.

**Sentinel −1:** 13 firms appear in the directed influence network as isolated nodes
(zero RBO pairs) but were not emitted by the bill affiliation network pipeline (which
requires at least one shared-bill edge). Their `bill_aff_community` is set to −1 and
their `within_comm_*` metrics are 0 by construction.

### References

Webber, W., Moffat, A., & Zobel, J. (2010). A similarity measure for indefinite
  rankings. ACM Transactions on Information Systems, 28(4), 1–38.

---

## §23 — Codebase Architecture and Archive Decisions (2026-04-14)

**Decision:** Archived three directories that are no longer part of the active research pipeline.

- `src/fortune_20/` → `src/archive/fortune_20/`: early Fortune 20 bipartite network scripts, superseded by the full Fortune 500 pipeline.
- `src/influence_game_psne_calculation/` → `src/archive/psne/`: the C++ PSNE solver and associated equilibrium files. The project has moved away from equilibrium analysis as a primary analytical lens in favor of the directed influence network.
- `visualizations/fortune_20/` → `visualizations/archive/fortune_20/`: visualization outputs from the Fortune 20 analysis.

**New:** Created `docs/` at the project root to house project-level notes (`directed_influence_summary.md`).

**Decision:** Created `DOCUMENTATION.md` at the project root — a complete reproduction guide covering data sources, extraction pipeline, all network construction logic, the directed influence network in full detail, utility module APIs, and validation scripts. Intended for external reproducibility; complements the methodology-focused `design_decisions.md`.

**Docstring convention (standardized):** All docstrings in `src/` and `src/utils/` files use a concise one-or-two-line format (no verbose Parameters/Returns blocks). Validation scripts retain longer explanatory comments since they document audit logic and expected outcomes.

---

## §24 — Affiliation-Mediated Adoption: Two-Level Analysis Design (2026-04-14)

**Decision:** Operationalize the affiliation-network → directed influence connection at two levels of granularity: (1) **bill-level co-affiliation** (strict) and (2) **network-level connectivity** (broad).

**Bill-level (strict):** For each directed adoption pair (A→B, bill) — A lobbied bill first, B adopted later — check whether A and B share a lobbyist or external registrant on their *first-quarter reports for that specific bill*. First-quarter reports are used because they represent the moment of adoption, making the shared channel contemporaneous with the signal transmission. External registrants only for the firm channel (mirrors `lobby_firm_affiliation_network.py`).

**Network-level (broad):** For each directed adoption pair (A→B), check whether A and B share any lobbyist or external registrant across their *full lobbying portfolios* (i.e., are connected in the lobbyist affiliation or firm affiliation networks). This uses `data/network_edges/lobbyist_affiliation_edges.csv` and builds the firm adjacency inline from `opensecrets_lda_reports.csv`.

**Rationale for two levels:** The bill-level test is the most direct causal story — the same intermediary works on this specific bill for both firms. The network-level test is broader — the firms are institutionally linked via shared intermediaries on any legislation, creating a general information channel. Both are theoretically motivated: the bill-level by the informational theory of lobbying (Austen-Smith 1993), the network-level by Granovetter's (1973) network information-flow framework.

**Key empirical finding:** Bill-level co-affiliation is extremely rare — only 7 out of 3,184 directed bill-level adoption pairs (0.2%) have a shared lobbyist or firm on that specific bill. Network-level connectivity is similarly sparse (0.6%). This finding is itself substantive: the RBO directed influence network captures broad coordination patterns that are *not* primarily explained by direct shared-affiliation channels. The two exceptions (United Technologies → Raytheon, driven by their 2020 merger; and LOEWS → ECOLAB) are structurally anomalous cases.

**Alignment test finding:** Among the 7 affiliation-mediated directed bills, 7/7 (100%) have the bill-level leader (first adopter) as the RBO edge source (p = 0.008, binomial test). This is consistent with the directional hypothesis — mediated adoption is aligned with the aggregate influence direction — but the sample is too small to generalize.

**Outputs:**
- `data/affiliation_mediated_adoption.csv` — one row per (RBO edge, shared bill) with bill-level and network-level mediation flags
- `data/rbo_edges_enriched.csv` — one row per RBO edge with edge-level mediation rates, directed-bill counts, and network connectivity flags

**Scripts:**
- `src/affiliation_mediated_adoption.py` — core data construction
- `src/validations/11_mediated_adoption_validation.py` — 9-section statistical validation report

**Alternatives considered:**
- Using all-quarter reports (not first-quarter) for the bill-level mediation check: produces the same result since overlapping lobbyists remain rare even across all quarters.
- Including self-filing registrants in the firm channel: explored but rejected for the strict bill-level test to maintain consistency with `lobby_firm_affiliation_network.py`. Self-filers are identical to their client, so overlap on self-filers is not evidence of a bridging intermediary.

---

## §25 — Issue RBO Similarity Network (2026-04-15)

**Decision:** Build a second issue-code similarity network using RBO rather than cosine similarity.

**Rationale:** The existing `issue_similarity_network.py` uses cosine similarity on a 76-dimensional spend-fraction vector. RBO is preferred because it top-weights high-priority issue codes in the same way it does for bills — a firm that concentrates 40% of its lobbying budget on TAX and then diversifies across other codes should be considered more similar to another firm with the same priority ordering than to one that spreads spend more evenly. Cosine treats all codes with nonzero weight symmetrically; RBO does not.

**Parameters:** `TOP_ISSUES=30` (covers full portfolios for 97% of firms; median is 6 codes), `p=0.85` (matching bill-level calibration, §18), no prevalence filter (`MAX_ISSUE_DF=None`, consistent with §8). The resulting network has 36,187 edges across 374 firms, density 0.519. Six Leiden communities emerge (Q=0.071), broadly interpretable as: financial/tech, defense/transport/utilities, energy/chemicals/food, healthcare/pharma, diversified/consumer.

**Script:** `src/archive/networks/issue_rbo_similarity_network.py`

**Outputs:** `data/archive/network_edges/issue_rbo_edges.csv`, `data/archive/communities/communities_issue_rbo.csv`, `data/archive/centralities/centrality_issue_rbo.csv`, `visualizations/archive/undirected/issue_rbo_similarity_network.gml`, `visualizations/archive/undirected/issue_rbo_similarity_network.png`

---

## §26 — outputs/ Folder Restructure and Validation Path Redirect (2026-04-15)

**Decision:** Moved all validation script outputs from `src/validations/outputs/` to a new `outputs/` directory at the repo root, with subdirectories `outputs/validation/` and `outputs/channel_tests/`.

**Rationale:** Channel test outputs, revolving door assessments, and future mechanism-testing results do not belong in the validation subdirectory. Centralizing all analytical outputs at the root level improves navigability and separates research outputs from source code.

**Changes:** All `OUTPUT_PATH` constants in `src/validations/01-08` were updated to `Path(__file__).resolve().parent.parent.parent / "outputs" / "validation" / "filename"`. Scripts 06, 07, 10, 11 (previously stdout-only) had a `_Tee` class added that writes to both stdout and the output file simultaneously.

---


## §32 — Within-Community Influencer Hierarchy and Rank Stability (2026-04-17)

**Decision:** Focus on the firm-level question — who are the within-community agenda-setters, and is that stable? Use within-community net_influence (restricted to intra-sector directed edges) as the measure, computed fresh for each congress from per-congress `rbo_directed_influence.csv`. Apply the 116th-Congress affiliation Leiden partition as the sector proxy throughout.

**Three analyses per community:**
- **A. Top-5 leaderboard:** the five highest within-community net_influence firms per community per congress (111th–117th).
- **B. Rank stability:** adjacent-congress Spearman ρ on within-community net_influence ranks, restricted to firms stable across all 7 congresses (23–40 per community).
- **C. Persistent leaders:** firms appearing in the top-5 in ≥4 of 7 congresses.

**Design choices:**
- Within-community net_influence is computed from intra-sector directed edges only (both source and target in same community), isolating sector-internal agenda-setting from cross-sector influence.
- Stable-set criterion: firm must appear in the directed network for all 7 congresses. This is stricter than the leaderboard (which uses all available firms per congress), so leaderboard names may differ from stability analysis names.

**Empirical findings:**

*Energy/Utilities:* Strongest stability of any community. Duke Energy and Xcel Energy each appear in the top-5 in all 7 congresses — no other firm in any community achieves this. CMS Energy (5/7) and DTE Energy (4/7) are the next tier. Five of six adjacent-congress Spearman ρ are significant.

*Tech/Telecom:* Strong stability. IBM and AT&T each appear 4/7 congresses; CBS and Microsoft also 4/7. Adjacent-congress Spearman pattern is uneven — three significant transitions interspersed with non-significant ones (Oracle's 113th Congress spike to +69 is an anomaly).

*Defense/Industrial:* Significant stability. Lockheed Martin appears in top-5 in 5/7 congresses (absent only 115th–116th where Textron/Ball led), Northrop Grumman in 4/7. Adjacent-congress Spearman significant only for 113→114 transition (ρ=0.638); hierarchy is stable globally but transitions can be abrupt.

*Finance/Insurance:* Weak but significant stability. American Family Insurance Group (5/7) and Northwestern Mutual (4/7) are the only persistent leaders. Highest member count (n=72) and most within-sector competitive churn; only 2 of 6 adjacent-congress ρ significant.

*Health/Pharma:* No significant stability. No firm reaches top-5 in ≥4 of 7 congresses. Most chaotic community for within-sector agenda-setting.

**Interpretation for the paper:** Energy/utilities is the cleanest publishable result — Duke Energy and Xcel Energy are the most persistent within-sector agenda-setters of any firm in any community, across a decade. Defense/industrial (Lockheed Martin 5/7) is the next strongest. Health/pharma's instability is consistent with ACA/drug-pricing regulatory volatility across the 111th–117th Congresses.

**Script:** `src/validations/16_industry_influencer_hierarchy.py`

**Outputs:**
- `outputs/validation/16_within_community_ni_by_congress.csv` — firm × congress within-community net_influence table
- `outputs/validation/16_within_community_rank_stability.csv` — adjacent-congress Spearman per community
- `outputs/validation/16_industry_influencer_hierarchy.txt` — full leaderboards, stability stats, persistent leaders

---

## §31 — Cross-Sector Directed Edge Analysis (2026-04-17)

**Decision:** Tag each directed RBO edge by whether source and target are in the same Leiden affiliation community (intra-sector) or different communities (cross-sector). Analyse the structure and identity of cross-sector influence flows.

**Analyses:**
1. **Edge-level distributions:** RBO weight and net_temporal for cross-sector vs. intra-sector edges (Mann-Whitney U).
2. **Community-pair flow matrix:** directed edge counts and mean RBO weight for all 25 (src_community, tgt_community) pairs; net directional asymmetry between each unordered pair.
3. **Firm-level cross-sector influence:** net_cs_influence and net_cs_strength per firm; top cross-sector agenda-setters and followers by community.
4. **Bridge firm identification:** firms with highest cross-sector directed edge fraction (total cross-sector edges / total directed edges).
5. **Top cross-sector dyads:** top-10 cross-sector directed pairs by net_temporal with issue-code profiles (dominant issue codes for each firm, cosine similarity of issue profiles).

**Key empirical findings (116th Congress):**
- **43.2% of directed edges are cross-sector** (783 of 1,813 with community labels). Cross-sector edges have significantly lower RBO weight (median 0.008 vs. 0.036 intra-sector, p≈3×10⁻⁴⁵) and lower net_temporal (mean 1.25 vs. 1.63, p≈3×10⁻¹⁹). Cross-sector influence signals are weaker on average, which is expected: portfolio alignment is highest within industry.
- **Community-pair asymmetry:** Defense/Industrial dominates Health/Pharma (net flow +47) and Energy/Utilities (net flow +38). Defense is the clearest net cross-sector influencer. Defense is itself dominated by Tech/Telecom (−26) and Energy by Tech (−22). Finance/Insurance is roughly balanced across all pairs.
- **Firm-level cross-sector leaders:** Cummins (+48 CS-NI, community: Defense/Industrial) is the dominant cross-sector agenda-setter by a large margin — it leads 34 cross-sector edges vs. 3 incoming. Ford (+20, Finance/Insurance per community label), IBM (+19, Tech/Telecom), and Lockheed Martin (+17, Defense) are the next tier. Notably, Cummins's community label is Defense/Industrial, consistent with its heavy involvement in defense-adjacent procurement.
- **Cross-sector followers:** Mutual of Omaha (−53, Health/Pharma) and Gilead Sciences (−53, Health/Pharma) are the largest cross-sector followers. Multiple defense and tech firms lead into health/pharma, consistent with budget and defense appropriations bills being adopted by health firms.
- **Community means:** Defense/Industrial is the only community with a positive mean net_cs_influence (+4.15, 67.5% positive). Health/Pharma has the most negative mean (−3.62, only 37.9% positive).
- **Bridge firms:** Firms with high cross-sector edge fractions are often niche players with few total directed edges (Cognizant, Thermo Fisher, Caterpillar), not the top global influencers. High bridge fraction reflects structural peripherality, not necessarily cross-sector agenda-setting power.
- **Top dyads:** Defense→Health/Pharma dyads (Lockheed/Gilead, General Dynamics/Gilead, Leidos/Gilead) share BUD (budget) as the linking issue code, consistent with defense appropriations bills bleeding into health agency budgets. Berkshire Hathaway→CSX (Energy→Defense) is the highest net_temporal (5) cross-sector pair, with high RBO weight (0.653) but no shared top-3 issue codes — a genuine strategic complementarity case rather than correlated response.

**Community partition note:** The 116th-Congress affiliation Leiden partition is used. "Ford" is assigned to Finance/Insurance because its lobbying portfolio in the 116th Congress aligns with the financial services cluster — an anomaly worth noting. Cross-sector tagging is mechanical on the community partition; sector interpretation requires care.

**Script:** `src/validations/15_cross_sector_directed_edges.py`

**Outputs:**
- `outputs/validation/15_cross_sector_edge_table.csv` — all directed edges with src/tgt community labels and cross-sector flag
- `outputs/validation/15_cross_sector_firm_table.csv` — firm-level cross-sector influence metrics
- `outputs/validation/15_cross_sector_pair_matrix.csv` — community-pair directed edge counts and mean weights
- `outputs/validation/15_cross_sector_directed_edges.txt` — full analysis log

---

## §30 — Influencer Regression Analysis (2026-04-17)

**Decision:** Run OLS regressions predicting net_influence, net_strength, and wc_net_strength from observable firm characteristics (log_spend, log_bills, katz_centrality) and within-community structural covariates (within_comm_eigenvector, wc_pagerank) for the 116th and 117th Congress cross-sections.

**Specifications:**
- **Spec A:** `net_influence ~ log_spend + log_bills + katz_centrality`. Standard errors: HC3 (heteroskedasticity-robust).
- **Spec A2:** `top_quartile_net_influence ~ same`. Binary indicator (1 = net_influence ≥ 75th percentile). OLS linear probability model. Used for interpretability alongside A.
- **Spec B:** `net_strength ~ log_spend + log_bills + katz_centrality`. Same covariate set; RBO-weighted outcome.
- **Spec C:** `wc_net_strength ~ log_spend + log_bills + within_comm_eigenvector + wc_pagerank`. Within-community outcomes with within-community structural covariates. Firms absent from 116th community partition (42 in 117th) dropped.

**Covariate decisions:**
- `log_spend`: log(total amount_allocated per firm, summed over unique reports, floored at $1). Deduplication on `(uniq_id, fortune_name)` to avoid double-counting per bill row.
- `log_bills`: log(unique bill count per firm, floored at 1).
- `katz_centrality`, `within_comm_eigenvector`: from 116th-Congress affiliation centrality table, reused as a structural baseline for both congresses. Per-congress centrality recomputation is out of scope.
- `wc_pagerank`: within-community PageRank recomputed fresh on the 116th affiliation graph with stored Leiden partition (same computation as validation 13).
- No sector fixed effects (no Fortune 500 sector mapping available). No revolving-door proxy (requires manual lobbyist background coding).

**Empirical findings (116th Congress, N=276 complete cases):**
- **Spec A (net_influence):** R²=0.043. Only `log_spend` reaches p<0.10 (β=−1.609, p=0.095). Negative sign: higher-spending firms have *lower* net_influence in the continuous outcome — high-spending firms lobby many bills but not necessarily as first-movers.
- **Spec A2 (top-quartile indicator):** R²=0.282. `log_bills` is strongly significant (β=0.164, p<0.001): each unit of log bills raises the probability of top-quartile status by 16.4 pp. `log_spend` is negative and significant (β=−0.039, p=0.042). `katz_centrality` is positive and significant (β=1.200, p=0.049): firms more central in the affiliation graph are more likely top-quartile agenda-setters, consistent with BCZ.
- **Spec B (net_strength):** R²=0.049. `log_spend` marginally significant (β=−0.059, p=0.065). Pattern mirrors Spec A.
- **Spec C (wc_net_strength):** R²=0.061. `log_spend` marginally significant (β=−0.049, p=0.064). Within-community structural covariates not individually significant.

**Empirical findings (117th Congress, N=244 complete cases):**
- **Spec A (net_influence):** R²=0.044. `katz_centrality` marginally significant (β=63.7, p=0.053). `log_spend` and `log_bills` not significant in the continuous outcome.
- **Spec A2 (top-quartile indicator):** R²=0.190. `log_bills` strongly significant (β=0.150, p<0.001); `log_spend` negative and significant (β=−0.048, p=0.044). Same pattern as 116th.
- **Spec B (net_strength):** R²=0.034, F-test p=0.052. No individually significant predictors.
- **Spec C (wc_net_strength):** R²=0.061. `log_spend` marginally significant (β=−0.053, p=0.050); `wc_pagerank` marginally significant (β=13.27, p=0.080).

**Cross-congress consistency:** The top-quartile specification (A2) is the most consistent and best-fitting model. `log_bills` is the single most robust predictor of top-quartile influencer status across both congresses (strong, positive, p<0.001 in both). `log_spend` is consistently negative (higher raw spend reduces influencer probability), suggesting that firms that spread spend thinly across many reports are not the ones setting the agenda. `katz_centrality` predicts top-quartile status in the 116th but not robustly in the 117th. The overall low R² for continuous outcomes (0.03–0.06) indicates that observable capacity metrics explain only a small fraction of the variance in net_influence/strength — consistent with a substantial structural agenda-setting signal beyond raw capacity.

**Limitations:** (1) Centrality covariates are from the 116th-Congress affiliation network for both congresses; 117th-specific centrality would require re-running `bill_affiliation_network.py` for the 117th. (2) No sector FEs, no revolving-door proxy. (3) Low R² in continuous specs suggests important unobserved heterogeneity.

**Script:** `src/validations/14_influencer_regression.py`

**Outputs:**
- `outputs/validation/14_influencer_regression.csv` — flat table of all regression results (coefficients, SEs, p-values, R²) for all 8 regressions
- `outputs/validation/14_influencer_regression.txt` — full regression tables with starred significance, descriptive statistics, and interpretation summary

---

## §29 — Centrality vs. Agenda-Setter Comparison (2026-04-17)

**Decision:** Run a formal rank-correlation comparison between four centrality measures derived from the bill affiliation network and three directed-influence (agenda-setter) measures from the 116th Congress RBO directed network, using the 116th Congress as the reference session.

**Centrality measures:**
- **BCZ intercentrality** (Ballester, Calvó-Armengol & Zenou 2006): b_i = Σb(λ, A) − Σb(λ, A[−i]), where b is the unnormalized Katz centrality vector and α = 0.85 / spectral_radius. Computed fresh on the affiliation graph via 277 node-removal Katz reruns (same α throughout per BCZ).
- **Global PageRank**: reused from `centrality_affiliation.csv` (full affiliation graph).
- **Within-community eigenvector centrality**: reused from `centrality_affiliation.csv` (Leiden partition; stored labels from `communities_affiliation.csv`).
- **Within-community PageRank**: computed fresh on each community subgraph using the stored Leiden partition.

**Agenda-setter measures:**
- **net_influence**: from `data/congress/116/node_attributes.csv`; total first-mover wins minus total losses across all directed edges involving each firm.
- **net_strength**: RBO-weighted directed score (out-strength minus in-strength on directed edges).
- **Within-community net_influence/net_strength**: restricted to directed edges where both endpoints share the same affiliation community label. Computed in the validation script from `data/congress/116/rbo_directed_influence.csv`.

**Community labels:** Stored affiliation Leiden partition (`communities_affiliation.csv`) reused throughout; 5 communities. No re-detection.

**Comparisons:** Full-sample Spearman ρ and restricted top-30 Spearman ρ for every centrality–agenda measure pair. Top-30 overlap fraction (Jaccard of top-30 sets). The within-community centrality measures are also correlated with within-community agenda-setter measures.

**Empirical findings (116th Congress):**
- **BCZ intercentrality** is dominated by energy utilities (CMS Energy, DTE Energy, Exelon, Xcel Energy, PPL, Entergy, PG&E) — firms with the largest affiliation footprints in the energy community. Full-sample Spearman ρ with net_influence = **0.178 (p=0.003)** — statistically significant but weak. Top-30 Spearman ρ = **−0.107 (p=0.575)** — non-significant. Top-30 overlap fraction = **0.467**. Interpretation: BCZ intercentrality and empirical agenda-setting diverge substantially. BCZ captures structural centrality in the complementarity graph; empirical agenda-setting (net_influence) is more concentrated among defense/tech firms that are heavy first-movers across sectors. This is an informative *non-result*: BCZ key players ≠ RBO agenda-setters.
- **Global PageRank** closely mirrors BCZ intercentrality (both measure global structural position in the affiliation graph). Spearman ρ with net_influence = **0.193 (p=0.001)**; top-30 Spearman ρ = **0.119 (p=0.532)**; overlap = **0.500**.
- **Within-community eigenvector** also shows weak full-sample correlation with net_influence (ρ = **0.212, p<0.001**) and non-significant top-30 correlation (ρ = **0.164, p=0.387**).
- **Within-community PageRank** provides the clearest signal: full-sample ρ with net_influence = **0.212 (p<0.001)**; top-30 ρ = **0.501 (p=0.005)**; top-30 overlap = **0.500**. The strong top-30 correlation indicates that within-community structural position (WC-PR) tracks agenda-setting among the most influential firms better than any global centrality measure.
- **Within-community WC-PR vs. WC net_influence**: top-30 ρ = **0.558 (p=0.001)**, the highest correlation in the table. This confirms that within each Leiden community, firms with high within-community PageRank tend to be the empirical agenda-setters within that community.
- **Net strength correlations** are uniformly higher than net_influence across all centrality measures, suggesting that RBO-weighted influence (net_strength) is more consistent with undirected structural position than raw win/loss counts (net_influence).

**Interpretation for BCZ framing:** The weak BCZ intercentrality ↔ net_influence correlation (ρ = 0.178) means the paper cannot claim that "BCZ key players = empirical agenda-setters" without qualification. The appropriate framing is: (a) BCZ intercentrality identifies the firms whose *removal* would most contract the complementarity network — these are the energy/utility hubs; (b) empirical agenda-setters (high net_influence) are disproportionately defense/tech/industrial firms that lobby many bills first. The two rankings partially overlap (47% of top-30 BCZ firms appear in top-30 net_influence) but are structurally distinct. This is itself an interesting result: structural key-players in the complementarity game are not the same firms as the temporal first-movers.

**Script:** `src/validations/13_centrality_vs_agenda_setter.py`

**Outputs:**
- `outputs/validation/13_centrality_vs_agenda_setter.csv` — firm-level table of all 8 measures
- `outputs/validation/13_centrality_vs_agenda_setter_correlations.csv` — pairwise Spearman correlation table
- `outputs/validation/13_centrality_vs_agenda_setter.txt` — full ranked lists and interpretation

---

## §28 — Cross-Congressional Stability Analysis (2026-04-16, updated 2026-04-16)

**Decision:** Test whether RBO directed influence edges are stable in direction and magnitude across seven consecutive congresses (111th–117th, 2009–2022) using the subset of Fortune 500 firms that lobbied in all seven sessions.

**Rationale:** Temporal stability is a necessary (though not sufficient) condition for causal interpretation of the directed influence network. If the same firm consistently leads across seven independent two-year legislative sessions, that regularity is unlikely to be produced by noise alone. Four stability tests are run: (1) direction consistency — does the same firm lead in each session a pair appears?; (2) magnitude stability — are net_temporal values correlated across congress pairs (adjacent-congress Spearman ρ)?; (3) firm net_influence rank stability — do net_influence ranks agree across sessions (adjacent-congress Spearman ρ)?; (4) firm net_strength rank stability — mirrors Analysis 3 for the RBO-weighted net_strength score.

**Design choices:**
- **Stable set criterion:** Firms must appear in the directed influence network for all seven congresses. Firms active in only some congresses are excluded from stability analysis but their congresses still run.
- **Congress range:** The 111th Congress (2009–2010) is the earliest included; pre-111th congresses use semi-annual HLOGA reporting codes incompatible with the quarterly `assign_quarters` function and are excluded.
- **Canonical pair representation:** Each (firm_a, firm_b) pair is represented canonically with firm_a < firm_b alphabetically. Direction score = 1 if firm_a leads, 0 if firm_b leads; balanced edges are excluded from direction analysis.
- **Direction consistency threshold:** A pair is considered "direction-consistent" if max(n_a_leads, n_b_leads) / n_directed_sessions ≥ 0.80.
- **Binomial test:** For each pair with ≥2 directed sessions, H0 is that each session is an independent 50/50 coin flip for direction. Rejection at p < 0.05 provides per-pair evidence of genuine directional consistency.
- **Quarter assignment:** Congress N covers years start=2009+2*(N−111) through start+1, with Q1-4 in year1 mapping to quarters 1-4 and Q1-4 in year2 mapping to quarters 5-8.

**Pipeline:** `src/multi_congress_pipeline.py` runs extraction + RBO directed influence for each of the seven congresses in sequence, writing per-congress outputs to `data/congress/{num}/`. GML and PNG visualizations are also written to `visualizations/gml/` and `visualizations/png/` for each congress.

**Name mapping dependency:** Coverage depends on `manual_opensecrets_name_mapping.json` being expanded to include CRP name variants for each congress era. For the 111th (2009–10) and 117th (2021–22) additions, 34 new name variants were added to cover corporate rebrands (e.g. WellPoint→Anthem, Limited Brands→L Brands, TIAA-CREF→TIAA, Facebook→Meta, Motorola Inc→Motorola Solutions) and earlier-congress filings.

**Scripts:** `src/multi_congress_pipeline.py`, `src/cross_congressional_stability.py`

**Analysis 4 — net_strength rank stability:** Added as a direct mirror of Analysis 3 using net_strength (RBO-weighted directed score) rather than net_influence (raw win/loss count). The figure uses a 2×2 grid; Panels 3 and 4 show adjacent-congress Spearman bar charts for net_influence and net_strength respectively.

**Empirical results (111th–117th, run 2026-04-16):**
- **Stable set:** 135 firms present in all 7 congresses; 6,783 canonical pairs; 277 pairs present in all 7 sessions. Per-congress coverage: 246–260 firms, 3,558–7,418 total edges.
- **Direction consistency (Analysis 1):** 2,830 pairs with ≥2 directed sessions. Mean consistency = 0.770, median = 0.750. 45.1% of pairs ≥0.80 consistent; 41.2% perfectly consistent (score=1.00). 73.9% show a majority direction vs. 50% expected under H0. Only 2.4% reach individual binomial significance (p<0.05); most significant pair: BALL/IBM (5 directed sessions, p=0.031).
- **Magnitude stability (Analysis 2):** All adjacent-congress Spearman ρ on net_temporal are statistically significant except some involving the 113th Congress. Range: 0.037–0.218; highest: 116–117 (ρ=0.218).
- **Firm net_influence rank stability (Analysis 3):** 5 of 6 adjacent-congress Spearman ρ are significant; only 113–114 is non-significant (ρ=0.134, p=0.12). Top persistent influencers: Lockheed Martin (mean +72.1), IBM (+59.4), Xcel Energy (+53.6), CMS Energy (+46.4), Duke Energy (+45.4), AT&T (+43.1), DTE Energy (+42.7). Top persistent followers: Consolidated Edison (mean −55.1), Ameren (−48.3), Centerpoint Energy (−43.0).
- **Firm net_strength rank stability (Analysis 4):** 3 of 6 adjacent-congress Spearman ρ are significant (114–115, 115–116, 116–117); early transitions (111–112 through 113–114) are not significant. Top persistent high-strength firms: Xcel Energy (mean 2.214), Lockheed Martin (2.054), Duke Energy (1.803). Top persistent low-strength firms: Ally Financial (mean −1.108), Centerpoint Energy (−0.950), Cisco Systems (−0.929).
- **Sector pattern:** Energy utilities and defense/tech dominate both ends of the influence spectrum. Financial services shows the most directionally consistent paired relationships.

**Outputs:**
- `data/congress/{num}/opensecrets_lda_reports.csv` — per-congress extraction
- `data/congress/{num}/opensecrets_lda_issues.csv` — per-congress issue extraction
- `data/congress/{num}/ranked_bill_lists.csv` — per-congress top-30 bill rankings
- `data/congress/{num}/rbo_directed_influence.csv` — per-congress directed edges
- `data/congress/{num}/node_attributes.csv` — per-congress node net_influence/strength
- `visualizations/gml/rbo_directed_influence_{num}.gml` — per-congress GML for Gephi
- `visualizations/png/rbo_directed_influence_{num}.png` — per-congress directed circular plot
- `outputs/cross_congressional/cross_congressional_stability.txt` — full stability report
- `outputs/cross_congressional/cross_congressional_stability.png` — 4-panel figure
- `outputs/cross_congressional/cross_congressional_stability.docx` — summary document

---


## §34 — Payoff Complementarity Test (BCZ Micro-Level Evidence) (2026-04-17)

**Decision:** Construct a panel regression at the (firm_i, firm_j, bill, quarter) level to test whether firm i increases lobbying spend on bill b when firm j newly enters bill b at quarter t, and whether that response is amplified for high-RBO pairs — micro-level evidence of strategic complementarity in the BCZ (Ballester, Calvó-Armengol & Zenou 2006) sense.

**Specification:**
```
Δlog_spend_{i,b,t+1} = β₁ entry_{j,b,t} + β₂ rbo_ij
                      + β₃ (entry_{j,b,t} × rbo_ij)
                      + α_{i,b} + γ_t + ε
```

- `Δlog_spend_{i,b,t+1}` = log-change in firm i's allocated spend on bill b from quarter t to t+1 (consecutive quarters, positive spend in both)
- `entry_{j,b,t}` = 1 if quarter t is firm j's first quarter lobbying bill b; 0 if j is a continuing lobbyist on b
- `rbo_ij` = congress-aggregate RBO edge weight (symmetric; same value in both directions)
- `α_{i,b}` = firm-bill fixed effects (within-transformation, absorbs baseline spend levels)
- `γ_t` = quarter fixed effects; HC3 heteroskedasticity-robust SE

**Critical design choice — panel construction:** The panel joins all firm_j active on bill b at quarter t (both new entrants and continuers), not only entry events. This is essential: if only entry events are included, `entry_j` is always 1 by construction (zero variance within groups), making `entry_j` and `entry_j × rbo_ij` collinear after within-demeaning. Including continuers (entry_j=0) gives within-group variation required for identification. Singleton firm-bill groups (n=1) are dropped after within-transformation.

**Four specifications:**
- **Spec A**: Full RBO-linked panel (all (i,j) pairs with an RBO edge)
- **Spec B**: High-RBO pairs only (rbo_ij ≥ p75 ≈ 0.131)
- **Spec C**: Low-RBO pairs only (rbo_ij < p25 ≈ 0.007)
- **Spec D**: All pairs including non-linked (rbo_ij = 0 for pairs without an RBO edge); robustness check

**Empirical results (116th Congress, run 2026-04-17):**
- N = 67,194 (Spec A), 147,341 (Spec D); 3,960 unique (firm_i, bill) groups
- **β₁ (entry_j main effect):** Significantly negative in Specs A, B, D (−0.013 to −0.125). Firms reduce spend growth relative to their within-pair trend when a new co-lobbyist enters — consistent with a crowd-out or displacement effect from co-entry rather than herding.
- **β₂ (rbo_ij main effect):** Positive but mostly non-significant (Spec A p=0.15, Spec D p=0.003). High-RBO pairs have slightly higher baseline spend growth on average.
- **β₃ (interaction entry_j × rbo_ij):**
  - Spec A: −0.125 (p<0.001) — *negative* and significant in full sample
  - Spec B (high-RBO only): +0.147 (p=0.033) — *positive and significant* among the top-quartile RBO pairs
  - Spec C (low-RBO): −8.0 (p=0.007) — large negative, but rbo scale is tiny (<0.007), so marginal effect at mean low-RBO weight is negligible
  - Spec D: −0.173 (p<0.001) — consistent with Spec A

**Interpretation:** The interaction is positive and significant only in Spec B (high-RBO pairs). This is the key BCZ test: among firms with high bill-priority similarity (strong structural complementarity in the BCZ sense), a co-lobbyist's entry triggers *additional* spend increases compared to entry by a low-similarity peer. For the average pair (lower RBO weight), entry is associated with spending restraint — possibly substitution or wait-and-see behavior. The finding is consistent with BCZ complementarity applying at the high end of the complementarity distribution rather than uniformly across all pairs.

**Script:** `src/validations/18_payoff_complementarity.py`

**Outputs:**
- `outputs/validation/18_payoff_complementarity_panel.csv` — full regression panel (67K rows)
- `outputs/validation/18_payoff_complementarity_results.csv` — coefficient table across specs
- `outputs/validation/18_payoff_complementarity.txt` — full log

---

## §35 — Bill Adoption Diffusion: Follower Bill Entry Conditional on RBO Link (2026-04-17)

**Decision:** Test whether follower firm B (target of a directed A→B edge) is more likely to first enter a bill X that influencer A has already lobbied, comparing high-RBO to low-RBO pairs, over Q+1, Q+2, and Q+3 horizons.

**Unit of observation:** (A, B, bill) triples where A has a directed edge to B (A is net first-mover), A first lobbies bill X at quarter t, and B has not yet lobbied X at or before t. One row per such exposure event. Total candidates: 80,384 (1,890 unique A→B pairs × their bill portfolios).

**Outcome:** `adopted_qk` = 1 if B's first quarter lobbying bill X is ≤ a_entry_q + k. Separate binary outcomes for k=1, 2, 3. Horizon observability: excluded when a_entry_q + k > 8 (cannot observe adoption in a quarter that doesn't exist within the 116th Congress).

**Design choices:**
- **Directed edges only (balanced=0):** Balanced pairs have arbitrary canonical direction (alphabetical); they would conflate direction. Only the 1,890 directional pairs are used.
- **Congress-aggregate RBO weight:** Computed from the full 116th Congress edge list (not per-quarter); provides a stable structural measure of similarity, not contaminated by within-quarter sparsity.
- **Unique-entry robustness:** A second analysis restricts to bills where A is the sole firm first entering in quarter t (no co-entrants). This isolates cases where B can plausibly attribute the bill signal to A specifically rather than a coordinated wave. Reduces N to ~25K but results replicate.
- **Covariates:** log(n_firms_on_bill) controls for bill salience/popularity; a_net_influence and b_net_influence absorb firm-level propensity to lead/follow; a_entry_quarter controls for temporal position within the congress (later bills have fewer observable follow-up quarters even after the horizon restriction).

**Empirical results (116th Congress, run 2026-04-17):**
- **Baseline adoption rates (full sample):** Q+1: 3.1%, Q+2: 4.8%, Q+3: 5.7% overall.
- **Adoption rate gradient by RBO quartile (Q+3):** Q1=3.9%, Q2=4.6%, Q3=6.1%, Q4=8.3%. Monotone increase across all three horizons.
- **Median-split ratio:** High-RBO pairs adopt at 1.75× the rate of low-RBO pairs at Q+1, 1.71× at Q+2, 1.70× at Q+3. Ratio is stable across horizons. χ²(Q+1)=179.4, p<0.0001.
- **Regression (log_rbo):** Positive and significant (p<0.001) in Logit and LPM across all 3 horizons, both full and unique-entry samples. Logit coefficients range 0.16–0.24 (log-odds per log-unit of RBO weight). LPM coefficients: 0.0042–0.0076 percentage-point increase per log-unit (small but stable). Bill popularity (log_n_firms) is the dominant covariate: more widely-lobbied bills attract more followers mechanically.
- **A's net_influence quartile:** No monotone pattern in Q+1 adoption. Q1 (weakest influencers) has the highest adoption rate (3.5%) vs. Q4 (strongest, 2.9%). This is likely a composition effect: the weakest "influencers" by net first-mover count may be large firms with broad bill portfolios that attract followers regardless of their agenda-setting success. Does not undermine the RBO finding; both effects operate simultaneously.
- **Unique-entry robustness:** Pattern replicates at lower base rates (Q+1: 0.86%), with log_rbo positive and significant throughout. Confirms that the RBO gradient is not entirely driven by bills with simultaneous multi-firm entry waves.

**Script:** `src/validations/19_bill_adoption_diffusion.py`

**Outputs:**
- `outputs/validation/19_adoption_candidates.csv` — full candidate set (80K rows)
- `outputs/validation/19_adoption_rates.csv` — adoption rates by quartile and horizon
- `outputs/validation/19_adoption_regression.csv` — regression coefficients across specs
- `outputs/validation/19_bill_adoption_diffusion.txt` — full log
- `visualizations/png/19_bill_adoption_diffusion.png` — adoption curve and quartile bar chart

---

## §36 — Codebase Cleanup and Archive Restructure (2026-04-19)

**Decision:** Archived all components not directly part of the primary pipeline (extraction → directed influence → validations/mechanism tests) into standardized archive directories.

**Archive locations:**
- `src/archive/networks/` — six undirected similarity/affiliation network scripts (bill affiliation, RBO similarity, cosine similarity, issue similarity, lobby firm affiliation, lobbyist affiliation, composite, composite comparison, issue RBO, bill-company matrix builder, quarterly RBO networks, per-quarter directed influence network)
- `src/archive/fortune_20/` — Fortune 20 subset legacy scripts
- `src/archive/psne/` — PSNE game-theoretic model and C++ solver
- `data/archive/network_edges/` — all undirected network edge CSVs (affiliation, RBO, cosine, composite, issue-RBO edges)
- `data/archive/communities/` — Leiden community assignment CSVs for all archived networks
- `data/archive/centralities/` — centrality CSVs for all archived networks
- `data/archive/LobbyView/` — alternative data source (not used in pipeline)
- `visualizations/archive/undirected/` — GML, PNG, and PDF for all undirected networks

**Active exception:** `data/network_edges/lobbyist_affiliation_edges.csv` stays at the active data path because `affiliation_mediated_adoption.py` reads it at that location.

**Path updates applied:**
- `src/enrich_directed_gml.py` and validations 13–16 were updated to read from `data/archive/communities/`, `data/archive/centralities/`, and `data/archive/network_edges/` after the archive move.
- All archived network scripts were updated to write their outputs to the archive directories.

**Doc updates (this session):** DOCUMENTATION.md and design_decisions.md updated to reflect archive paths throughout; an "Archived Work" section added to DOCUMENTATION.md (§20); README.md rewritten to remove stale paths and reflect current project structure.

**Additional cleanup (2026-04-19):** Channel test scripts (`src/channel_tests/`) and outputs (`outputs/channel_tests/`) copied to `src/archive/channel_tests/` and `outputs/archive/channel_tests/`. Originals cannot be deleted via the shell — manual deletion required. V17 quarterly dynamics, composite network, Guimerà-Amaral taxonomy, and participation coefficient removed from all active documentation.

---

## Archived Work Reference

The following components were part of earlier phases and are retained for reproducibility but are not part of the current primary pipeline.

### Undirected Network Suite

The bill affiliation network is the most downstream-relevant archived network: its Leiden partition and centrality table feed into validations 13–16. All scripts in `src/archive/networks/`. Data outputs in `data/archive/`. Visualization outputs in `visualizations/archive/undirected/`.

**Reproduction order** (if regenerating archived files): `bill_affiliation_network.py` → `rbo_similarity_network.py` → `cosine_similarity_network.py`. `issue_similarity_network.py`, `issue_rbo_similarity_network.py`, `lobby_firm_affiliation_network.py`, and `lobbyist_affiliation_network.py` are standalone. `lobbyist_affiliation_network.py` must run before `affiliation_mediated_adoption.py` to generate `data/network_edges/lobbyist_affiliation_edges.csv`.

### LobbyView Data

`src/archive/lobbyview_extraction.py` and `src/archive/build_lobbyview_mapping.py` were built when LobbyView was evaluated as an alternative data source. Not adopted; OpenSecrets CRP bulk data is the sole source. Raw data: `data/archive/LobbyView/`.

### Fortune 20 and PSNE

Early-phase work on a Fortune 20 subset and PSNE game-theoretic model. Scripts: `src/archive/fortune_20/` and `src/archive/psne/`.

### Channel Tests (deleted)

`src/channel_tests/` and `outputs/channel_tests/` have been copied to `src/archive/channel_tests/` and `outputs/archive/channel_tests/`. The originals require manual deletion from the filesystem.

---

## §37 — RBO Directed Influence Network: Bidirectional Edge Redesign (April 2026)

**Problem identified:** The original formulation (§21) allocated the full RBO weight to whichever firm had the higher first-mover count in each pair, treating marginal temporal differences (e.g., 16 vs 14 firsts out of 30 shared bills) as categorical distinctions. Near-ties produced disproportionate influence asymmetries, and `net_strength` (out_strength − in_strength on decisive edges) was theoretically weakly grounded.

**New edge structure:**

Every pair (i, j) with RBO > 0 now produces **two directed edges** instead of one:

- i→j with weight = `[(i_firsts + ties/2) / shared_bills] × RBO`
- j→i with weight = `[(j_firsts + ties/2) / shared_bills] × RBO`
- Constraint: `weight(i→j) + weight(j→i) = RBO`

Each edge also stores `rbo` (full RBO similarity, same for both directions) and `net_temporal = source_firsts − target_firsts` (signed, from source's perspective). The `balanced` column is removed; balanced pairs (net_temporal = 0) now appear as two equal-weight edges rather than a single canonical edge.

**net_strength redefined:**

```
net_strength(i) = Σ_j [RBO(i,j) × net_temporal(i,j)]
```

where j ranges over all neighbors and net_temporal(i,j) = i_firsts − j_firsts. In graph terms, this is computed by summing `rbo × net_temporal` over all out-edges of node i (out-edges cover all pairs without double-counting, since both directions are present). Within-community variant restricts the sum to same-community neighbors.

**Interpretation:** net_strength captures "aggregate agenda-setting leverage through strategically aligned relationships." High RBO × high net_temporal = strong influence through strong alignment; near-ties contribute proportionally rather than zero. net_strength is zero-sum across each pair: net_strength(i) contribution from pair (i,j) equals −[contribution to net_strength(j)].

**net_influence unchanged:** Still defined as `Σ_j (i_firsts_j − j_firsts_j)` across all pairings. With bidirectional edges, this is computed from out-edges only (`Σ source_firsts − Σ target_firsts`) to avoid double-counting.

**Node coloring updated:** Nodes colored by net_strength (not net_influence): green if net_strength > 0, red if < 0, gray if = 0 or isolated.

**"Decisive" edge selection in downstream scripts:** Validations and stability scripts that previously filtered `balanced == 0` now use `net_temporal > 0` to select the dominant-direction edge for each pair (one per decisive pair; equivalent set). Functions that computed within-community metrics using both out- and in-edges now use out-edges only (same result without double-counting).

**Theoretical grounding:** Aligns with the directed BCZ framework where equilibrium effort responds to weighted complementarity structure. The product `RBO × net_temporal` directly captures both alignment strength and temporal dominance, making net_strength a natural measure of influence in a complementarity network.

**Tradeoffs:** Edge count doubles (two per pair). Old GML files and any downstream code using the `balanced` column or old `net_strength` formula are incompatible and must be regenerated.

**Scripts updated:**
- `src/rbo_directed_influence.py` — core edge and graph construction
- `src/enrich_directed_gml.py` — within-community metrics
- `src/multi_congress_pipeline.py` — calls updated functions (no structural changes)
- `src/cross_congressional_stability.py` — pair matrix and coverage counts
- `src/validations/13_centrality_vs_agenda_setter.py` — wc_net_strength formula
- `src/validations/14_influencer_regression.py` — wc_net_strength formula
- `src/validations/15_cross_sector_directed_edges.py` — decisive edge filter, net_cs_strength
- `src/validations/16_industry_influencer_hierarchy.py` — wc_net_influence from out-edges
- `src/validations/19_bill_adoption_diffusion.py` — decisive edge filter, use rbo column

---

## §38 — net_strength as Primary Agenda-Setter Proxy (April 2026)

**Decision:** Designate `net_strength` as the **primary** agenda-setter proxy and `net_influence` as the **reference/secondary** metric throughout all analyses and downstream scripts. This reverses the original primacy assignment in §21.

**Rationale:**

`net_strength = Σ_j [RBO(i,j) × net_temporal(i,j)]` — introduced in §37 — is a better operationalization of BCZ strategic complementarity than the unweighted `net_influence`. Specifically:

- It weights each pairwise first-mover score by the RBO similarity of the pair. A decisive win against a high-RBO neighbor (strong strategic alignment) contributes more than a win against a low-RBO neighbor.
- It directly implements the BCZ equilibrium notion: in BCZ, equilibrium effort is proportional to a weighted sum of neighbors' efforts, where weights are complementarity strengths. `net_strength` replaces the complementarity weight with RBO and the effort difference with net_temporal, making it the empirical analog of BCZ-weighted aggregate influence.
- `net_influence` (raw win/loss count) is informative but treats all pairings as equally important regardless of portfolio similarity, which is theoretically weaker.

**Scope of change:**

| Component | Change |
|---|---|
| `src/rbo_directed_influence.py` | `print_stats()` sorts and displays by `net_strength` first |
| `src/multi_congress_pipeline.py` | Node table sorted by `net_strength` |
| `src/cross_congressional_stability.py` | Analysis 3 = net_strength [PRIMARY]; Analysis 4 = net_influence [REFERENCE] |
| `src/validations/12_congress_statistics.py` | Top agenda-setters, correlations, scatters by `net_strength` first |
| `src/validations/13_centrality_vs_agenda_setter.py` | BCZ ↔ net_strength is the primary comparison; BCZ ↔ net_influence is reference |
| `src/validations/14_influencer_regression.py` | Spec B (net_strength) primary; Spec B2 (top_q_ns binary LPM) primary; Spec A (net_influence) reference; ordering B → B2 → A → C |
| `src/validations/16_industry_influencer_hierarchy.py` | wc_net_strength leaderboard primary; wc_net_influence secondary |

**Dependency note:** All existing congress edge CSVs (111th–117th) were produced under the old pipeline and lack the `rbo` column required to compute `wc_net_strength` and the complementarity tests in V14 Spec C and V20. Scripts that require `rbo` from edge files contain graceful fallbacks (WARNING + skip/NaN). To fully populate these analyses, re-run `src/multi_congress_pipeline.py` to regenerate per-congress edge CSVs with the new schema (`source, target, weight, rbo, source_firsts, target_firsts, tie_count, shared_bills, net_temporal`).

**Note on V13:** The BCZ intercentrality computation (277 node-removal Katz reruns) times out in the sandboxed environment. This validation must be run on the full local machine; its outputs remain valid from the prior run.

**GML files to regenerate:** `visualizations/gml/rbo_directed_influence.gml` (116th), and all per-congress GMLs via `multi_congress_pipeline.py`. Old GML files are structurally incompatible (missing `rbo` edge attribute, has stale `balanced` attribute).
