# Design Decisions — Fortune 500 Lobbying Network Analysis

**Project:** Layer 1 network construction for Fortune 500 co-lobbying analysis
**Authors:** Prathit Kurup, Victoria Figueroa
**Data:** OpenSecrets CRP, 116th Congress (2019–2020), Fortune 500 firms

This document records all significant design decisions made during the project,
the reasoning behind each choice, and alternatives that were considered and
rejected. It is intended to support the methods section of the final paper and
to allow any researcher to reconstruct the analytical choices.

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

**Decision:** Use three complementary bill-level similarity signals — affiliation,
cosine similarity, and Rank-Biased Overlap — and combine them multiplicatively in
the composite network.  See §10 (cosine), §11 (RBO), and §12 (composite) for
detailed rationales.

**Summary of the three signals:**
- **Affiliation** (`affil_norm = shared_bills / N_total`): raw co-lobbying breadth,
  normalized by the total filtered bill universe.  Interpretable as "what fraction
  of all available bills do these two firms jointly lobby?"
- **Cosine similarity**: geometric alignment of portfolio-share (frac) vectors.
  Captures directional agreement across the full bill portfolio.
- **RBO**: priority-ranking agreement.  Captures whether the firms' high-spend
  legislative priorities are the same.

**Why not Bray-Curtis breadth×depth?**
Bray-Curtis is a standard ecology dissimilarity metric (Bray & Curtis, 1957) that
measures per-bill frac differences.  It was used in an earlier version of this
project but replaced because: (a) cosine similarity is a cleaner geometric measure
of directional alignment; (b) affil_norm provides the "breadth" signal more
transparently without the exponential calibration parameter λ; (c) the composite
formula `affil_norm × cosine × rbo` separates the three dimensions cleanly and is
easier to interpret.

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

## §7 — Three-Tier Community Centrality

**Decision:** Use a three-tier centrality framework adapted from Guimerà &
Amaral (2005) to identify:
1. **Industry leaders** — firms most central within their own lobbying coalition.
2. **Cross-industry connectors** — firms whose lobbying spans multiple coalitions.
3. **Global hubs** — firms with overall network prominence.

**Tier 1: Within-community eigenvector centrality**
Eigenvector centrality is computed on each community subgraph independently.
This captures recursive "important neighbors" structure: a firm is central not
just because it has many strong connections, but because those connections are
themselves to highly connected firms. Preferred over z-score (normalized weighted
degree) as the primary industry-leader metric because it better reflects coalition
leadership dynamics.
Falls back to weighted degree for communities with < 3 nodes or when the power
iteration doesn't converge.

**Tier 2: Guimerà-Amaral participation coefficient (P)**
`P_i = 1 − Σ_c (κ_ic / κ_i)²`
where κ_ic = total edge weight from i to community c, κ_i = total weighted degree.
P = 0: firm connects exclusively within its own community.
P → 1: firm connects evenly across all communities.
This is the key metric for identifying cross-industry political entrepreneurs —
firms that bridge across lobbying coalitions and could facilitate cross-sector
coordination. Grounded in Guimerà & Amaral (2005, Nature, 433, 895–900).

**Tier 2b: Guimerà-Amaral z-score (within-community degree z-score)**
`z_i = (κ_is − mean(κ_s)) / std(κ_s)`
Retained alongside eigenvector for methodological comparison (Guimerà & Amaral
2005 original formulation).

**Tier 3: Global PageRank**
PageRank on the full graph captures overall network prominence. Comparing a
firm's global PageRank to its within-community eigenvector reveals whether
prominence stems from within-industry dominance or cross-industry bridging.

**Role classification:** Guimerà-Amaral 7-role taxonomy combines z and P:
- provincial_hub: z ≥ 2.5, P < 0.30 (dominant in community, stays in lane)
- connector_hub: z ≥ 2.5, 0.30 ≤ P < 0.75 (dominant AND cross-industry)
- kinless_hub: z ≥ 2.5, P ≥ 0.75 (highly cross-industry hub)
- ultra_peripheral: z < 2.5, P < 0.05 (almost entirely within community)
- peripheral: z < 2.5, 0.05 ≤ P < 0.625
- non_hub_connector: z < 2.5, 0.625 ≤ P < 0.80 (bridges without dominance)
- kinless: z < 2.5, P ≥ 0.80

**Why not CBCM?** The Community-Based Centrality Measure could also identify
cross-industry connectors but lacks the Guimerà-Amaral role taxonomy, which
provides a principled classification system grounded in theoretical network
science and widely used in the literature on functional cartography.

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

## §9 — Community Comparison Methodology

**Decision:** Compare Leiden partitions from the four networks (composite,
affiliation, cosine, RBO) using NMI, ARI, and Hungarian-aligned confusion matrices.
See §14 for the four-way comparison methodology.

**Rationale:**
- **NMI (Normalized Mutual Information):** Measures information shared between
  two labelings; NMI = 1 if identical, NMI = 0 if independent. Insensitive to
  community label permutation.
- **ARI (Adjusted Rand Index):** Measures pairwise agreement; adjusted for
  chance so ARI ≈ 0 for random partitions, ARI = 1 for identical partitions.
  ARI can be negative (worse than chance).
- **Hungarian alignment:** Community IDs from two partitions are arbitrary
  integers; the Hungarian algorithm finds the optimal bijection between
  community IDs (maximizing overlap) before building a confusion matrix.
  This makes the matrix human-readable without affecting NMI/ARI.

**Finding (pre-filtering):** NMI ≈ 0.15, ARI ≈ 0.15 — low agreement, consistent
with the mega-bill distortion collapsing both partitions toward near-random
(Q ≈ 0.02). Post-filtering results are expected to show higher agreement as
both networks recover meaningful community structure (Q ≈ 0.18).

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

**Source:** `src/issue_similarity_network.py`

---

## §12 — Composite Similarity Network: Three-Signal Design

**Decision:** Build a composite company-to-company similarity network by
multiplying three signals:

```
composite(i,j) = affil_norm(i,j) × cosine(i,j) × rbo(i,j)

affil_norm(i,j) = shared_bills(i,j) / N_total_bills
```

where:
- **affil_norm** ∈ [0, 1]: fraction of the total filtered bill universe that
  both firms co-lobby.  `shared_bills(i,j)` = count of bills (post mega-bill
  filter) that both firm i and firm j lobby for.  `N_total_bills` = 2285 in
  the 116th Congress OpenSecrets data.
  Example: two firms sharing 10 bills → affil_norm = 10/2285 ≈ 0.0044.
- **cosine** ∈ [0, 1]: cosine similarity of portfolio-share (frac) vectors —
  geometric alignment of lobbying budgets.
- **rbo** ∈ [0, 1]: Rank-Biased Overlap on bill priority rankings — agreement
  on which bills matter most.

**Why affil_norm = shared_bills / N_total:**
It is the simplest, most transparent normalization: the raw shared-bill count
expressed as a fraction of all available bills.  No parameters to calibrate.
Direct interpretation: "what fraction of all lobbied bills do these two firms
address together?"  This is pure breadth, measured on the absolute scale of
the full bill universe rather than relative to each firm's portfolio.
The cosine and RBO components provide depth and priority alignment respectively.

**Implementation:**
`affil_norm` is computed directly in `composite_similarity_network.py` via a
binary firm×bill matrix dot product, using the same mega-bill-filtered bill
universe as the cosine and RBO scripts.  Cosine and RBO weights come from
their pre-computed CSVs (`cosine_edges.csv`, `rbo_edges.csv`).  An inner join
restricts composite edges to pairs present in both CSVs with shared_bills > 0.

**Observed weight scale (116th Congress):**
The most-overlapping pair shares 111 of 2285 bills (affil_norm ≈ 0.049), so
composite weights peak around 0.004.  This is expected: the small scale is
purely a consequence of the N_total denominator.  Relative ordering is
preserved, so community detection and centrality are unaffected.

**Rationale for multiplicative combination:**
A multiplicative formula enforces AND logic across all three dimensions:
a pair must simultaneously have high bill overlap AND portfolio alignment AND
priority-ranking agreement to receive a high composite weight.  A pair weak
on any single dimension is automatically down-weighted without manual thresholds.

**Sparsification:**
The three-way inner join produces a sparser graph than any individual component
network (1,200 composite edges vs 1,516 cosine and 2,771 RBO edges in 116th
Congress data with thresholds at 0).

**Alternative considered — additive combination:**
`w = α·affil + β·cosine + γ·rbo` requires hand-tuning three weights and does
not enforce AND logic.  Rejected.

**Alternative considered — geometric mean:**
`(affil × cosine × rbo)^(1/3)` normalises scale but attenuates the multi-signal
penalty for pairs weak on one dimension.  Rejected.

**Source:** `src/composite_similarity_network.py`
**Comparison script:** `src/composite_community_comparison.py`
**Validation:** `src/validations/07_composite_network_validation.py`

---

## §13 — Katz-Bonacich Centrality

**Decision:** Add Katz-Bonacich centrality as a fourth centrality measure
alongside within-community eigenvector, PageRank, and the Guimerà-Amaral
metrics. It is computed for all networks (including the composite).

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
**Validation:** `src/validations/07_composite_network_validation.py` Section C

---

## §14 — Four-Way Community Comparison Methodology

**Decision:** Compare Leiden partitions from composite, affiliation, cosine, and
RBO networks using all C(4,2) = 6 pairwise NMI and ARI scores, plus a firm-level
consensus stability classification.

**Rationale:**
The four networks measure progressively different aspects of co-lobbying similarity:
- Affiliation uses raw bill co-lobbying breadth (shared bill count, normalized).
- Cosine measures geometric alignment of portfolio-share vectors across all bills.
- RBO measures top-ranked priority agreement.
- Composite enforces simultaneous alignment on all three via affil_norm × cosine × rbo.

If the four partitions are highly similar (NMI > 0.70 for most pairs), it suggests
that a single underlying dimension — industry sector — dominates community structure
regardless of which similarity metric is used. This would support using any metric
for community identification and focus attention on the nodes that diverge.

If partitions differ significantly (NMI < 0.40), it suggests that different metrics
reveal genuinely different coalition structures. In this case the composite partition
is the most conservative — it identifies firms whose similarity is robust to metric
choice — and community-divergent firms are theoretically interesting as potential
cross-coalition operators.

**Consensus stability taxonomy:**
- fully_stable: firm assigned to the same (Hungarian-aligned) community in all 4 networks.
- partially_stable: matches composite in ≥ 1 other network.
- composite_divergent: composite places the firm in a different community than affiliation, cosine, AND RBO.
- absent_from_composite: firm not in the composite network (isolated under triple-filter).

**Source:** `src/composite_community_comparison.py`
**Outputs:** `data/community_comparison_composite.csv`, `data/nmi_ari_matrix.csv`

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
cosine or composite network have genuine zero co-lobbying overlap post-filtering.
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

## References

Bonacich, P. (1987). Power and centrality: A family of measures. American Journal
  of Sociology, 92(5), 1170–1182.

Guimerà, R., & Amaral, L.A.N. (2005). Functional cartography of complex metabolic
  networks. Nature, 433, 895–900.

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
