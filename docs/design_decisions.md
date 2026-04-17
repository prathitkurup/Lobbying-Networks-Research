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

## §19 — Quarterly RBO Networks: Window Design and Temporal Evolution Methodology

**Script:** `src/rbo_quarterly_networks.py`

**Decision:** Build 8 independent quarterly RBO similarity networks — one per
quarter of the 116th Congress (Q1 2019 through Q4 2020) — and analyze how the
network structure evolves over time using three complementary methods.

### Quarter Assignment

The `report_type` field in `opensecrets_lda_reports.csv` distinguishes original
filings (`q1`, `q2`, `q3`, `q4`), amendments (`q1a`, `q2a`, etc.), termination
reports (`q1t`, `q2t`, etc.), and termination amendments (`q1ta`, `q2ta`). All
variants belonging to the same reporting period are assigned to the same quarter
(e.g., `q1`, `q1a`, `q1t`, `q1ta` all → quarter 1 of that year). This preserves
the full lobbying activity attributed to that quarter rather than discarding
amendment-corrected filings.

A `quarter` column (integer 1–8) was added to `opensecrets_lda_reports.csv`
using year × quarter-within-year: 2019 Q1=1, …, 2019 Q4=4, 2020 Q1=5, …,
2020 Q4=8. The mapping is: `base_q = int(report_type[1])`, `year_offset =
0 (2019) or 4 (2020)`, `quarter = base_q + year_offset`.

**Alternative rejected:** Using only original filings (`q1/q2/q3/q4`) and
discarding amendments would undercount lobbying activity, particularly for firms
that file corrected spending figures. The `ind='y'` filter (§16) already handles
deduplication between originals and amendments at the extraction stage.

### Independent Windows vs. Rolling/Cumulative Aggregation

Each quarter is treated as an independent observation window — only filings with
a `report_type` prefix matching that quarter are included. This choice means:

- **Edges reflect current activity:** Two firms that lobbied the same bills in
  Q3 but diverged in Q4 will have a high RBO edge in Q3 and potentially no edge
  in Q4. This is the correct representation for studying dynamic strategic
  alignment.

- **Comparability across quarters:** Each network is built with identical RBO
  parameters (`p=0.85`, `top_bills=30`, `min_weight=0.0`) and Leiden resolution
  (`γ=0.75`), matching the full-congress `rbo_similarity_network.py`.

**Cumulative aggregation was rejected** because it conflates early and late
congressional activity, masks strategic pivots, and is inconsistent with the
existing full-congress baseline.

### MAX_BILL_DF Behavior in Quarterly Networks

`MAX_BILL_DF = 50` (§4) is applied per-quarter. Per-quarter bill prevalence is
naturally lower than congress-wide prevalence — omnibus bills that attract 100+
firms across two years may attract only 40 firms in a single quarter. As a result,
fewer bills are filtered per quarter. This is the correct behavior: a bill with
>50 firms lobbying it *within a single quarter* still carries no strategic
alignment signal and should be removed, but bills that would only exceed the
threshold if aggregated should not be pre-emptively excluded.

### Temporal Analysis A — Community Stability (NMI + ARI)

Leiden partitions from consecutive quarters are compared using:
- **NMI** (normalized mutual information, arithmetic average method, sklearn):
  label-permutation invariant, measures how much knowing one partition reduces
  uncertainty about the other. Range [0, 1].
- **ARI** (adjusted Rand index, sklearn): measures pairwise co-clustering
  agreement adjusted for chance. Range [−1, 1], with 1 = identical partitions.

Both metrics are computed over the intersection of firms present in both quarters.
This correctly handles firms that enter or exit the lobbying network across quarters.
The use of set intersection rather than union is deliberate: firms not in both
quarters cannot contribute to partition comparison and would need arbitrary community
assignments.

**Interpretation calibration:** NMI > 0.70 = high stability, 0.40–0.70 = moderate,
< 0.40 = low. These thresholds follow the convention used in the temporal network
community detection literature (Mucha et al., 2010).

**Empirical result:** Mean NMI = 0.503, Mean ARI = 0.318 across 7 consecutive-
quarter transitions — moderate stability. The Q4 2019 → Q1 2020 transition shows
the lowest NMI (0.447) and ARI (0.260), consistent with a structural shock from the
COVID-19 pandemic disrupting the lobbying agenda at the start of 2020.

### Temporal Analysis B — Network Metric Trajectories

Per-quarter: nodes, edges, density, mean RBO weight, weighted clustering coefficient
(NetworkX `average_clustering` with `weight="weight"`), Leiden modularity Q, and
community count. These track macro-level structural evolution independent of
community labeling.

**Empirical highlights:**
- Q1 2019 is the smallest but most cohesive quarter (172 nodes, density 0.0772,
  mean weight 0.167, Q = 0.467) — firms that lobby in Q1 of a new Congress have
  strongly concentrated bill portfolios.
- 2020 Q1 shows a sharp drop in density (0.070) and mean weight (0.089) with
  the highest modularity rebound (Q = 0.382), signaling fragmentation of the
  co-lobbying consensus at the pandemic onset.
- 2020 Q3 reaches the lowest mean RBO weight (0.061) and modularity (0.227),
  reflecting the broadest and most diffuse bill agendas during peak COVID
  legislative activity.

### Temporal Analysis C — Centrality Rank Stability (Spearman ρ)

PageRank (weighted, NetworkX default `alpha=0.85`) is computed per quarter. The
Spearman rank correlation between consecutive quarters is calculated over the
intersection of firms present in both. Spearman ρ is appropriate because the
research question concerns whether the *relative ordering* of firms by influence
is stable, not the absolute PageRank values (which are not comparable across
networks of different sizes).

**Why PageRank for influence:** PageRank captures recursive structural influence —
a firm is influential if it is connected to other influential firms — which is the
appropriate operationalization of structural power in lobbying similarity networks.

**Empirical result:** Mean Spearman ρ = 0.616 (all p < 0.001) — moderate influence
stability. The Q4 2019 → Q1 2020 transition again shows the lowest ρ (0.519),
consistent with COVID disruption. This implies that while the same firms tend to
dominate across quarters, the influence hierarchy is not fixed — a finding
consistent with dynamic coalition formation in the lobbying literature (Hojnacki
et al., 2012; Carpenter et al., 1998).

### Output Files

| File | Contents |
|---|---|
| `data/rbo_edges_q{1..8}.csv` | Edge list per quarter (source, target, weight) |
| `data/communities_rbo_q{1..8}.csv` | Leiden community assignments per quarter |
| `data/rbo_quarterly_stats.csv` | 8-row network metric table |
| `data/rbo_quarterly_nmi_ari.csv` | 7-row temporal analysis A results |
| `data/rbo_quarterly_spearman.csv` | 7-row temporal analysis C results |

### References

Carpenter, D.P., Esterling, K.M., & Lazer, D.M.J. (1998). The strength of weak
  ties in lobbying networks: Evidence from health-care politics in the United States.
  Journal of Theoretical Politics, 10(4), 417–444.

Hojnacki, M., Kimball, D.C., Baumgartner, F.R., Berry, J.M., & Leech, B.L. (2012).
  Studying Organizational Advocacy and Influence: Reexamining Interest Group
  Research. Annual Review of Political Science, 15, 379–399.

Mucha, P.J., Richardson, T., Macon, K., Porter, M.A., & Onnela, J.P. (2010).
  Community structure in time-dependent, multiscale, and multiplex networks.
  Science, 328(5980), 876–878.

---

## §20 — Directed Influence Network: Temporal Bill-Adoption Precedence

**Script:** `src/directed_influence_network.py`

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

`net_influence` (integer) is the primary node metric and is computed as the
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

**Script:** `src/issue_rbo_similarity_network.py`

**Outputs:** `data/network_edges/issue_rbo_edges.csv`, `data/communities/communities_issue_rbo.csv`, `data/centralities/centrality_issue_rbo.csv`, `visualizations/gml/issue_rbo_similarity_network.gml`, `visualizations/png/issue_rbo_similarity_network.png`

---

## §26 — outputs/ Folder Restructure and Validation Path Redirect (2026-04-15)

**Decision:** Moved all validation script outputs from `src/validations/outputs/` to a new `outputs/` directory at the repo root, with subdirectories `outputs/validation/` and `outputs/channel_tests/`.

**Rationale:** Channel test outputs, revolving door assessments, and future mechanism-testing results do not belong in the validation subdirectory. Centralizing all analytical outputs at the root level improves navigability and separates research outputs from source code.

**Changes:** All `OUTPUT_PATH` constants in `src/validations/01-08` were updated to `Path(__file__).resolve().parent.parent.parent / "outputs" / "validation" / "filename"`. Scripts 06, 07, 10, 11 (previously stdout-only) had a `_Tee` class added that writes to both stdout and the output file simultaneously.

---

## §27 — Channel Tests: Monitoring Capacity (Ch1) and Issue Overlap (Ch3) (2026-04-15)

**Decision:** Build two standalone channel test scripts to empirically evaluate alternative explanations for the RBO directed influence signal.

**Channel 1 — Monitoring Capacity:** Tests whether "influencers" in the RBO network are simply firms with larger lobbying operations that track legislation earlier. Capacity proxies: total spend, reports filed, bills lobbied, unique lobbyists, issue codes covered. Finds partial support: bills lobbied (ρ=0.204, p<0.001) and total spend (ρ=0.128, p<0.05) correlate with net_influence; for directed pairs, sources have significantly more bills lobbied (p<0.001), more spend (p<0.001), and more lobbyists (p<0.001) than targets. Interpretation: capacity contributes to first-mover advantage but does not exhaust the signal (Bertrand, Bombardini & Trebbi 2014).

**Channel 3 — Issue-Space Correlated Response:** Tests whether directed pairs share the same regulatory domain and independently adopt the same bills due to domain relevance rather than influence transmission. Directed pairs have substantially higher issue cosine similarity (mean 0.432) than balanced pairs (0.371) or random firm pairs (0.198), all differences significant (p<0.001). Issue similarity positively predicts RBO edge weight (ρ=0.137, p<0.001). Mediation rate is invariant across issue-overlap quartiles (0.2% throughout), supporting a correlated-response interpretation: even high-overlap pairs are not mediated through shared channels, yet still form directed pairs — consistent with independent domain-monitoring. This is the strongest evidence for the correlated-response channel (Hojnacki 1997; Schlozman & Tierney 1986).

**Combined implication:** Both channels partially explain the RBO directed influence signal. The most parsimonious account: (a) firms in the same regulatory domain track the same legislation independently (issue overlap → correlated adoption); (b) within that correlated adoption, resource-richer firms move first more often (capacity gap → temporal precedence); (c) genuine information transmission through shared lobbyists/firms is rare and structurally limited by the dominance of in-house lobbying among Fortune 500 firms.

**Scripts:** `src/channel_tests/test_channel1_monitoring_capacity.py`, `src/channel_tests/test_channel3_issue_overlap.py`

**Outputs:** `outputs/channel_tests/channel1_monitoring_capacity.{txt,png}`, `outputs/channel_tests/channel3_issue_overlap.{txt,png}`, `outputs/channel_tests/revolving_door_assessment.txt`

---

## §28 — Cross-Congressional Stability Analysis (2026-04-16, updated 2026-04-16)

**Decision:** Test whether RBO directed influence edges are stable in direction and magnitude across seven consecutive congresses (111th–117th, 2009–2022) using the subset of Fortune 500 firms that lobbied in all seven sessions.

**Rationale:** Temporal stability is a necessary (though not sufficient) condition for causal interpretation of the directed influence network. If the same firm consistently leads across seven independent two-year legislative sessions, that regularity is unlikely to be produced by noise alone. Four stability tests are run: (1) direction consistency — does the same firm lead in each session a pair appears?; (2) magnitude stability — are net_temporal values correlated across congress pairs (Spearman ρ) and concordant across all seven (Kendall's W)?; (3) firm net_influence rank stability — do net_influence ranks agree across sessions (adjacent-congress Spearman + Kendall's W)?; (4) firm net_strength rank stability — mirrors Analysis 3 for the RBO-weighted net_strength score.

**Design choices:**
- **Stable set criterion:** Firms must appear in the directed influence network for all seven congresses. Firms active in only some congresses are excluded from stability analysis but their congresses still run.
- **Congress range:** The 111th Congress (2009–2010) is the earliest included; pre-111th congresses use semi-annual HLOGA reporting codes incompatible with the quarterly `assign_quarters` function and are excluded.
- **Canonical pair representation:** Each (firm_a, firm_b) pair is represented canonically with firm_a < firm_b alphabetically. Direction score = 1 if firm_a leads, 0 if firm_b leads; balanced edges are excluded from direction analysis.
- **Direction consistency threshold:** A pair is considered "direction-consistent" if max(n_a_leads, n_b_leads) / n_directed_sessions ≥ 0.80.
- **Binomial test:** For each pair with ≥2 directed sessions, H0 is that each session is an independent 50/50 coin flip for direction. Rejection at p < 0.05 provides per-pair evidence of genuine directional consistency.
- **Kendall's W:** Computed with chi-squared approximation (Siegel & Castellan 1988). Pairs with fewer than 3 non-NaN sessions are excluded.
- **Quarter assignment:** Congress N covers years start=2009+2*(N−111) through start+1, with Q1-4 in year1 mapping to quarters 1-4 and Q1-4 in year2 mapping to quarters 5-8.

**Pipeline:** `src/multi_congress_pipeline.py` runs extraction + RBO directed influence for each of the seven congresses in sequence, writing per-congress outputs to `data/congress/{num}/`. GML and PNG visualizations are also written to `visualizations/gml/` and `visualizations/png/` for each congress.

**Name mapping dependency:** Coverage depends on `manual_opensecrets_name_mapping.json` being expanded to include CRP name variants for each congress era. For the 111th (2009–10) and 117th (2021–22) additions, 34 new name variants were added to cover corporate rebrands (e.g. WellPoint→Anthem, Limited Brands→L Brands, TIAA-CREF→TIAA, Facebook→Meta, Motorola Inc→Motorola Solutions) and earlier-congress filings.

**Scripts:** `src/multi_congress_pipeline.py`, `src/cross_congressional_stability.py`

**Analysis 4 — net_strength rank stability:** Added as a direct mirror of Analysis 3 using net_strength (RBO-weighted directed score) rather than net_influence (raw win/loss count). The figure uses a 2×2 grid; Panels 3 and 4 show adjacent-congress Spearman bar charts for net_influence and net_strength respectively.

**Empirical results (111th–117th, run 2026-04-16):**
- **Stable set:** 135 firms present in all 7 congresses; 6,783 canonical pairs; 277 pairs present in all 7 sessions. Per-congress coverage: 246–260 firms, 3,558–7,418 total edges.
- **Direction consistency (Analysis 1):** 2,830 pairs with ≥2 directed sessions. Mean consistency = 0.770, median = 0.750. 45.1% of pairs ≥0.80 consistent; 41.2% perfectly consistent (score=1.00). 73.9% show a majority direction vs. 50% expected under H0. Only 2.4% reach individual binomial significance (p<0.05); most significant pair: BALL/IBM (5 directed sessions, p=0.031).
- **Magnitude stability (Analysis 2):** All adjacent-congress Spearman ρ on net_temporal are statistically significant except some involving the 113th Congress. Range: 0.037–0.218; highest: 116–117 (ρ=0.218). Kendall's W=0.125, p=1.00 is a degenerate result (large n=3,291 pairs with NaN imputation); individual Spearman values are the reliable evidence.
- **Firm net_influence rank stability (Analysis 3):** 5 of 6 adjacent-congress Spearman ρ are significant; only 113–114 is non-significant (ρ=0.134, p=0.12). Kendall's W=0.343, p<0.0001 — strong concordance across all 7 sessions. Top persistent influencers: Lockheed Martin (mean +72.1), IBM (+59.4), Xcel Energy (+53.6), CMS Energy (+46.4), Duke Energy (+45.4), AT&T (+43.1), DTE Energy (+42.7). Top persistent followers: Consolidated Edison (mean −55.1), Ameren (−48.3), Centerpoint Energy (−43.0).
- **Firm net_strength rank stability (Analysis 4):** 3 of 6 adjacent-congress Spearman ρ are significant (114–115, 115–116, 116–117); early transitions (111–112 through 113–114) are not significant. Kendall's W=0.273, p<0.0001 — significant global concordance at a lower level than net_influence. Top persistent high-strength firms: Xcel Energy (mean 2.214), Lockheed Martin (2.054), Duke Energy (1.803). Top persistent low-strength firms: Ally Financial (mean −1.108), Centerpoint Energy (−0.950), Cisco Systems (−0.929).
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
