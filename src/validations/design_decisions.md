# Design Decisions — Fortune 500 Lobbying Network Analysis

**Project:** Layer 1 network construction for Fortune 500 co-lobbying analysis
**Authors:** Prathit Kurup, Victoria Figueroa
**Data:** LobbyView, 116th Congress (2019–2020), Fortune 500 firms

This document records all significant design decisions made during the project,
the reasoning behind each choice, and alternatives that were considered and
rejected. It is intended to support the methods section of the final paper and
to allow any researcher to reconstruct the analytical choices.

---

## §1 — Data Structure: One Row Per Bill Per Report

**Decision:** Accept the raw data format from `extraction.py` (one row per
bill per report) and aggregate **before** network construction, not inside
`extraction.py`.

**Rationale:** `extraction.py`'s `join_reports()` function divides each filing's
reported spend equally across all bills in that report (`amount / num_bills`),
producing one row per (bill, report) pair. This design is correct for spend
accounting — it correctly tracks how spend was distributed across filing periods.
Aggregating inside `extraction.py` would lose the per-report audit trail. The
correct approach is to aggregate at network construction time.

**Pre-processing fix:**
- Affiliation network: `df.drop_duplicates(subset=["client_name","bill_id"])`
  — reduces to presence/absence; amount is irrelevant for affiliation.
- BC similarity: `df.groupby(["client_name","bill_id"])["amount"].sum()`
  — collapses to true total allocated spend per (firm, bill).

**Validation:** Run `validations/01_extraction_audit.py` to quantify
duplication extent. Typically ~40% of (firm, bill) pairs have multiple rows,
with the worst offenders at 10–20 rows.

---

## §2 — Cartesian Product Inflation Bug

**Decision:** Fix the cartesian product inflation in the original edge
construction by deduplicating before the clique-building loop.

**Bug description:** Without deduplication, `df.groupby("bill_id")["client_name"]
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
are **kept in the analysis** for BC similarity computations.
- They remain in each firm's total budget denominator (preserving frac = share
  of total lobbying portfolio).
- They do NOT produce any co-lobbying edges (no pairs to form).
- Excluding them would artificially inflate fracs on multi-firm bills and distort
  the portfolio composition metric.

**Validation:** Run `validations/03_sparsity_analysis.py` for the full null
model comparison.

---

## §4 — Mega-Bill Prevalence Filtering (MAX_BILL_DF = 50)

**Decision:** Exclude bills lobbied by more than 50 unique firms from edge
construction (and from the BC pairing loop). This is implemented as a
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

**Two-stage filtering for BC similarity:**
- Stage 1: Compute total_budget and fracs on **all bills** (including mega-bills)
  to preserve the economic meaning: `frac_ib = spend on bill b / total lobbying budget`.
- Stage 2: Build BC pairs from **filtered bills only** (excluding mega-bills).
  This removes the spurious BC ≈ 1.0 score that arises when two firms each
  allocate a tiny, equal frac to an omnibus bill they had no strategic choice
  about lobbying (e.g., both allocate 0.002 → BC = 1 − 0/0.004 = 1.0).

**Validation:** Run `validations/04_mega_bill_diagnosis.py` for the full
prevalence distribution and modularity comparison.

---

## §5 — Choice of Similarity Metric: Breadth × Depth (Bray-Curtis)

**Decision:** Use a composite `weight(i,j) = breadth(i,j) × depth(i,j)` where:
- `depth(i,j) = mean BC similarity over shared bills` (per-bill portfolio alignment)
- `breadth(i,j) = 1 − exp(−λ × shared_bill_count)` (saturating reward for overlap)
- `λ = log(2) / median(shared_bills)` (auto-calibrated so breadth = 0.5 at median)

**Rationale — Bray-Curtis:** The Bray-Curtis dissimilarity is standard in ecology
for comparing species compositions across samples (Bray & Curtis, 1957). Applied
here, it measures how similarly two firms distribute their lobbying budgets across
shared bills. Unlike cosine similarity, BC is insensitive to total budget magnitude
(only portfolio proportions matter), which is appropriate given the wide variation
in Fortune 500 lobbying budgets.

**Rationale — Breadth term:** Two firms that each lobby exactly one shared bill
with identical fracs get BC = 1.0 on that bill. A breadth term that rewards
coordination across many bills distinguishes genuine strategic alignment (consistent
co-lobbying across many issues) from coincidental alignment on a single bill.
The exponential saturation means breadth approaches 1 asymptotically — additional
shared bills beyond the saturation point contribute diminishing weight.

**λ calibration:** Setting λ = log(2) / median(shared_bills) makes breadth = 0.5
at the median shared-bill count, so the median pair gets neither the maximum breadth
reward nor is penalized relative to it. This is a natural, data-adaptive calibration.
Robustness to λ is verified via Spearman rank correlation at λ × 0.5, × 2, × 4.

**Alternative considered — IDF weighting:** Down-weight bills lobbied by many firms
(analogous to TF-IDF). Rejected in favor of a different approach planned for a
later layer; see research notes.

**Alternative considered — Simple shared-bill count:** The affiliation network
uses this as a baseline. It has the advantage of interpretability but lacks
the portfolio-alignment signal that BC captures.

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

**Decision:** Exclude firms with total lobbying budget = $0 from BC similarity
computation (with a printed warning listing the excluded firms). These firms
remain in the affiliation network.

**Rationale:** A firm with $0 total spend produces `frac = 0/0 = NaN` for all
bills, which would silently corrupt BC similarity scores (NaN propagates through
arithmetic). The LDA filing is valid even with $0 spend — firms are legally
required to file even if they spent less than the $5,000 threshold. These firms
listed bills but reported zero lobbying expenditure.

**Known affected firms (116th Congress):**
- Air Products & Chemicals
- Crown Holdings
- Dick's Sporting Goods
- Treehouse Foods

**Node count discrepancy (291 vs. 289):**
The affiliation and BC networks have different node counts because:
1. Crown Holdings: zero total budget → excluded by BC zero-budget guard,
   but present in affiliation (presence-based, no spend required).
2. Republic Services: non-zero budget ($70k), but $0 allocated on its only
   shared bill (hr2923-116) via equal-split. BC = 1 − |0 − f_j| / (0 + f_j) = 0
   for all pairs → filtered by `result[result["weight"] > 0]`.
3. Five additional firms: all bills are singletons (no shared bills with any
   other firm) → no edges in either network, hence absent from both graphs.

---

## §9 — Community Comparison Methodology

**Decision:** Compare Leiden partitions from the affiliation and BC networks
using NMI, ARI, and Hungarian-aligned confusion matrices.

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

## §11 — Issue Similarity Weight Range (Not Bounded [0,1])

**Decision:** The issue similarity network uses weight = sum(BC_issue) /
sqrt(shared_issue_count) with NORMALIZE=True. These weights are NOT bounded
between 0 and 1.

**Mathematical foundation:**
- Each Bray-Curtis score BC_k(i,j) per shared issue k is bounded in [0, 1].
- Summing N such scores gives a maximum of N (all perfect BC = 1.0).
- Dividing by sqrt(N) gives a maximum of sqrt(N).
- With 75 issue codes, the theoretical maximum is sqrt(75) ≈ 8.66.
- Without normalization (NORMALIZE=False), the maximum is 75.

**Empirical distribution:** Across the Fortune 500 LDA issues data:
- Median weight ≈ 0.81 (normalized)
- Maximum observed weight ≈ 3.28 (pairs sharing ~9–10 issues with high BC).
- 33.6% of edges exceed weight 1.0; only 1.2% exceed weight 2.0.

**Design rationale — the sqrt normalization as compromise:**

Three normalization strategies and their tradeoffs:

1. **Raw sum** (weight = N × mean_BC, unbounded up to 75):
   - Pros: Captures full breadth signal; pairs with many shared issues
     get much heavier weights than pairs with few issues.
   - Cons: Weight dominated by breadth, not depth; quality of alignment
     becomes secondary to multiplicity.

2. **Plain mean** (weight = mean_BC, bounded [0, 1]):
   - Pros: Weights directly interpretable as similarity scores in [0, 1].
   - Cons: Removes breadth signal entirely; a pair coordinating on 1 issue
     at BC = 0.9 gets the same weight as a pair coordinating on 50 issues
     at BC = 0.9. Ignores strategic multi-issue coordination.

3. **sqrt normalization** (weight = sum_BC / sqrt(N), bounded [0, sqrt(75)] ≈ [0, 8.66]):
   - Pros: Balances breadth (sub-linear reward) with depth (per-issue alignment).
     Pairs with many shared issues get higher weights, but growth is sublinear
     (diminishing returns), so weight quality (BC alignment) remains competitive
     with quantity (issue count).
   - Cons: Weights exceed [0, 1]; cannot be interpreted as proportions or
     probabilities.

**CHOSEN: Option 3.** This is most appropriate for community detection: Leiden
and related algorithms are designed for arbitrary positive edge weights and do
not assume [0, 1] bounds. The sqrt normalization's reward structure (depth +
sub-linear breadth) provides good signal for discovering coalitions that
coordinate both strategically deep (high BC on shared bills) and broadly
(across multiple issue domains).

**Interpretation:**
- DO NOT interpret weight 3.0 as "30% similar" or "3 out of 10."
- DO interpret as: "A and B are aligned on multiple issues. If they share 9
  issues with perfect BC on each, weight = 9/√9 = 3."
- For centrality and community detection, these unbounded weights are
  appropriate and expected.

**Contrast with bill-level BC network:**
The bill similarity network uses weight = mean_BC × breadth, where breadth =
1 − exp(−λ × shared_bill_count) ∈ [0, 1). The exponential breadth term
naturally saturates at 1, giving a final weight bounded in [0, 1]. This is a
different design choice appropriate for that network's structure (many more
bills, different questions). Issue networks use sqrt normalization instead.

**Validation:** Run `validations/05_issue_score_range.py` to see the
mathematical proof, empirical distribution, and share-count breakdown.

---

## References

Bray, J.R., & Curtis, J.T. (1957). An ordination of the upland forest communities
  of southern Wisconsin. Ecological Monographs, 27(4), 325–349.

Guimerà, R., & Amaral, L.A.N. (2005). Functional cartography of complex metabolic
  networks. Nature, 433, 895–900.

Hojnacki, M., Kimball, D.C., Baumgartner, F.R., Berry, J.M., & Leech, B.L. (2012).
  Studying Organizational Advocacy and Influence: Reexamining Interest Group
  Research. Annual Review of Political Science, 15, 379–399.

Koger, G., & Victor, J.N. (2009). Polarized Agents: Campaign Contributions by
  Lobbyists. PS: Political Science & Politics, 42(3), 485–488.

Manning, C.D., Raghavan, P., & Schütze, H. (2008). Introduction to Information
  Retrieval. Cambridge University Press.

Traag, V.A., Waltman, L., & van Eck, N.J. (2019). From Louvain to Leiden: guaranteeing
  well-connected communities. Scientific Reports, 9, 5234.
