"""
Validation and analysis of bill-level affiliation-mediated adoption.

Runs after affiliation_mediated_adoption.py has produced:
  data/affiliation_mediated_adoption.csv
  data/rbo_edges_enriched.csv

Analyses:
  1. Overall mediation rates (directed vs tied bill pairs)
  2. Mediation rate by RBO edge type (directed vs balanced edge)
  3. Lag distribution: mediated vs non-mediated (Mann-Whitney U test)
  4. Top broker lobbyists and firms (by number of mediated adoptions bridged)
  5. Per-bill mediation frequency (most-transmitted bills)
  6. Edge-level mediation rate distribution
  7. Alignment test: do mediated bills confirm the RBO edge direction?
  8. Co-mediation: how often lobbyist and firm channels overlap

Run: python src/validations/11_mediated_adoption_validation.py
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR

MEDIATION_CSV = DATA_DIR / "affiliation_mediated_adoption.csv"
ENRICHED_CSV  = DATA_DIR / "rbo_edges_enriched.csv"
OUTPUT_PATH = Path(__file__).resolve().parent.parent.parent / "outputs" / "validation" / "11_mediated_adoption_validation.txt"

class _Tee:
    """Write to both stdout and a file simultaneously."""
    def __init__(self, *streams): self.streams = streams
    def write(self, text):
        for s in self.streams: s.write(text)
    def flush(self):
        for s in self.streams: s.flush()

SEP = "-" * 72


def load_data():
    """Load mediation and enriched edge datasets; validate required columns."""
    df  = pd.read_csv(MEDIATION_CSV)
    rbo = pd.read_csv(ENRICHED_CSV)

    required_med = [
        "rbo_source", "rbo_target", "rbo_balanced", "bill",
        "leader", "follower", "lag_quarters", "is_bill_directed",
        "is_lobbyist_mediated", "is_firm_mediated", "is_any_mediated",
        "shared_lobbyists", "shared_firms",
        "net_lob_connected", "net_firm_connected", "net_any_connected",
    ]
    required_enr = [
        "source", "target", "net_temporal", "weight",
        "any_mediation_rate", "lobbyist_mediation_rate", "firm_mediation_rate",
    ]
    for col in required_med:
        assert col in df.columns,  f"Missing column in mediation CSV: {col}"
    for col in required_enr:
        assert col in rbo.columns, f"Missing column in enriched CSV: {col}"

    return df, rbo


# ── 1. Overall mediation rates ───────────────────────────────────────────────

def section_overall(df):
    print(SEP)
    print("1. OVERALL MEDIATION RATES")
    print(SEP)

    all_pairs  = df
    directed   = df[df["is_bill_directed"]]
    tied       = df[~df["is_bill_directed"]]

    print(f"  Total (edge, bill) records:     {len(all_pairs):,}")
    print(f"  Directed (lag > 0):             {len(directed):,}  "
          f"({100*len(directed)/len(all_pairs):.1f}%)")
    print(f"  Tied (lag == 0):                {len(tied):,}  "
          f"({100*len(tied)/len(all_pairs):.1f}%)")

    print()
    for label, sub in [("Directed pairs", directed), ("Tied pairs", tied)]:
        n = len(sub)
        if n == 0:
            continue
        lob_r  = sub["is_lobbyist_mediated"].mean()
        firm_r = sub["is_firm_mediated"].mean()
        any_r  = sub["is_any_mediated"].mean()
        print(f"  {label} (n={n:,}):")
        print(f"    Lobbyist-mediated: {sub['is_lobbyist_mediated'].sum():,}  ({100*lob_r:.1f}%)")
        print(f"    Firm-mediated:     {sub['is_firm_mediated'].sum():,}  ({100*firm_r:.1f}%)")
        print(f"    Any-mediated:      {sub['is_any_mediated'].sum():,}  ({100*any_r:.1f}%)")
    print()


# ── 2. Mediation rate by RBO edge type ───────────────────────────────────────

def section_by_edge_type(df):
    print(SEP)
    print("2. MEDIATION RATE BY RBO EDGE TYPE (directed vs balanced edge)")
    print(SEP)
    print("  Directed (lag > 0) bill pairs within each edge type:\n")

    directed = df[df["is_bill_directed"]]
    for edge_type, label in [(0, "Decisive RBO pair (net_temporal ≠ 0)"),
                              (1, "Balanced RBO pair (net_temporal = 0)")]:
        sub = directed[directed["rbo_balanced"] == edge_type]
        if len(sub) == 0:
            continue
        print(f"  {label}  (n={len(sub):,}):")
        print(f"    Any-mediated:      {sub['is_any_mediated'].sum():,}  "
              f"({100*sub['is_any_mediated'].mean():.1f}%)")
        print(f"    Lobbyist-mediated: {sub['is_lobbyist_mediated'].sum():,}  "
              f"({100*sub['is_lobbyist_mediated'].mean():.1f}%)")
        print(f"    Firm-mediated:     {sub['is_firm_mediated'].sum():,}  "
              f"({100*sub['is_firm_mediated'].mean():.1f}%)")
    print()


# ── 3. Lag distribution: mediated vs non-mediated ────────────────────────────

def section_lag(df):
    print(SEP)
    print("3. LAG DISTRIBUTION: MEDIATED vs NON-MEDIATED")
    print(SEP)
    print("  Hypothesis (Carpenter et al. 1998): shared affiliation compresses")
    print("  the adoption lag — signals travel faster through a live channel.\n")

    directed = df[df["is_bill_directed"]]

    for channel, col in [("Any", "is_any_mediated"),
                          ("Lobbyist", "is_lobbyist_mediated"),
                          ("Firm", "is_firm_mediated")]:
        med   = directed[directed[col]]["lag_quarters"]
        nomed = directed[~directed[col]]["lag_quarters"]

        if len(med) < 2 or len(nomed) < 2:
            continue

        stat, pval = stats.mannwhitneyu(med, nomed, alternative="less")

        print(f"  {channel}-mediated (n={len(med):,})  vs  "
              f"Non-mediated (n={len(nomed):,}):")
        print(f"    Mean lag  — mediated: {med.mean():.2f} q  |  "
              f"non-mediated: {nomed.mean():.2f} q")
        print(f"    Median    — mediated: {med.median():.1f} q  |  "
              f"non-mediated: {nomed.median():.1f} q")
        print(f"    Mann-Whitney U (one-sided, mediated < non-mediated): "
              f"U={stat:.0f}  p={pval:.4f}")

        lag_dist = (directed.groupby([col, "lag_quarters"])
                    .size()
                    .rename("count")
                    .reset_index())
        print(f"    Lag distribution (mediated=True):")
        med_dist = lag_dist[lag_dist[col]].sort_values("lag_quarters")
        for _, row in med_dist.iterrows():
            bar = "█" * min(int(row["count"] / max(med_dist["count"]) * 20), 20)
            print(f"      lag={int(row['lag_quarters'])} q:  {row['count']:>4}  {bar}")
        print()


# ── 4. Top broker lobbyists and firms ────────────────────────────────────────

def section_brokers(df):
    print(SEP)
    print("4. TOP BROKER LOBBYISTS AND FIRMS")
    print(SEP)
    print("  Brokers: intermediaries whose affiliation appears in mediated")
    print("  directed adoptions — the proposed transmission conduit.\n")

    directed_mediated = df[df["is_bill_directed"]]

    # Lobbyist brokers
    lob_rows = directed_mediated[directed_mediated["is_lobbyist_mediated"]].copy()
    if len(lob_rows):
        lob_exploded = (
            lob_rows.assign(lobbyist=lob_rows["shared_lobbyists"].str.split("|"))
            .explode("lobbyist")
        )
        lob_exploded["lobbyist"] = lob_exploded["lobbyist"].str.strip()
        lob_exploded = lob_exploded[lob_exploded["lobbyist"] != ""]
        top_lob = (
            lob_exploded.groupby("lobbyist")
            .agg(
                mediated_adoptions = ("bill", "count"),
                unique_bills       = ("bill", "nunique"),
                unique_pairs       = ("rbo_source", "nunique"),
            )
            .sort_values("mediated_adoptions", ascending=False)
            .head(15)
        )
        print("  Top 15 broker lobbyists:")
        print(f"  {'Lobbyist':<45} {'Adoptions':>9}  {'Bills':>6}  {'Pairs':>6}")
        for lob, row in top_lob.iterrows():
            print(f"  {lob:<45} {int(row['mediated_adoptions']):>9}  "
                  f"{int(row['unique_bills']):>6}  {int(row['unique_pairs']):>6}")
        print()

    # Firm brokers
    firm_rows = directed_mediated[directed_mediated["is_firm_mediated"]].copy()
    if len(firm_rows):
        firm_exploded = (
            firm_rows.assign(firm=firm_rows["shared_firms"].str.split("|"))
            .explode("firm")
        )
        firm_exploded["firm"] = firm_exploded["firm"].str.strip()
        firm_exploded = firm_exploded[firm_exploded["firm"] != ""]
        top_firm = (
            firm_exploded.groupby("firm")
            .agg(
                mediated_adoptions = ("bill", "count"),
                unique_bills       = ("bill", "nunique"),
                unique_pairs       = ("rbo_source", "nunique"),
            )
            .sort_values("mediated_adoptions", ascending=False)
            .head(15)
        )
        print("  Top 15 broker lobbying firms:")
        print(f"  {'Firm':<50} {'Adoptions':>9}  {'Bills':>6}  {'Pairs':>6}")
        for firm, row in top_firm.iterrows():
            print(f"  {firm:<50} {int(row['mediated_adoptions']):>9}  "
                  f"{int(row['unique_bills']):>6}  {int(row['unique_pairs']):>6}")
        print()


# ── 5. Per-bill mediation frequency ──────────────────────────────────────────

def section_bills(df):
    print(SEP)
    print("5. PER-BILL MEDIATION FREQUENCY (top 20 most-transmitted bills)")
    print(SEP)
    print("  Bills with the highest lobbyist-mediated directed adoption count.\n")

    directed = df[df["is_bill_directed"]]
    bill_stats = (
        directed.groupby("bill")
        .agg(
            total_directed_pairs   = ("rbo_source", "count"),
            lobbyist_mediated      = ("is_lobbyist_mediated", "sum"),
            firm_mediated          = ("is_firm_mediated", "sum"),
            any_mediated           = ("is_any_mediated", "sum"),
            mean_lag               = ("lag_quarters", "mean"),
        )
        .assign(any_mediation_rate=lambda x: x["any_mediated"] / x["total_directed_pairs"])
        .sort_values("any_mediated", ascending=False)
        .head(20)
    )

    print(f"  {'Bill':<15} {'Dir.Pairs':>9}  {'Lob.Med':>7}  {'Firm.Med':>8}  "
          f"{'Any.Med':>7}  {'Rate':>6}  {'MeanLag':>8}")
    for bill, row in bill_stats.iterrows():
        print(f"  {bill:<15} {int(row['total_directed_pairs']):>9}  "
              f"{int(row['lobbyist_mediated']):>7}  {int(row['firm_mediated']):>8}  "
              f"{int(row['any_mediated']):>7}  {row['any_mediation_rate']:>6.2f}  "
              f"{row['mean_lag']:>8.2f}")
    print()


# ── 6. Edge-level mediation rate distribution ─────────────────────────────────

def section_edge_distribution(rbo):
    print(SEP)
    print("6. EDGE-LEVEL MEDIATION RATE DISTRIBUTION")
    print(SEP)
    print("  For each directed RBO edge: fraction of its directed bills that are")
    print("  affiliation-mediated (any channel).\n")

    directed_rbo = rbo[rbo["net_temporal"] != 0].dropna(subset=["any_mediation_rate"])

    print(f"  Directed RBO edges with >= 1 directed bill:  {len(directed_rbo):,}")
    print(f"  Mean any_mediation_rate:   {directed_rbo['any_mediation_rate'].mean():.3f}")
    print(f"  Median:                    {directed_rbo['any_mediation_rate'].median():.3f}")
    print(f"  Edges with rate > 0:       "
          f"{(directed_rbo['any_mediation_rate'] > 0).sum():,}  "
          f"({100*(directed_rbo['any_mediation_rate'] > 0).mean():.1f}%)")
    print(f"  Edges with rate == 1.0:    "
          f"{(directed_rbo['any_mediation_rate'] == 1.0).sum():,}")

    # Histogram buckets
    bins  = [0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.01]
    labels = ["0", "(0–0.1]", "(0.1–0.25]", "(0.25–0.5]", "(0.5–0.75]", "(0.75–1.0]"]
    counts, _ = np.histogram(directed_rbo["any_mediation_rate"], bins=bins)
    print(f"\n  Distribution of any_mediation_rate:")
    for label, count in zip(labels, counts):
        bar = "█" * min(int(count / max(counts) * 25), 25)
        print(f"    {label:<12} {count:>5}  {bar}")
    print()

    # Top 10 most-mediated directed edges
    top = directed_rbo.nlargest(10, "any_mediation_rate")
    print("  Top 10 directed edges by any_mediation_rate:")
    print(f"  {'Source':<40} {'Target':<40} {'Rate':>6}  {'DirBills':>8}")
    for _, row in top.iterrows():
        print(f"  {row['source']:<40} {row['target']:<40} "
              f"{row['any_mediation_rate']:>6.2f}  {int(row['directed_bills']):>8}")
    print()


# ── 7. Network-level connectivity analysis ───────────────────────────────────

def section_network_level(df):
    print(SEP)
    print("7. NETWORK-LEVEL CONNECTIVITY vs BILL-LEVEL CO-AFFILIATION")
    print(SEP)
    print("  Bill-level: same lobbyist/firm appears on both companies' first-quarter")
    print("  reports FOR THAT SPECIFIC BILL. Network-level: companies share any")
    print("  lobbyist/firm across their FULL lobbying portfolios.\n")
    print("  Finding: direct bill-level co-affiliation is rare; most RBO influence")
    print("  pairs are not connected even at the portfolio level — the RBO network")
    print("  captures a broader coordination signal (Carpenter et al. 1998).\n")

    directed = df[df["is_bill_directed"]]
    n = len(directed)

    print(f"  Directed bill pairs (n={n:,}):")
    print(f"  {'Channel':<40} {'Bill-level':>12}  {'Network-level':>15}")
    print(f"  {'-'*68}")
    for label, bill_col, net_col in [
        ("Lobbyist", "is_lobbyist_mediated", "net_lob_connected"),
        ("Firm (external K-street)", "is_firm_mediated", "net_firm_connected"),
        ("Any", "is_any_mediated", "net_any_connected"),
    ]:
        b_n = directed[bill_col].sum()
        b_p = 100 * directed[bill_col].mean()
        n_n = directed[net_col].sum()
        n_p = 100 * directed[net_col].mean()
        print(f"  {label:<40} {b_n:>5} ({b_p:>4.1f}%)  {n_n:>6} ({n_p:>4.1f}%)")

    print()
    # For the rare network-connected pairs, show their RBO weight vs non-connected
    for label, col in [("Lobbyist-network", "net_lob_connected"),
                        ("Firm-network", "net_firm_connected")]:
        conn    = directed[directed[col]]["lag_quarters"]
        nonconn = directed[~directed[col]]["lag_quarters"]
        if len(conn) < 2:
            continue
        stat, pval = stats.mannwhitneyu(conn, nonconn, alternative="less")
        print(f"  {label} lag (n={len(conn)}):  "
              f"mean={conn.mean():.2f} q  vs  non-connected mean={nonconn.mean():.2f} q")
        print(f"    Mann-Whitney U (connected < non-connected):  p={pval:.4f}")
    print()


# ── 8. Alignment test ────────────────────────────────────────────────────────

def section_alignment(df):
    print(SEP)
    print("8. ALIGNMENT TEST: DO MEDIATED BILLS CONFIRM THE RBO EDGE DIRECTION?")
    print(SEP)
    print("  For directed RBO edges (balanced=0), check whether the bill-level")
    print("  leader (first adopter) is also the RBO edge source — i.e., whether")
    print("  affiliation-mediated bills are consistent with the aggregate direction.\n")
    print("  If shared affiliation channels information from influencer to follower,")
    print("  mediated bills should more often have rbo_source as the leader.\n")

    # Only directed RBO edges, directed bills (lag > 0)
    sub = df[(df["rbo_balanced"] == 0) & (df["is_bill_directed"])].copy()
    if len(sub) == 0:
        print("  No data — skipping.\n")
        return

    sub["leader_is_source"] = sub["leader"] == sub["rbo_source"]

    for med_label, mask in [
        ("All directed bills",            sub["leader_is_source"].notna()),
        ("Non-mediated directed bills",   ~sub["is_any_mediated"]),
        ("Any-mediated directed bills",    sub["is_any_mediated"]),
        ("Lobbyist-mediated bills",        sub["is_lobbyist_mediated"]),
        ("Firm-mediated bills",            sub["is_firm_mediated"]),
    ]:
        grp = sub[mask]
        if len(grp) == 0:
            continue
        rate = grp["leader_is_source"].mean()
        # Binomial test: is the alignment rate > 0.5?
        n_align  = grp["leader_is_source"].sum()
        n_total  = len(grp)
        binom    = stats.binomtest(int(n_align), int(n_total), p=0.5, alternative="greater")
        print(f"  {med_label} (n={n_total:,}):")
        print(f"    Leader == RBO source:  {int(n_align):,} / {n_total:,}  "
              f"({100*rate:.1f}%)")
        print(f"    Binomial test (rate > 0.5):  p={binom.pvalue:.4f}")
    print()


# ── 9. Co-mediation: lobbyist and firm channel overlap ────────────────────────

def section_co_mediation(df):
    print(SEP)
    print("9. CO-MEDIATION: LOBBYIST AND FIRM CHANNEL OVERLAP")
    print(SEP)
    print("  How often do both channels fire on the same directed adoption?\n")

    directed = df[df["is_bill_directed"]]
    n = len(directed)

    lob_only  = directed["is_lobbyist_mediated"] & ~directed["is_firm_mediated"]
    firm_only = directed["is_firm_mediated"]     & ~directed["is_lobbyist_mediated"]
    both      = directed["is_lobbyist_mediated"] & directed["is_firm_mediated"]
    neither   = ~directed["is_lobbyist_mediated"] & ~directed["is_firm_mediated"]

    print(f"  {'Category':<28} {'Count':>7}  {'Pct':>7}")
    print(f"  {'Lobbyist only':<28} {lob_only.sum():>7}  "
          f"{100*lob_only.mean():>6.1f}%")
    print(f"  {'Firm only':<28} {firm_only.sum():>7}  "
          f"{100*firm_only.mean():>6.1f}%")
    print(f"  {'Both channels':<28} {both.sum():>7}  "
          f"{100*both.mean():>6.1f}%")
    print(f"  {'Neither (unmediated)':<28} {neither.sum():>7}  "
          f"{100*neither.mean():>6.1f}%")
    print(f"  {'TOTAL':<28} {n:>7}  100.0%\n")


# -- Main ---------------------------------------------------------------------

def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _orig_stdout = sys.stdout
    _f = open(OUTPUT_PATH, "w")
    sys.stdout = _Tee(_orig_stdout, _f)

    try:
        print(f"\n{'═'*72}")
        print("AFFILIATION-MEDIATED ADOPTION — VALIDATION REPORT")
        print(f"{'═'*72}\n")
        print("  Grounding: Carpenter et al. (1998), Koger & Victor (2009),")
        print("  Hojnacki & Kimball (1998). Shared lobbyists and lobbying firms")
        print("  create information bridges; bill-level adoption ordering tests")
        print("  whether those bridges predict the direction of legislative adoption.\n")

        df, rbo = load_data()

        print(f"  Loaded {len(df):,} (edge, bill) records  "
              f"and {len(rbo):,} RBO edge records.\n")

        section_overall(df)
        section_by_edge_type(df)
        section_lag(df)
        section_brokers(df)
        section_bills(df)
        section_edge_distribution(rbo)
        section_network_level(df)
        section_alignment(df)
        section_co_mediation(df)

        print(f"{'═'*72}")
        print("END OF REPORT")
        print(f"{'═'*72}\n")

    finally:
        sys.stdout = _orig_stdout
        _f.close()


if __name__ == "__main__":
    main()
