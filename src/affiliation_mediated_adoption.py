"""
Bill-level affiliation-mediated adoption analysis.

For each RBO edge pair (A, B), iterates over their shared top-30 bills and
checks two levels of affiliation:

  Bill-level (strict): A and B share a lobbyist or registrant on their
    first-quarter reports for that specific bill.

  Network-level (broad): A and B share any lobbyist or registrant across
    their full lobbying portfolios (i.e., connected in the affiliation network).

Outputs:
  data/affiliation_mediated_adoption.csv   — one row per (edge, bill)
  data/rbo_edges_enriched.csv             — one row per RBO edge with mediation rates
"""

import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR, OPENSECRETS_OUTPUT_CSV

# -- Constants ----------------------------------------------------------------
INCLUDE_BALANCED   = True   # include balanced (tied) RBO pairs as a comparison group
FIRM_EXTERNAL_ONLY = True   # firm channel: external registrants only (mirrors lobby_firm_affiliation_network.py)
WRITE_CSV          = True

LOB_AFFIL_EDGES = DATA_DIR / "network_edges" / "lobbyist_affiliation_edges.csv"


# -- Data loading -------------------------------------------------------------

def load_data():
    """Load reports, RBO edges, and ranked bill lists."""
    reports = pd.read_csv(OPENSECRETS_OUTPUT_CSV).dropna(subset=["bill_number"])
    rbo     = pd.read_csv(DATA_DIR / "rbo_directed_influence.csv")
    ranked  = pd.read_csv(DATA_DIR / "ranked_bill_lists.csv")
    return reports, rbo, ranked


# -- Lookup builders ----------------------------------------------------------

def build_first_quarter_lookup(reports):
    """Return {(firm, bill): min_quarter} across all congress quarters."""
    return (
        reports.groupby(["fortune_name", "bill_number"])["quarter"]
        .min()
        .to_dict()
    )


def build_first_q_registrants(reports, external_only=True):
    """Return {(firm, bill): set(registrant)} using only the firm's first-quarter reports on each bill.

    external_only=True restricts to non-self-filer (K-street) registrants.
    """
    df = reports.copy()
    if external_only:
        # is_self_filer stored as string "True"/"False" when read from CSV
        df = df[df["is_self_filer"].astype(str).str.lower() != "true"]
    df = df.dropna(subset=["registrant"])

    first_q = (
        df.groupby(["fortune_name", "bill_number"])["quarter"]
        .min()
        .reset_index()
        .rename(columns={"quarter": "first_q"})
    )
    df = df.merge(first_q, on=["fortune_name", "bill_number"])
    df = df[df["quarter"] == df["first_q"]]

    return (
        df.groupby(["fortune_name", "bill_number"])["registrant"]
        .apply(set)
        .to_dict()
    )


def build_first_q_lobbyists(reports):
    """Return {(firm, bill): set(lobbyist_name)} using only the firm's first-quarter reports on each bill."""
    df = reports.dropna(subset=["lobbyists"]).copy()

    first_q = (
        df.groupby(["fortune_name", "bill_number"])["quarter"]
        .min()
        .reset_index()
        .rename(columns={"quarter": "first_q"})
    )
    df = df.merge(first_q, on=["fortune_name", "bill_number"])
    df = df[df["quarter"] == df["first_q"]]

    # Explode pipe-separated lobbyist names into one row per name
    df = df.assign(lobbyist=df["lobbyists"].str.split("|")).explode("lobbyist")
    df["lobbyist"] = df["lobbyist"].str.strip()
    df = df[df["lobbyist"] != ""]

    return (
        df.groupby(["fortune_name", "bill_number"])["lobbyist"]
        .apply(set)
        .to_dict()
    )


def build_top_bills(ranked):
    """Return {company: set(bill_number)} from ranked bill lists."""
    return ranked.groupby("company")["bill_number"].apply(set).to_dict()


def build_network_adjacency(reports, external_only=True):
    """Return undirected adjacency sets for lobbyist-network and firm-network connectivity.

    Uses all reports (not bill-level) to match how affiliation networks are built.
    Returns (lob_adj, firm_adj) where each is a set of (source, target) pairs (both directions).
    """
    # Lobbyist adjacency: read from pre-computed edges file
    lob_adj = set()
    if LOB_AFFIL_EDGES.exists():
        lob_edges = pd.read_csv(LOB_AFFIL_EDGES)
        for _, row in lob_edges.iterrows():
            lob_adj.add((row["source"], row["target"]))
            lob_adj.add((row["target"], row["source"]))

    # Firm adjacency: build inline from reports (mirrors lobby_firm_affiliation_network.py)
    df = reports.copy()
    if external_only:
        df = df[df["is_self_filer"].astype(str).str.lower() != "true"]
    df = df.dropna(subset=["registrant", "fortune_name"])

    registrant_clients = df.groupby("registrant")["fortune_name"].apply(lambda x: sorted(set(x)))
    firm_adj = set()
    for _, clients in registrant_clients.items():
        for i in range(len(clients)):
            for j in range(i + 1, len(clients)):
                firm_adj.add((clients[i], clients[j]))
                firm_adj.add((clients[j], clients[i]))

    return lob_adj, firm_adj


# -- Core analysis ------------------------------------------------------------

def analyze_edges(rbo, top_bills, first_q, firm_reg, lob_reg,
                   lob_adj, firm_adj, include_balanced):
    """Produce one record per (RBO edge, shared bill) with bill-level and network-level mediation flags."""
    edges = rbo if include_balanced else rbo[rbo["balanced"] == 0]

    records = []
    for _, edge in edges.iterrows():
        src      = edge["source"]
        tgt      = edge["target"]
        balanced = int(edge["balanced"])

        # Network-level connectivity is pair-level (same for every bill in pair)
        net_lob_connected  = (src, tgt) in lob_adj
        net_firm_connected = (src, tgt) in firm_adj

        shared = top_bills.get(src, set()) & top_bills.get(tgt, set())

        for bill in shared:
            q_src = first_q.get((src, bill))
            q_tgt = first_q.get((tgt, bill))
            if q_src is None or q_tgt is None:
                continue

            # Bill-level temporal ordering
            if q_src < q_tgt:
                leader, follower = src, tgt
                q_leader, q_follower = q_src, q_tgt
            elif q_tgt < q_src:
                leader, follower = tgt, src
                q_leader, q_follower = q_tgt, q_src
            else:
                leader = follower = None
                q_leader = q_follower = q_src

            lag         = q_follower - q_leader
            is_directed = lag > 0

            # Bill-level: shared affiliation on each firm's first-quarter report for this bill
            shared_firms = firm_reg.get((src, bill), set()) & firm_reg.get((tgt, bill), set())
            shared_lobs  = lob_reg.get((src, bill), set())  & lob_reg.get((tgt, bill), set())

            records.append({
                "rbo_source":              src,
                "rbo_target":              tgt,
                "rbo_balanced":            balanced,
                "bill":                    bill,
                "leader":                  leader,
                "follower":                follower,
                "q_leader":                q_leader,
                "q_follower":              q_follower,
                "lag_quarters":            lag,
                "is_bill_directed":        is_directed,
                # Bill-level mediation (same bill, first-quarter reports)
                "shared_lobbyist_count":   len(shared_lobs),
                "shared_firm_count":       len(shared_firms),
                "shared_lobbyists":        "|".join(sorted(shared_lobs)),
                "shared_firms":            "|".join(sorted(shared_firms)),
                "is_lobbyist_mediated":    len(shared_lobs) > 0,
                "is_firm_mediated":        len(shared_firms) > 0,
                "is_any_mediated":         len(shared_lobs) > 0 or len(shared_firms) > 0,
                # Network-level connectivity (any shared lobbyist/firm across full portfolio)
                "net_lob_connected":       net_lob_connected,
                "net_firm_connected":      net_firm_connected,
                "net_any_connected":       net_lob_connected or net_firm_connected,
            })

    return pd.DataFrame(records)


# -- Edge-level aggregation ---------------------------------------------------

def build_edge_summary(df, rbo):
    """Aggregate to one row per RBO edge: directed-bill mediation rates + network connectivity."""
    directed_bills = df[df["is_bill_directed"]]

    agg = (
        directed_bills
        .groupby(["rbo_source", "rbo_target"])
        .agg(
            directed_bills         = ("bill", "count"),
            lobbyist_mediated      = ("is_lobbyist_mediated", "sum"),
            firm_mediated          = ("is_firm_mediated", "sum"),
            any_mediated           = ("is_any_mediated", "sum"),
            mean_lag_quarters      = ("lag_quarters", "mean"),
            # Network-level: constant per pair, take first value
            net_lob_connected      = ("net_lob_connected",  "first"),
            net_firm_connected     = ("net_firm_connected", "first"),
            net_any_connected      = ("net_any_connected",  "first"),
        )
        .reset_index()
    )
    agg["lobbyist_mediation_rate"] = agg["lobbyist_mediated"] / agg["directed_bills"]
    agg["firm_mediation_rate"]     = agg["firm_mediated"]     / agg["directed_bills"]
    agg["any_mediation_rate"]      = agg["any_mediated"]      / agg["directed_bills"]

    # Merge mediation rates into the original RBO edge table
    enriched = rbo.merge(
        agg,
        left_on=["source", "target"],
        right_on=["rbo_source", "rbo_target"],
        how="left",
    ).drop(columns=["rbo_source", "rbo_target"])

    return enriched


# -- Main ---------------------------------------------------------------------

def main():
    print("Loading data...")
    reports, rbo, ranked = load_data()
    print(f"  Reports: {len(reports):,} rows  |  "
          f"RBO edges: {len(rbo):,}  |  "
          f"Firms in ranked lists: {ranked['company'].nunique()}")

    print("Building lookups...")
    first_q   = build_first_quarter_lookup(reports)
    firm_reg  = build_first_q_registrants(reports, external_only=FIRM_EXTERNAL_ONLY)
    lob_reg   = build_first_q_lobbyists(reports)
    top_bills = build_top_bills(ranked)
    lob_adj, firm_adj = build_network_adjacency(reports, external_only=FIRM_EXTERNAL_ONLY)
    print(f"  (firm, bill) first-quarter entries: {len(first_q):,}")
    print(f"  (firm, bill) registrant entries:    {len(firm_reg):,}")
    print(f"  (firm, bill) lobbyist entries:      {len(lob_reg):,}")
    print(f"  Lobbyist-network pairs:             {len(lob_adj) // 2:,}")
    print(f"  Firm-network pairs:                 {len(firm_adj) // 2:,}")

    print("\nAnalyzing bill-level adoptions across all RBO edges...")
    df = analyze_edges(rbo, top_bills, first_q, firm_reg, lob_reg,
                       lob_adj, firm_adj, INCLUDE_BALANCED)

    directed = df[df["is_bill_directed"]]
    tied     = df[~df["is_bill_directed"]]

    print(f"\n  Total bill-level records:        {len(df):,}")
    print(f"  Directed (lag > 0):              {len(directed):,}")
    print(f"  Tied (same quarter):             {len(tied):,}")
    print(f"\n  Lobbyist-mediated (directed):    "
          f"{directed['is_lobbyist_mediated'].sum():,}  "
          f"({100*directed['is_lobbyist_mediated'].mean():.1f}%)")
    print(f"  Firm-mediated (directed):        "
          f"{directed['is_firm_mediated'].sum():,}  "
          f"({100*directed['is_firm_mediated'].mean():.1f}%)")
    print(f"  Any-mediated (directed):         "
          f"{directed['is_any_mediated'].sum():,}  "
          f"({100*directed['is_any_mediated'].mean():.1f}%)")
    print(f"\n  Mean lag (bill-mediated):        "
          f"{directed[directed['is_any_mediated']]['lag_quarters'].mean():.2f} quarters")
    print(f"  Mean lag (non-mediated):         "
          f"{directed[~directed['is_any_mediated']]['lag_quarters'].mean():.2f} quarters")
    print(f"\n  Network-level connectivity (directed bill pairs):")
    print(f"    Lobbyist-network connected:    "
          f"{directed['net_lob_connected'].sum():,}  "
          f"({100*directed['net_lob_connected'].mean():.1f}%)")
    print(f"    Firm-network connected:        "
          f"{directed['net_firm_connected'].sum():,}  "
          f"({100*directed['net_firm_connected'].mean():.1f}%)")

    if WRITE_CSV:
        out = DATA_DIR / "affiliation_mediated_adoption.csv"
        df.to_csv(out, index=False)
        print(f"\n  Bill-level output  -> {out.name}  ({len(df):,} rows)")

        enriched = build_edge_summary(df, rbo)
        enriched_out = DATA_DIR / "rbo_edges_enriched.csv"
        enriched.to_csv(enriched_out, index=False)
        print(f"  Edge-level output  -> {enriched_out.name}  ({len(enriched):,} rows)")


if __name__ == "__main__":
    main()
