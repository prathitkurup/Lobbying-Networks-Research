from pathlib import Path

ROOT         = Path(__file__).parent.parent
DATA_DIR     = ROOT / "data"
LOBBYVIEW_DIR = DATA_DIR / "LobbyView"

NAME_MAPPING      = DATA_DIR / "fortune500_name_mapping.json"
OUTPUT_CSV        = DATA_DIR / "fortune500_lda_reports.csv"
OUTPUT_ISSUES_CSV = DATA_DIR / "fortune500_lda_issues.csv"

TARGET_CONGRESS      = 116
CONGRESS_FILING_YEARS = [2019, 2020]  # 116th Congress: Jan 2019 – Jan 2021

# -- Prevalence filtering --
# Bills lobbied by more than MAX_BILL_DF unique firms are excluded from network
# construction. These omnibus/appropriations bills (CARES Act, HEROES Act, etc.)
# create spurious co-lobbying edges carrying no strategic alignment signal —
# analogous to stop-word removal in NLP (Manning et al., 2008, §6.2).
#
# Empirical calibration: the 16 bills with df > 50 account for 97.5% of all
# affiliation edges and collapse Leiden modularity from Q = 0.18 → Q = 0.02.
# Set to None to disable filtering entirely.
MAX_BILL_DF  = 50

# For issue-level networks: issue codes are much broader (75 codes), and most
# codes are widely shared by design. No frequency filter is applied by default.
MAX_ISSUE_DF = None
