"""
Build fortune500_opensecrets_name_mapping.json from OpenSecrets CRP data.

Scans lob_lobbying.txt for CRP-standardized client and ultorg strings, maps
them to Fortune 500 canonicals via two-tier normalization, then merges a
manual-variant dictionary for names that cannot auto-match.

Run: python build_opensecrets_mapping.py
"""

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR, NAME_MAPPING, OPENSECRETS_NAME_MAPPING

LOBBYING_FILE  = DATA_DIR / "OpenSecrets" / "lob_lobbying.txt"
CONGRESS_YEARS = {"2019", "2020"}
SCAN_ALL_YEARS = False

# ---------------------------------------------------------------------------
# Manual variants for names that cannot be auto-matched.
# Keys are Fortune 500 canonicals; values are CRP strings to add.
# These cover: digit-vs-word numerals, merged/renamed entities, single-word
# canonicals whose CRP name adds descriptive words, and punctuation mismatches.
# ---------------------------------------------------------------------------
MANUAL_VARIANTS = {
    # "Exxon Mobil" (space) cannot match canonical "EXXONMOBIL" (no space)
    "EXXONMOBIL":                  ["Exxon Mobil"],
    # "Ford Motor Co" bare-strips to "FORD MOTOR" -> does not match "FORD"
    "FORD":                        ["Ford Motor Co"],
    # "Lowe's Companies" bare-strips to "LOWES COMPANIES" -> does not match "LOWES"
    "LOWE'S":                      ["Lowe's Companies"],
    # "Costco Wholesale" bare-strips to "COSTCO WHOLESALE" -> does not match "COSTCO"
    "COSTCO":                      ["Costco Wholesale"],
    # "U.S." punct-strips to "U S" (space) so bare key "US BANCORP" != "U S BANCORP"
    "U.S. BANCORP":                ["US Bancorp"],
    # Numeral form vs. word form; company split into Fox Corp in March 2019
    "TWENTY-FIRST CENTURY FOX":    ["21st Century Fox", "Twenty-First Century Fox",
                                    "Fox Corp", "Fox Corporation"],
    # Renamed from United Continental Holdings to United Airlines Holdings in 2019
    "UNITED CONTINENTAL HOLDINGS": ["United Airlines Holdings", "United Airlines"],
    # CRP uses "DuPont Co" for the legacy entity; new entity is "DuPont de Nemours"
    "DUPONT DE NEMOURS":           ["DuPont Co", "DuPont de Nemours",
                                    "E.I. du Pont de Nemours"],
    # "!" is not in the default punct regex; add as backup
    "YUM BRANDS":                  ["YUM! Brands"],
    # BB&T merged with SunTrust to form Truist Financial in Dec 2019
    "BB&T CORP.":                  ["BB&T Corp", "BB&T", "Truist Financial",
                                    "Truist Bank"],
    "SUNTRUST BANKS":              ["SunTrust Banks", "SunTrust Financial",
                                    "Truist Financial", "Truist Bank"],
    # Celgene acquired by BMS in late 2019; filed as "Celgene Corp" in CRP.
    # Do NOT include "Bristol-Myers Squibb" here — that is BRISTOL-MYERS SQUIBB's
    # own canonical name and would collide with it in the lookup.
    "CELGENE":                     ["Celgene Corp"],
    # L3 Technologies merged with Harris to form L3Harris in June 2019
    "L3 TECHNOLOGIES":             ["L3Harris Technologies", "L3 Technologies",
                                    "Harris Corp"],
    # Rockwell Collins was acquired by United Technologies and became Collins Aerospace
    "ROCKWELL COLLINS":            ["Collins Aerospace", "Rockwell Collins"],
    # First Data was acquired by Fiserv in July 2019
    "FIRST DATA":                  ["Fiserv", "First Data Corp"],
    # AIG is the common abbreviation; CRP uses "American International Group"
    "AIG":                         ["American International Group",
                                    "AIG American International Group"],
    # Fannie Mae's legal name is Federal National Mortgage Association
    "FANNIE MAE":                  ["Federal National Mortgage Association",
                                    "Fannie Mae"],
    # Freddie Mac's legal name is Federal Home Loan Mortgage Corp
    "FREDDIE MAC":                 ["Federal Home Loan Mortgage",
                                    "Federal Home Loan Mortgage Corp",
                                    "Freddie Mac"],
    # HEALTH removed from suffix list to prevent false positives;
    # add explicit variant so "UnitedHealth Group" / "United Health Group" still matches
    "UNITED HEALTH GROUP":         ["UnitedHealth Group", "United Health Group",
                                    "United HealthCare Services"],
    # Viacom merged with CBS to form ViacomCBS in August 2019
    "VIACOM":                      ["ViacomCBS", "Viacom Inc", "CBS Corp",
                                    "Paramount Global"],
}

# ---------------------------------------------------------------------------
# Corporate suffix tokens — stripped for bare-normalization matching.
# Ordered longest-first so greedy removal handles overlapping tokens safely.
# ---------------------------------------------------------------------------
_SUFFIX_TOKENS = (
    "CORPORATION", "INCORPORATED", "INTERNATIONAL", "TECHNOLOGIES",
    "TECHNOLOGY", "PHARMACEUTICALS", "PHARMACEUTICAL", "COMMUNICATIONS",
    "HOLDINGS", "HOLDING", "FINANCIAL", "ENTERPRISES", "INDUSTRIES",
    "SOLUTIONS", "SERVICES", "PROPERTIES", "RESOURCES", "NETWORKS",
    "HEALTHCARE", "ELECTRONICS", "ELECTRIC", "PARTNERS", "PRODUCTS",
    "BANCSHARES", "INSURANCE", "CAPITAL", "GLOBAL", "ENERGY", "BRANDS",
    "SYSTEMS", "NETWORK", "PHARMA", "BANCORP", "PROPERTY",
    "AMERICAS", "AMERICA", "GROUP", "INTL", "BANK", "CORP", "TECH",
    "LIMITED", "COMPANY", "COMPANIES", "WHOLESALE", "MOTOR",
    "INC", "LTD", "LLC", "PLC", "LLP", "CO", "LP",
    "US", "USA", "NA",
    # Web domain suffixes (e.g. Amazon.com -> AMAZON COM -> AMAZON)
    "COM", "NET", "ORG", "IO",
)
_SUFFIX_RE = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in _SUFFIX_TOKENS) + r")\b"
)
# Punctuation to replace with spaces (includes ! for "YUM! Brands" etc.)
_PUNCT_RE = re.compile(r"[,.\-&'/!]")
# Collapse spaced single-letter abbreviations: "U S" -> "US", "J P" -> "JP"
_ABBREV_RE = re.compile(r"\b([A-Z]) ([A-Z])\b")


def _norm_exact(s):
    """Uppercase and collapse whitespace only."""
    return re.sub(r"\s+", " ", s.upper().strip())


def _norm_bare(s):
    """Uppercase, strip punctuation, collapse abbreviations, remove suffix tokens."""
    s = _PUNCT_RE.sub(" ", s.upper())
    # Collapse spaced-initial sequences: "U S BANCORP" -> "US BANCORP"
    # Run twice to catch triple sequences like "A T T"
    s = _ABBREV_RE.sub(r"\1\2", s)
    s = _ABBREV_RE.sub(r"\1\2", s)
    s = _SUFFIX_RE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def load_canonical_names():
    """Return Fortune 500 canonical names from lobbyview_fortune500_name_mapping.json."""
    with open(NAME_MAPPING) as f:
        return list(json.load(f).keys())


def build_lookup(canonicals):
    """Build exact_lookup and bare_lookup dicts from canonical names."""
    exact = {}
    bare  = {}
    for name in canonicals:
        exact[_norm_exact(name)] = name
        bk = _norm_bare(name)
        if bk and bk not in bare:
            bare[bk] = name
    return exact, bare


def scan_crp_names(path, years):
    """Collect unique CRP-standardized client and ultorg strings."""
    seen = set()
    with open(path, "r", encoding="latin-1", errors="replace") as f:
        reader = csv.reader(f, quotechar="|", delimiter=",", doublequote=False)
        for row in reader:
            if len(row) <= 14:
                continue
            if years is not None and row[14].strip() not in years:
                continue
            client = row[5].strip()
            ultorg = row[6].strip()
            if client:
                seen.add(client)
            if ultorg:
                seen.add(ultorg)
    return seen


def match_crp_names(crp_names, exact_lookup, bare_lookup):
    """Resolve each CRP name to a Fortune 500 canonical and return variants dict."""
    variants = defaultdict(list)
    for name in sorted(crp_names):
        canonical = exact_lookup.get(_norm_exact(name))
        if canonical is None:
            canonical = bare_lookup.get(_norm_bare(name))
        if canonical is None:
            continue
        if _norm_exact(name) == _norm_exact(canonical):
            continue
        variants[canonical].append(name)
    return dict(variants)


def apply_manual_variants(variants, canonicals):
    """Merge MANUAL_VARIANTS into the auto-detected variants dict.

    Also removes any manual-variant name that was spuriously auto-assigned to a
    *different* canonical, preventing lookup collisions (last-write-wins issue).
    """
    canonical_set = set(canonicals)
    added = 0
    removed = 0

    # Build set of all manual variant strings (norm_exact) -> their correct canonical
    manual_norm_to_canonical = {}
    for canonical, names in MANUAL_VARIANTS.items():
        for n in names:
            manual_norm_to_canonical[_norm_exact(n)] = canonical

    # Remove any auto-detected variation that is "owned" by a manual entry
    # and has been assigned to the wrong canonical.
    for canonical, var_list in list(variants.items()):
        cleaned = []
        for v in var_list:
            correct = manual_norm_to_canonical.get(_norm_exact(v))
            if correct is not None and correct != canonical:
                removed += 1  # strip from wrong canonical
            else:
                cleaned.append(v)
        variants[canonical] = cleaned

    # Now add manual variants to their correct canonicals
    for canonical, names in MANUAL_VARIANTS.items():
        if canonical not in canonical_set:
            print(f"  [manual] Warning: {canonical!r} not in canonical list — skipped")
            continue
        existing = set(variants.get(canonical, []))
        new_names = [n for n in names if n not in existing
                     and _norm_exact(n) != _norm_exact(canonical)]
        if new_names:
            variants.setdefault(canonical, [])
            variants[canonical].extend(new_names)
            added += len(new_names)

    print(f"  Manual variants merged: {added} new strings added, "
          f"{removed} conflicting auto-detections removed")
    return variants


def coverage_report(canonicals, crp_names, exact_lookup, bare_lookup,
                    variants, manual=None):
    """Print a coverage summary including manual variant contributions."""
    covered = set()
    for name in crp_names:
        c = exact_lookup.get(_norm_exact(name))
        if c is None:
            c = bare_lookup.get(_norm_bare(name))
        if c:
            covered.add(c)
    # Also count canonicals that will be reached via their canonical key directly
    for c in canonicals:
        if _norm_exact(c) in exact_lookup:
            covered.add(c)
    # And via manual variants
    if manual:
        for c in manual:
            if c in set(canonicals):
                covered.add(c)

    auto_variants  = sum(1 for c in canonicals if c in variants and variants.get(c))
    print(f"\n-- Coverage report --")
    print(f"  Fortune 500 canonicals          : {len(canonicals)}")
    print(f"  Firms resolvable from CRP data  : {len(covered)} / {len(canonicals)} "
          f"({100 * len(covered) / len(canonicals):.1f}%)")
    print(f"  Firms with auto-detected variants: {auto_variants}")

    unmatched = sorted(set(canonicals) - covered)
    if unmatched:
        preview = ", ".join(unmatched[:15])
        tail    = " ..." if len(unmatched) > 15 else ""
        print(f"  Not matched ({len(unmatched)}): {preview}{tail}")
    return unmatched


def build_and_write(canonicals, variants):
    """Write the mapping JSON."""
    mapping = {
        name: {
            "variations":   sorted(set(variants.get(name, []))),
            "subsidiaries": [],
        }
        for name in canonicals
    }
    OPENSECRETS_NAME_MAPPING.parent.mkdir(parents=True, exist_ok=True)
    with open(OPENSECRETS_NAME_MAPPING, "w") as f:
        json.dump(mapping, f, indent=2)
    total_variants = sum(len(v["variations"]) for v in mapping.values())
    print(f"\n  Written -> {OPENSECRETS_NAME_MAPPING}")
    print(f"  Total variation strings stored  : {total_variants:,}")


def main():
    print("Loading Fortune 500 canonical names...")
    canonicals = load_canonical_names()
    print(f"  Canonical names: {len(canonicals)}")

    exact_lookup, bare_lookup = build_lookup(canonicals)

    scope = "all years" if SCAN_ALL_YEARS else "2019-2020 (116th Congress)"
    print(f"\nScanning {LOBBYING_FILE.name} for CRP names ({scope})...")
    years      = None if SCAN_ALL_YEARS else CONGRESS_YEARS
    crp_names  = scan_crp_names(LOBBYING_FILE, years)
    print(f"  Unique CRP name strings found: {len(crp_names):,}")

    print("\nMatching CRP names to Fortune 500 canonicals (auto)...")
    variants = match_crp_names(crp_names, exact_lookup, bare_lookup)

    print("Merging manual variants...")
    variants = apply_manual_variants(variants, canonicals)

    unmatched = coverage_report(canonicals, crp_names, exact_lookup, bare_lookup,
                                variants, manual=MANUAL_VARIANTS)

    sample = [(c, variants[c][:3]) for c in canonicals
              if c in variants and variants[c]][:12]
    if sample:
        print("\nSample auto+manual matches:")
        for canonical, vs in sample:
            print(f"  {canonical:35s} <- {vs}")

    build_and_write(canonicals, variants)

    if unmatched:
        print(f"\n-- {len(unmatched)} firms with no CRP match (genuine non-lobbiers "
              f"or unresolvable name differences) --")
        for f in unmatched:
            print(f"  {f}")


if __name__ == "__main__":
    main()
