"""
Enrich lobbyview_fortune500_name_mapping.json with subsidiary names sourced from
SEC Exhibit 21 filings (10-K annual reports). Only subsidiaries that appear as
actual client names in the LobbyView clients table are added.

Run: python build_lobbyview_mapping.py
Note: Makes ~500 HTTP requests to SEC EDGAR. Expect ~5-10 minutes runtime.
"""

import json
import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
from config import DATA_DIR, LOBBYVIEW_DIR, NAME_MAPPING


# -- SEC config --

SEC_HEADERS = {
    "User-Agent": "Prathit Kurup (pkurup@bowdoin.edu)",
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}
SEC_HEADERS_WWW = {
    "User-Agent": "Prathit Kurup (pkurup@bowdoin.edu)",
    "Accept-Encoding": "gzip, deflate",
}
TICKERS_URL     = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SLEEP           = 0.2  # be polite to SEC servers


# -- Helpers --

def norm(s):
    """Uppercase, strip punctuation and extra whitespace."""
    s = s.upper()
    s = re.sub(r"[^A-Z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def jaccard(a, b):
    A, B = set(norm(a).split()), set(norm(b).split())
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def get_json(url, headers):
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()

def get_text(url, headers):
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return r.text


# -- SEC fetching --

def load_sec_tickers():
    data = get_json(TICKERS_URL, headers=SEC_HEADERS_WWW)
    return list(data.values())

def match_to_sec(canonical, sec_entries):
    """Best Jaccard match between canonical name and SEC company title."""
    best, best_score = None, 0.0
    for entry in sec_entries:
        score = jaccard(canonical, entry.get("title", ""))
        if score > best_score:
            best_score, best = score, entry
    return best if best_score >= 0.25 else None

def find_latest_10k(submissions):
    recent = submissions.get("filings", {}).get("recent", {})
    for form, acc, date in zip(
        recent.get("form", []),
        recent.get("accessionNumber", []),
        recent.get("filingDate", [])
    ):
        if form in ("10-K", "10-K/A"):
            return acc, date
    return None, None

def archive_base(cik_int, accession):
    acc = accession.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc}/"

def find_ex21(index_json):
    """Pick the Exhibit 21 file from an accession folder listing."""
    items = index_json.get("directory", {}).get("item", [])
    candidates = [
        it["name"] for it in items
        if any(k in it["name"].lower() for k in ["ex21", "ex-21", "exhibit21", "exh21"])
        and it["name"].lower().endswith((".htm", ".html", ".txt"))
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda x: (not x.lower().endswith((".htm", ".html")), len(x)))
    return candidates[0]


# -- Exhibit 21 parsing --

def parse_ex21(html):
    """Extract subsidiary names from Exhibit 21 HTML/text."""
    soup = BeautifulSoup(html, "html.parser")
    rows = []

    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            cells = [c for c in cells if c]
            if not cells:
                continue
            name = cells[0]
            if "company name" in name.lower():
                continue
            if len(name) >= 3:
                rows.append(name)

    # fallback: look for lines with common corporate suffixes
    if not rows:
        for line in soup.get_text("\n", strip=True).splitlines():
            line = line.strip()
            if 3 <= len(line) <= 120:
                if re.search(r"\b(INC|LLC|LP|LTD|CORP|CORPORATION|COMPANY|HOLDINGS)\b", norm(line)):
                    rows.append(line)

    seen, out = set(), []
    for name in rows:
        key = norm(name)
        if key and key not in seen:
            seen.add(key)
            out.append(name)
    return out


# -- Main pipeline --

def fetch_subsidiaries_from_sec(canonical_names):
    """
    For each canonical company name, fetch its SEC Exhibit 21 subsidiaries.
    Returns a dict: canonical_name -> list of raw subsidiary name strings.
    """
    sec_entries = load_sec_tickers()
    result = {}

    for i, canon in enumerate(canonical_names):
        print(f"  [{i+1}/{len(canonical_names)}] {canon}", end=" ... ", flush=True)

        sec_match = match_to_sec(canon, sec_entries)
        if not sec_match:
            print("no SEC match")
            result[canon] = []
            continue

        cik_int = int(sec_match["cik_str"])
        time.sleep(SLEEP)

        try:
            submissions = get_json(
                SUBMISSIONS_URL.format(cik=str(cik_int).zfill(10)),
                headers=SEC_HEADERS
            )
        except Exception:
            print("submissions fetch failed")
            result[canon] = []
            continue

        accession, _ = find_latest_10k(submissions)
        if not accession:
            print("no 10-K found")
            result[canon] = []
            continue

        base = archive_base(cik_int, accession)
        time.sleep(SLEEP)

        try:
            idx = get_json(base + "index.json", headers=SEC_HEADERS_WWW)
        except Exception:
            print("index fetch failed")
            result[canon] = []
            continue

        ex21_file = find_ex21(idx)
        if not ex21_file:
            print("no EX-21")
            result[canon] = []
            continue

        time.sleep(SLEEP)
        try:
            html = get_text(base + ex21_file, headers=SEC_HEADERS_WWW)
        except Exception:
            print("EX-21 fetch failed")
            result[canon] = []
            continue

        subs = parse_ex21(html)
        print(f"{len(subs)} subsidiaries")
        result[canon] = subs

    return result


def cross_reference_with_lobbyview(sec_subsidiaries):
    """
    Keep only subsidiary names that appear as client names in LobbyView.
    Returns a dict: canonical_name -> list of matched LobbyView client names.
    """
    clients = pd.read_csv(LOBBYVIEW_DIR / "lobbyview_clients.csv", usecols=["client_name"])
    lobbyview_names = set(clients["client_name"].dropna().apply(norm))

    matched = {}
    for canon, subs in sec_subsidiaries.items():
        hits = [s for s in subs if norm(s) in lobbyview_names]
        matched[canon] = hits

    total = sum(len(v) for v in matched.values())
    print(f"\nCross-reference complete: {total} subsidiaries matched to LobbyView clients.")
    return matched


def update_mapping(matched_subsidiaries):
    """Write matched subsidiaries into the JSON name mapping."""
    with open(NAME_MAPPING) as f:
        mapping = json.load(f)

    updated = 0
    for canon, subs in matched_subsidiaries.items():
        if canon in mapping and subs:
            existing = set(mapping[canon].get("subsidiaries", []))
            new = [s for s in subs if s not in existing]
            mapping[canon]["subsidiaries"] = list(existing | set(subs))
            updated += len(new)

    with open(NAME_MAPPING, "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"Updated {NAME_MAPPING.name} with {updated} new subsidiary entries.")


def main():
    print("Loading name mapping...")
    with open(NAME_MAPPING) as f:
        mapping = json.load(f)
    canonical_names = list(mapping.keys())

    print(f"\nFetching SEC Exhibit 21 data for {len(canonical_names)} companies...")
    sec_subsidiaries = fetch_subsidiaries_from_sec(canonical_names)

    print("\nCross-referencing with LobbyView clients...")
    matched = cross_reference_with_lobbyview(sec_subsidiaries)

    update_mapping(matched)


if __name__ == "__main__":
    main()
