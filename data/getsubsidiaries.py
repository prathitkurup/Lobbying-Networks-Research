import json
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import requests
import pandas as pd
from bs4 import BeautifulSoup

# ---------------------------
# CONFIG
# ---------------------------

# Put your canonical Fortune keys here (use your dict keys)
FORTUNE_CANONICAL = ['WALMART', 'EXXONMOBIL', 'APPLE', 'BERKSHIRE HATHAWAY', 'AMAZON', 'UNITED HEALTH GROUP', 'MCKESSON', 'CVS HEALTH', 'AT&T', 'AMERISOURCEBERGEN', 'FORD', 'GENERAL MOTORS', 'COSTCO', 'ALPHABET', 'CARDINAL HEALTH', 'WALGREENS BOOTS ALLIANCE', 'JPMORGAN CHASE & CO.', 'VERIZON', 'KROGER', 'GENERAL ELECTRIC', 'CHEVRON', 'FANNIE MAE', 'PHILLIPS 66', 'VALERO ENERGY', 'BANK OF AMERICA', 'MICROSOFT', 'HOME DEPOT', 'BOEING', 'WELLS FARGO', 'CITIGROUP', 'MARATHON PETROLEUM', 'COMCAST', 'ANTHEM', 'DELL TECHNOLOGIES', 'DUPONT DE NEMOURS', 'STATE FARM INSURANCE', 'JOHNSON & JOHNSON', 'IBM', 'TARGET', 'FREDDIE MAC', 'UNITED PARCEL SERVICE', "LOWE'S", 'INTEL', 'METLIFE', 'PROCTER & GAMBLE', 'UNITED TECHNOLOGIES', 'FEDEX', 'PEPSICO', 'ARCHER DANIELS MIDLAND', 'PRUDENTIAL FINANCIAL', 'CENTENE', 'ALBERTSONS', 'WALT DISNEY', 'SYSCO', 'HP', 'HUMANA', 'FACEBOOK', 'CATERPILLAR', 'ENERGY TRANSFER', 'LOCKHEED MARTIN', 'PFIZER', 'GOLDMAN SACHS GROUP', 'MORGAN STANLEY', 'CISCO SYSTEMS', 'CIGNA', 'AIG', 'HCA HEALTHCARE', 'AMERICAN AIRLINES GROUP', 'DELTA AIR LINES', 'CHARTER COMMUNICATIONS', 'NEW YORK LIFE INSURANCE', 'AMERICAN EXPRESS', 'NATIONWIDE', 'BEST BUY', 'LIBERTY MUTUAL INSURANCE GROUP', 'MERCK', 'HONEYWELL INTERNATIONAL', 'UNITED CONTINENTAL HOLDINGS', 'TIAA', 'TYSON FOODS', 'ORACLE', 'ALLSTATE', 'WORLD FUEL SERVICES', 'MASSACHUSETTS MUTUAL LIFE INSURANCE', 'TJX', 'CONOCOPHILLIPS', 'DEERE', 'TECH DATA', 'ENTERPRISE PRODUCTS PARTNERS', 'NIKE', 'PUBLIX SUPER MARKETS', 'GENERAL DYNAMICS', 'EXELON', 'PLAINS GP HOLDINGS', '3M', 'ABBVIE', 'CHS', 'CAPITAL ONE FINANCIAL', 'PROGRESSIVE', 'COCA-COLA', 'USAA', 'HEWLETT PACKARD ENTERPRISE', 'ABBOTT LABORATORIES', 'TWENTY-FIRST CENTURY FOX', 'MICRON TECHNOLOGY', 'TRAVELERS', 'RITE AID', 'NORTHROP GRUMMAN', 'ARROW ELECTRONICS', 'PHILIP MORRIS INTERNATIONAL', 'NORTHWESTERN MUTUAL', 'INTL FCSTONE', 'PBF ENERGY', 'RAYTHEON', 'KRAFT HEINZ', 'MONDELEZ INTERNATIONAL', 'U.S. BANCORP', "MACY'S", 'DOLLAR GENERAL', 'NUCOR', 'STARBUCKS', 'DXC TECHNOLOGY', 'ELI LILLY', 'THERMO FISHER SCIENTIFIC', 'US FOODS HOLDING', 'DUKE ENERGY', 'HALLIBURTON', 'CUMMINS', 'AMGEN', 'PACCAR', 'SOUTHERN', 'CENTURYLINK', 'INTERNATIONAL PAPER', 'UNION PACIFIC', 'DOLLAR TREE', 'PENSKE AUTOMOTIVE GROUP', 'QUALCOMM', 'BRISTOL-MYERS SQUIBB', 'GILEAD SCIENCES', 'JABIL', 'MANPOWERGROUP', 'SOUTHWEST AIRLINES', 'AFLAC', 'TESLA', 'AUTONATION', 'CBRE GROUP', 'LEAR', 'WHIRLPOOL', "MCDONALD'S", 'BROADCOM', 'MARRIOTT INTERNATIONAL', 'WESTERN DIGITAL', 'VISA', 'LENNAR', 'WELLCARE HEALTH PLANS', "KOHL'S", 'AECOM', 'SYNNEX', 'PNC FINANCIAL SERVICES', 'DANAHER', 'HARTFORD FINANCIAL SERVICES', 'ALTRIA GROUP', 'BANK OF NEW YORK MELLON', 'FLUOR', 'AVNET', 'ICAHN ENTERPRISES', 'OCCIDENTAL PETROLEUM', 'MOLINA HEALTHCARE', 'GENUINE PARTS', 'FREEPORT-MCMORAN', 'KIMBERLY-CLARK', 'TENET HEALTHCARE', 'SYNCHRONY FINANCIAL', 'CARMAX', 'HOLLYFRONTIER', 'PERFORMANCE FOOD GROUP', 'SHERWIN-WILLIAMS', 'EMERSON ELECTRIC', 'NGL ENERGY PARTNERS', 'XPO LOGISTICS', 'EOG RESOURCES', 'APPLIED MATERIALS', 'PG&E', 'NEXTERA ENERGY', 'C.H. ROBINSON WORLDWIDE', 'GAP', 'LINCOLN NATIONAL', 'DAVITA', 'JONES LANG LASALLE', 'WESTROCK', 'CDW', 'AMERICAN ELECTRIC POWER', 'COGNIZANT TECHNOLOGY SOLUTIONS', 'D.R. HORTON', 'BECTON DICKINSON', 'NORDSTROM', 'NETFLIX', 'ARAMARK', 'TEXAS INSTRUMENTS', 'GENERAL MILLS', 'SUPERVALU', 'COLGATE-PALMOLIVE', 'GOODYEAR TIRE & RUBBER', 'PAYPAL HOLDINGS', 'PPG INDUSTRIES', 'OMNICOM GROUP', 'CELGENE', 'JACOBS ENGINEERING GROUP', 'ROSS STORES', 'MARSH & MCLENNAN', 'MASTERCARD', "LAND O'LAKES", 'WASTE MANAGEMENT', 'ILLINOIS TOOL WORKS', 'ECOLAB', 'BOOKING HOLDINGS', 'CBS', 'PARKER-HANNIFIN', 'PRINCIPAL FINANCIAL', 'DTE ENERGY', 'BLACKROCK', 'UNITED STATES STEEL', 'COMMUNITY HEALTH SYSTEMS', 'KINDER MORGAN', 'QURATE RETAIL', 'LOEWS', 'ARCONIC', 'STANLEY BLACK & DECKER', 'TEXTRON', 'LAS VEGAS SANDS', 'ESTEE LAUDER', 'DISH NETWORK', 'STRYKER', 'KELLOGG', 'BIOGEN', 'ALCOA', 'ANADARKO PETROLEUM', 'DOMINION ENERGY', 'ADP', 'SALESFORCE.COM', 'L BRANDS', 'HENRY SCHEIN', 'NEWELL BRANDS', 'GUARDIAN LIFE INS. CO. OF AMERICA', "BJ'S WHOLESALE CLUB", 'BB&T CORP.', 'STATE STREET CORP.', 'VIACOM', 'AMERIPRISE FINANCIAL', 'CORE-MARK HOLDING', 'REINSURANCE GROUP OF AMERICA', 'VF', 'DISCOVER FINANCIAL SERVICES', 'GLOBAL PARTNERS', 'EDISON INTERNATIONAL', 'ONEOK', 'MURPHY USA', 'BED BATH & BEYOND', 'CONSOLIDATED EDISON', 'CSX', 'J.C. PENNEY', 'LKQ', 'FIRSTENERGY', 'STEEL DYNAMICS', 'LITHIA MOTORS', 'MGM RESORTS INTERNATIONAL', 'TENNECO', 'NVIDIA', 'SEMPRA ENERGY', 'FARMERS INSURANCE EXCHANGE', 'BALL', 'GROUP 1 AUTOMOTIVE', 'UNUM GROUP', 'XCEL ENERGY', 'RELIANCE STEEL & ALUMINUM', 'HUNTSMAN', 'NORFOLK SOUTHERN', 'LABORATORY CORP. OF AMERICA', 'CORNING', 'EXPEDIA GROUP', 'AUTOZONE', 'W.W. GRAINGER', 'QUANTA SERVICES', 'CROWN HOLDINGS', 'OFFICE DEPOT', 'BAXTER INTERNATIONAL', 'LAM RESEARCH', 'ENTERGY', 'CHARLES SCHWAB', 'L3 TECHNOLOGIES', 'NRG ENERGY', 'LIVE NATION ENTERTAINMENT', 'UNIVERSAL HEALTH SERVICES', 'MOLSON COORS BREWING', 'EBAY', 'AES', 'DEVON ENERGY', 'PACIFIC LIFE', 'CENTERPOINT ENERGY', 'DISCOVERY', 'BORGWARNER', 'TARGA RESOURCES', 'ALLY FINANCIAL', 'SUNTRUST BANKS', 'IQVIA HOLDINGS', 'AMERICAN FAMILY INSURANCE GROUP', 'DELEK US HOLDINGS', 'NAVISTAR INTERNATIONAL', 'CHESAPEAKE ENERGY', 'UNITED NATURAL FOODS', 'LEIDOS HOLDINGS', 'PULTEGROUP', 'EASTMAN CHEMICAL', 'REPUBLIC SERVICES', 'MOHAWK INDUSTRIES', 'SONIC AUTOMOTIVE', 'OWENS & MINOR', 'XEROX', 'BOSTON SCIENTIFIC', 'DCP MIDSTREAM', 'AUTOLIV', 'INTERPUBLIC GROUP', 'PUBLIC SERVICE ENTERPRISE GROUP', 'PVH', 'MOSAIC', 'ADVANCE AUTO PARTS', 'ALTICE USA', 'HORMEL FOODS', "O'REILLY AUTOMOTIVE", 'CALPINE', 'HERTZ GLOBAL HOLDINGS', 'FIRST DATA', 'PIONEER NATURAL RESOURCES', 'COTY', 'AGCO', 'MUTUAL OF OMAHA INSURANCE', 'VISTRA ENERGY', 'AVIS BUDGET GROUP', 'ADOBE', "PETER KIEWIT SONS'", 'NEWS CORP.', 'BRIGHTHOUSE FINANCIAL', 'VOYA FINANCIAL', 'AIR PRODUCTS & CHEMICALS', 'HILTON WORLDWIDE HOLDINGS', 'GAMESTOP', 'VERITIV', 'WILLIAMS', 'CAMPBELL SOUP', 'ROCKWELL COLLINS', 'THRIVENT FINANCIAL FOR LUTHERANS', 'WESTLAKE CHEMICAL', 'UNIVAR', 'J.B. HUNT TRANSPORT SERVICES', 'FRONTIER COMMUNICATIONS', 'JONES FINANCIAL (EDWARD JONES)', 'NATIONAL OILWELL VARCO', 'EVERSOURCE ENERGY', "DICK'S SPORTING GOODS", 'GENWORTH FINANCIAL', 'FIDELITY NATIONAL INFORMATION SERVICES', 'YUM CHINA HOLDINGS', 'RYDER SYSTEM', 'ANIXTER INTERNATIONAL', 'CAESARS ENTERTAINMENT', 'MASCO', 'THOR INDUSTRIES', 'ALASKA AIR GROUP', 'AMPHENOL', 'WESCO INTERNATIONAL', 'HUNTINGTON INGALLS INDUSTRIES', 'JEFFERIES FINANCIAL GROUP', 'DANA', 'EXPEDITORS INTL. OF WASHINGTON', 'EMCOR GROUP', 'DARDEN RESTAURANTS', 'SPARTANNASH', 'ASSURANT', 'UNITED RENTALS', 'LIBERTY MEDIA', 'ERIE INSURANCE GROUP', 'AUTO-OWNERS INSURANCE', 'CHENIERE ENERGY', 'FIFTH THIRD BANCORP', 'FOOT LOCKER', 'CONAGRA BRANDS', 'ZIMMER BIOMET HOLDINGS', 'TRACTOR SUPPLY', 'BERRY GLOBAL GROUP', 'ALLIANCE DATA SYSTEMS', 'HERSHEY', 'PPL', 'DEAN FOODS', 'BUILDERS FIRSTSOURCE', 'OSHKOSH', 'ENLINK MIDSTREAM', 'W.R. BERKLEY', 'WEC ENERGY GROUP', 'JETBLUE AIRWAYS', 'UGI', 'A-MARK PRECIOUS METALS', 'FIDELITY NATIONAL FINANCIAL', 'CONSTELLATION BRANDS', 'QUEST DIAGNOSTICS', 'ACTIVISION BLIZZARD', 'WEYERHAEUSER', 'RAYMOND JAMES FINANCIAL', "CASEY'S GENERAL STORES", 'KEURIG DR PEPPER', 'AMERICAN TOWER', 'APACHE', 'DOVER', 'KEYCORP', 'J.M. SMUCKER', 'CITIZENS FINANCIAL GROUP', 'MOTOROLA SOLUTIONS', 'MAGELLAN HEALTH', 'AMERICAN AXLE & MANUFACTURING', 'NEWMONT GOLDCORP', 'SPIRIT AEROSYSTEMS HOLDINGS', 'WESTERN & SOUTHERN FINANCIAL GROUP', 'FORTIVE', 'GRAYBAR ELECTRIC', 'NVR', 'AVERY DENNISON', 'CELANESE', 'AMERICAN FINANCIAL GROUP', 'TOLL BROTHERS', 'SANMINA', 'INSIGHT ENTERPRISES', 'OWENS CORNING', 'PACKAGING CORP. OF AMERICA', 'TRAVELCENTERS OF AMERICA', 'OLIN', 'ARTHUR J. GALLAGHER', 'MASTEC', 'ALLEGHANY', 'OWENS-ILLINOIS', 'ASBURY AUTOMOTIVE GROUP', 'CMS ENERGY', 'MARKEL', 'BLACKSTONE GROUP', 'AK STEEL HOLDING', 'HANESBRANDS', 'R.R. DONNELLEY & SONS', 'WAYFAIR', 'REGIONS FINANCIAL', 'WYNN RESORTS', 'ULTA BEAUTY', 'REGENERON PHARMACEUTICALS', 'BURLINGTON STORES', 'ROCKWELL AUTOMATION', 'NORTHERN TRUST', 'CHEMOURS', 'SEABOARD', 'MARATHON OIL', 'ASCENA RETAIL GROUP', "DILLARD'S", 'CINTAS', 'ADVANCED MICRO DEVICES', 'HESS', 'M&T BANK CORP.', 'ABM INDUSTRIES', 'BEACON ROOFING SUPPLY', 'NCR', 'IHEARTMEDIA', 'FRANKLIN RESOURCES', 'AMEREN', 'INTERCONTINENTAL EXCHANGE', 'S&P GLOBAL', 'POST HOLDINGS', 'ANALOG DEVICES', 'RALPH LAUREN', 'L3HARRIS TECHNOLOGIES', 'BOOZ ALLEN HAMILTON', 'POLARIS INDUSTRIES', 'CLOROX', 'REALOGY HOLDINGS', 'HD SUPPLY HOLDINGS', 'GRAPHIC PACKAGING HOLDING', 'OLD REPUBLIC INTERNATIONAL', 'INTUIT', 'NETAPP', 'TAPESTRY', 'ON SEMICONDUCTOR', 'INGREDION', 'ZOETIS', 'FISERV', 'TREEHOUSE FOODS', 'ROBERT HALF INTERNATIONAL', 'FIRST AMERICAN FINANCIAL', 'HARLEY-DAVIDSON', 'WINDSTREAM HOLDINGS', 'YUM BRANDS', 'WILLIAMS-SONOMA', 'SIMON PROPERTY GROUP', 'NAVIENT', 'WESTERN UNION', 'PEABODY ENERGY', 'LEVI STRAUSS']

# IMPORTANT: SEC requires a User-Agent identifying you (email recommended)
SEC_HEADERS = {
    "User-Agent": "Victoria Figueroa (vfigueroa@bowdoin.edu) research script",
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}

SEC_HEADERS_WWW = {
    "User-Agent": "Victoria Figueroa (vfigueroa@bowdoin.edu) research script",
    "Accept-Encoding": "gzip, deflate",
}

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"

# be polite
SLEEP_SECONDS = 0.2


# ---------------------------
# HELPERS
# ---------------------------

def norm(s: str) -> str:
    s = s.upper()
    s = re.sub(r"[^A-Z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def jaccard(a: str, b: str) -> float:
    A = set(norm(a).split())
    B = set(norm(b).split())
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def get_json(url: str, headers: dict) -> dict:
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()

def get_text(url: str, headers: dict) -> str:
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return r.text

def cik10(cik: int) -> str:
    return str(cik).zfill(10)

def accession_no_dashes(acc: str) -> str:
    return acc.replace("-", "")

def build_archive_base(cik_int: int, accession: str) -> str:
    # Example:
    # https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dashes}/
    return f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_no_dashes(accession)}/"

def find_latest_10k(submissions: dict) -> Optional[Tuple[str, str]]:
    """
    Returns (accessionNumber, filingDate) for the most recent 10-K/10-KA in the submissions 'recent' list.
    """
    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    dates = recent.get("filingDate", [])

    for form, acc, dt in zip(forms, accessions, dates):
        if form in ("10-K", "10-K/A"):
            return acc, dt
    return None

def fetch_filing_index(cik_int: int, accession: str) -> Optional[str]:
    """
    Gets filing index page (index.json) that lists files in that accession folder.
    """
    base = build_archive_base(cik_int, accession)
    index_url = base + "index.json"
    try:
        return get_text(index_url, headers=SEC_HEADERS_WWW)
    except Exception:
        return None

def parse_index_json(index_text: str) -> Optional[dict]:
    try:
        return json.loads(index_text)
    except Exception:
        return None

def pick_ex21_doc(index_json: dict) -> Optional[str]:
    """
    Try to select the EX-21 / Exhibit 21 file from the accession folder listing.
    Returns filename if found.
    """
    items = index_json.get("directory", {}).get("item", [])
    # heuristic: prefer names containing ex21 or exhibit21 or ex-21 and html/htm
    candidates = []
    for it in items:
        name = it.get("name", "")
        n = name.lower()
        if any(k in n for k in ["ex21", "ex-21", "exhibit21", "exh21", "exv21"]):
            if n.endswith((".htm", ".html", ".txt")):
                candidates.append(name)

    if candidates:
        # prefer html-ish
        candidates.sort(key=lambda x: (not x.lower().endswith((".htm", ".html")), len(x)))
        return candidates[0]

    # fallback: sometimes exhibits are in full submission text; not handled here
    return None

def extract_subsidiaries_from_html(html: str) -> List[Tuple[str, Optional[str]]]:
    """
    Returns list of (subsidiary_name, jurisdiction_optional).
    Attempts tables first, then fallbacks to line parsing.
    """
    soup = BeautifulSoup(html, "html.parser")
    rows = []

    # 1) table heuristic
    tables = soup.find_all("table")
    for table in tables:
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            cells = [c for c in cells if c]
            if not cells:
                continue
            # common pattern: [Company Name, Jurisdiction] or [Name]
            if len(cells) >= 2:
                name = cells[0]
                juris = cells[1]
                # skip obvious headers
                if "company name" in name.lower() and "domicile" in juris.lower():
                    continue
                if len(name) >= 2:
                    rows.append((name, juris))
            elif len(cells) == 1:
                name = cells[0]
                if len(name) >= 2:
                    rows.append((name, None))

    # 2) fallback: list items / paragraphs that look like entity names
    if not rows:
        text = soup.get_text("\n", strip=True)
        for line in text.splitlines():
            line = line.strip()
            if 3 <= len(line) <= 120:
                # very light filter: includes Corp/Inc/LLC/etc often
                if re.search(r"\b(INC|LLC|L\.P\.|LP|LTD|CORP|CORPORATION|COMPANY|HOLDINGS)\b", norm(line)):
                    rows.append((line, None))

    # de-dupe while preserving order
    seen = set()
    out = []
    for name, juris in rows:
        key = norm(name)
        if key and key not in seen:
            seen.add(key)
            out.append((name, juris))
    return out


# ---------------------------
# MAIN PIPELINE
# ---------------------------

def load_ticker_map() -> List[dict]:
    # SEC hosts this mapping file.  [oai_citation:2‡sec.gov](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data?utm_source=chatgpt.com)
    data = get_json(TICKERS_URL, headers=SEC_HEADERS_WWW)
    # format is dict of { "0": {...}, "1": {...} } etc.
    return list(data.values())

def match_company_to_sec_entry(canon: str, sec_entries: List[dict]) -> Optional[dict]:
    """
    Fuzzy match: best Jaccard between canonical and SEC "title"
    """
    best = None
    best_score = 0.0
    for e in sec_entries:
        title = e.get("title", "")
        sc = jaccard(canon, title)
        if sc > best_score:
            best_score = sc
            best = e
    # require at least some overlap
    if best_score < 0.25:
        return None
    return best

def run():
    sec_entries = load_ticker_map()

    records = []

    for canon in FORTUNE_CANONICAL:
        sec_match = match_company_to_sec_entry(canon, sec_entries)
        if not sec_match:
            records.append({
                "company": canon,
                "matched_sec_company_name": None,
                "ticker": None,
                "cik": None,
                "filing_date": None,
                "subsidiary_name": None,
                "subsidiary_jurisdiction": None,
                "source_url": None,
                "note": "No SEC company match found from company_tickers.json"
            })
            continue

        cik_int = int(sec_match["cik_str"])
        ticker = sec_match.get("ticker")
        title = sec_match.get("title")

        # submissions JSON
        sub_url = SUBMISSIONS_URL.format(cik10=cik10(cik_int))
        time.sleep(SLEEP_SECONDS)
        submissions = get_json(sub_url, headers=SEC_HEADERS)

        latest = find_latest_10k(submissions)
        if not latest:
            records.append({
                "company": canon,
                "matched_sec_company_name": title,
                "ticker": ticker,
                "cik": cik_int,
                "filing_date": None,
                "subsidiary_name": None,
                "subsidiary_jurisdiction": None,
                "source_url": None,
                "note": "No recent 10-K found in submissions recent list"
            })
            continue

        accession, filing_date = latest

        # list files in accession folder
        time.sleep(SLEEP_SECONDS)
        idx_text = fetch_filing_index(cik_int, accession)
        if not idx_text:
            records.append({
                "company": canon,
                "matched_sec_company_name": title,
                "ticker": ticker,
                "cik": cik_int,
                "filing_date": filing_date,
                "subsidiary_name": None,
                "subsidiary_jurisdiction": None,
                "source_url": None,
                "note": "Could not fetch index.json for accession folder"
            })
            continue

        idx_json = parse_index_json(idx_text)
        if not idx_json:
            records.append({
                "company": canon,
                "matched_sec_company_name": title,
                "ticker": ticker,
                "cik": cik_int,
                "filing_date": filing_date,
                "subsidiary_name": None,
                "subsidiary_jurisdiction": None,
                "source_url": None,
                "note": "index.json not parseable"
            })
            continue

        ex21 = pick_ex21_doc(idx_json)
        if not ex21:
            records.append({
                "company": canon,
                "matched_sec_company_name": title,
                "ticker": ticker,
                "cik": cik_int,
                "filing_date": filing_date,
                "subsidiary_name": None,
                "subsidiary_jurisdiction": None,
                "source_url": None,
                "note": "No EX-21 / Exhibit 21 doc found in accession folder"
            })
            continue

        base = build_archive_base(cik_int, accession)
        ex21_url = base + ex21

        # fetch exhibit
        time.sleep(SLEEP_SECONDS)
        html = get_text(ex21_url, headers=SEC_HEADERS_WWW)

        subs = extract_subsidiaries_from_html(html)

        if not subs:
            records.append({
                "company": canon,
                "matched_sec_company_name": title,
                "ticker": ticker,
                "cik": cik_int,
                "filing_date": filing_date,
                "subsidiary_name": None,
                "subsidiary_jurisdiction": None,
                "source_url": ex21_url,
                "note": "EX-21 found but no subsidiaries parsed"
            })
            continue

        for name, juris in subs:
            records.append({
                "company": canon,
                "matched_sec_company_name": title,
                "ticker": ticker,
                "cik": cik_int,
                "filing_date": filing_date,
                "subsidiary_name": name,
                "subsidiary_jurisdiction": juris,
                "source_url": ex21_url,
                "note": None
            })

    df = pd.DataFrame(records)
    df.to_csv("fortune500_subsidiaries_ex21.csv", index=False)
    print("Wrote: fortune500_subsidiaries_ex21.csv")
    print(df.head(20))

if __name__ == "__main__":
    run()