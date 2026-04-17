# Missing Nodes — RBO Directed Influence GEXF

68 of 289 Fortune 500 firms are absent from the exported GEXF
(`rbo_directed_influence.gexf`). Three distinct reasons:

---

## Category 1 — Originally Isolated (35 firms)

No edges of any kind in the full directed GML. These firms had enough lobbying
activity to produce a ranked bill list, but shared no top-30 bills with any other
firm — so RBO similarity was 0 across all pairs and they were never assigned an
edge.

ADOBE · AMERICAN TOWER · BOOKING HOLDINGS · BRIGHTHOUSE FINANCIAL · BROADCOM ·
C.H. ROBINSON WORLDWIDE · CAESARS ENTERTAINMENT · CINTAS · CONAGRA BRANDS ·
ENTERPRISE PRODUCTS PARTNERS · EOG RESOURCES · ESTEE LAUDER · GOODYEAR TIRE & RUBBER ·
HALLIBURTON · HANESBRANDS · LAND O'LAKES · LEVI STRAUSS · MARATHON OIL ·
MORGAN STANLEY · MOSAIC · NCR · NEWELL BRANDS · NEWS CORP. · NRG ENERGY ·
OCCIDENTAL PETROLEUM · OLIN · OWENS & MINOR · PBF ENERGY ·
PHILIP MORRIS INTERNATIONAL · RYDER SYSTEM · STARBUCKS · TENNECO ·
VOYA FINANCIAL · WEC ENERGY GROUP · YUM CHINA HOLDINGS

---

## Category 2 — Balanced-Only Connected (24 firms)

Had edges in the original graph but exclusively through balanced pairs — i.e., they
shared top-30 bills with other firms but first-mover timing was always tied (equal
firsts and losses in every pairwise comparison). No directed edge was ever assigned.
After balanced edges were stripped, these firms had no remaining connections.

ADVANCED MICRO DEVICES · ARCONIC · ASSURANT · CELANESE · COSTCO · EMERSON ELECTRIC ·
GOLDMAN SACHS GROUP · HARLEY-DAVIDSON · HERSHEY · JETBLUE AIRWAYS · KEURIG DR PEPPER ·
KIMBERLY-CLARK · LAS VEGAS SANDS · MARSH & MCLENNAN · MGM RESORTS INTERNATIONAL ·
NAVIENT · OWENS-ILLINOIS · QURATE RETAIL · TENET HEALTHCARE · VALERO ENERGY ·
WASTE MANAGEMENT · WESCO INTERNATIONAL · WESTERN DIGITAL · XEROX

---

## Category 3 — All Directed Edges Below Median Weight (9 firms)

Had directed edges (decisive first-mover outcomes) but all with RBO similarity below
the median cut of **0.022002**. These firms had temporal precedence signals but against
counterparts with very low portfolio overlap, so their edges did not survive the weight
filter. Note that two of these were meaningful agenda-setters in the count-based sense
but with weak portfolio alignment to their comparison firms.

| Firm | net_influence | out_strength | in_strength |
|---|---|---|---|
| BAXTER INTERNATIONAL | −6 | 0.0504 | 0.0462 |
| DISH NETWORK | −1 | 0.0783 | 0.0755 |
| LUMEN TECHNOLOGIES | −1 | 0.1507 | 0.8108 |
| PUBLIX SUPER MARKETS | −1 | 0.00 | 0.6897 |
| QUALCOMM | −1 | 0.00 | 0.1136 |
| KELLOGG | 0 | 0.2606 | 0.5837 |
| IHEARTMEDIA | +5 | 0.0862 | 0.1342 |
| **ALTRIA GROUP** | **+7** | 0.0291 | 0.0010 |
| **CHARTER COMMUNICATIONS** | **+11** | 0.1071 | 0.0188 |

ALTRIA and CHARTER COMMUNICATIONS are the most notable absences — both were
decisive agenda-setters (net_influence +7 and +11 respectively) but operated against
firms with very low bill-portfolio overlap, so their RBO-weighted connections all fell
below the median threshold.
