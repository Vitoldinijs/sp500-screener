"""Generate the fallback S&P 500 snapshot CSV.

This is a *fallback only* — the live pipeline scrapes the full current list
from Wikipedia and overwrites this file. This hand-maintained subset covers
all 11 GICS sectors with the most liquid large-caps so the screener still
functions if the scrape ever fails. Run: python scripts/make_universe_snapshot.py
"""
import csv
from pathlib import Path

# (ticker, name, sector) — dots will be normalised to dashes by the loader.
ROWS = [
    # Information Technology
    ("AAPL", "Apple Inc.", "Information Technology"),
    ("MSFT", "Microsoft Corp.", "Information Technology"),
    ("NVDA", "NVIDIA Corp.", "Information Technology"),
    ("AVGO", "Broadcom Inc.", "Information Technology"),
    ("ORCL", "Oracle Corp.", "Information Technology"),
    ("CRM", "Salesforce Inc.", "Information Technology"),
    ("ADBE", "Adobe Inc.", "Information Technology"),
    ("AMD", "Advanced Micro Devices", "Information Technology"),
    ("ACN", "Accenture plc", "Information Technology"),
    ("CSCO", "Cisco Systems", "Information Technology"),
    ("INTC", "Intel Corp.", "Information Technology"),
    ("QCOM", "Qualcomm Inc.", "Information Technology"),
    ("TXN", "Texas Instruments", "Information Technology"),
    ("IBM", "IBM Corp.", "Information Technology"),
    ("NOW", "ServiceNow Inc.", "Information Technology"),
    ("INTU", "Intuit Inc.", "Information Technology"),
    ("AMAT", "Applied Materials", "Information Technology"),
    ("MU", "Micron Technology", "Information Technology"),
    ("LRCX", "Lam Research", "Information Technology"),
    ("KLAC", "KLA Corp.", "Information Technology"),
    ("ADI", "Analog Devices", "Information Technology"),
    ("PANW", "Palo Alto Networks", "Information Technology"),
    ("SNPS", "Synopsys Inc.", "Information Technology"),
    ("CDNS", "Cadence Design", "Information Technology"),
    ("APH", "Amphenol Corp.", "Information Technology"),
    # Communication Services
    ("GOOGL", "Alphabet Inc. A", "Communication Services"),
    ("GOOG", "Alphabet Inc. C", "Communication Services"),
    ("META", "Meta Platforms", "Communication Services"),
    ("NFLX", "Netflix Inc.", "Communication Services"),
    ("DIS", "Walt Disney Co.", "Communication Services"),
    ("CMCSA", "Comcast Corp.", "Communication Services"),
    ("T", "AT&T Inc.", "Communication Services"),
    ("VZ", "Verizon Communications", "Communication Services"),
    ("TMUS", "T-Mobile US", "Communication Services"),
    ("CHTR", "Charter Communications", "Communication Services"),
    ("EA", "Electronic Arts", "Communication Services"),
    ("TTWO", "Take-Two Interactive", "Communication Services"),
    # Consumer Discretionary
    ("AMZN", "Amazon.com Inc.", "Consumer Discretionary"),
    ("TSLA", "Tesla Inc.", "Consumer Discretionary"),
    ("HD", "Home Depot", "Consumer Discretionary"),
    ("MCD", "McDonald's Corp.", "Consumer Discretionary"),
    ("NKE", "Nike Inc.", "Consumer Discretionary"),
    ("LOW", "Lowe's Companies", "Consumer Discretionary"),
    ("SBUX", "Starbucks Corp.", "Consumer Discretionary"),
    ("BKNG", "Booking Holdings", "Consumer Discretionary"),
    ("TJX", "TJX Companies", "Consumer Discretionary"),
    ("ORLY", "O'Reilly Automotive", "Consumer Discretionary"),
    ("GM", "General Motors", "Consumer Discretionary"),
    ("F", "Ford Motor", "Consumer Discretionary"),
    ("MAR", "Marriott International", "Consumer Discretionary"),
    ("CMG", "Chipotle Mexican Grill", "Consumer Discretionary"),
    # Consumer Staples
    ("PG", "Procter & Gamble", "Consumer Staples"),
    ("COST", "Costco Wholesale", "Consumer Staples"),
    ("WMT", "Walmart Inc.", "Consumer Staples"),
    ("KO", "Coca-Cola Co.", "Consumer Staples"),
    ("PEP", "PepsiCo Inc.", "Consumer Staples"),
    ("PM", "Philip Morris Intl", "Consumer Staples"),
    ("MDLZ", "Mondelez International", "Consumer Staples"),
    ("MO", "Altria Group", "Consumer Staples"),
    ("CL", "Colgate-Palmolive", "Consumer Staples"),
    ("TGT", "Target Corp.", "Consumer Staples"),
    ("KMB", "Kimberly-Clark", "Consumer Staples"),
    ("GIS", "General Mills", "Consumer Staples"),
    # Health Care
    ("LLY", "Eli Lilly & Co.", "Health Care"),
    ("UNH", "UnitedHealth Group", "Health Care"),
    ("JNJ", "Johnson & Johnson", "Health Care"),
    ("MRK", "Merck & Co.", "Health Care"),
    ("ABBV", "AbbVie Inc.", "Health Care"),
    ("PFE", "Pfizer Inc.", "Health Care"),
    ("TMO", "Thermo Fisher Scientific", "Health Care"),
    ("ABT", "Abbott Laboratories", "Health Care"),
    ("DHR", "Danaher Corp.", "Health Care"),
    ("AMGN", "Amgen Inc.", "Health Care"),
    ("ISRG", "Intuitive Surgical", "Health Care"),
    ("GILD", "Gilead Sciences", "Health Care"),
    ("CVS", "CVS Health", "Health Care"),
    ("MDT", "Medtronic plc", "Health Care"),
    ("BMY", "Bristol-Myers Squibb", "Health Care"),
    ("VRTX", "Vertex Pharmaceuticals", "Health Care"),
    ("REGN", "Regeneron Pharmaceuticals", "Health Care"),
    ("ELV", "Elevance Health", "Health Care"),
    # Financials
    ("BRK-B", "Berkshire Hathaway B", "Financials"),
    ("JPM", "JPMorgan Chase", "Financials"),
    ("V", "Visa Inc.", "Financials"),
    ("MA", "Mastercard Inc.", "Financials"),
    ("BAC", "Bank of America", "Financials"),
    ("WFC", "Wells Fargo", "Financials"),
    ("GS", "Goldman Sachs", "Financials"),
    ("MS", "Morgan Stanley", "Financials"),
    ("AXP", "American Express", "Financials"),
    ("BLK", "BlackRock Inc.", "Financials"),
    ("C", "Citigroup Inc.", "Financials"),
    ("SPGI", "S&P Global", "Financials"),
    ("SCHW", "Charles Schwab", "Financials"),
    ("CB", "Chubb Ltd.", "Financials"),
    ("PGR", "Progressive Corp.", "Financials"),
    # Industrials
    ("CAT", "Caterpillar Inc.", "Industrials"),
    ("GE", "GE Aerospace", "Industrials"),
    ("RTX", "RTX Corp.", "Industrials"),
    ("HON", "Honeywell International", "Industrials"),
    ("UNP", "Union Pacific", "Industrials"),
    ("BA", "Boeing Co.", "Industrials"),
    ("DE", "Deere & Co.", "Industrials"),
    ("LMT", "Lockheed Martin", "Industrials"),
    ("UPS", "United Parcel Service", "Industrials"),
    ("ETN", "Eaton Corp.", "Industrials"),
    ("ADP", "Automatic Data Processing", "Industrials"),
    ("GD", "General Dynamics", "Industrials"),
    ("NOC", "Northrop Grumman", "Industrials"),
    ("CSX", "CSX Corp.", "Industrials"),
    ("EMR", "Emerson Electric", "Industrials"),
    # Energy
    ("XOM", "Exxon Mobil", "Energy"),
    ("CVX", "Chevron Corp.", "Energy"),
    ("COP", "ConocoPhillips", "Energy"),
    ("SLB", "Schlumberger", "Energy"),
    ("EOG", "EOG Resources", "Energy"),
    ("MPC", "Marathon Petroleum", "Energy"),
    ("PSX", "Phillips 66", "Energy"),
    ("WMB", "Williams Companies", "Energy"),
    ("OXY", "Occidental Petroleum", "Energy"),
    ("VLO", "Valero Energy", "Energy"),
    # Utilities
    ("NEE", "NextEra Energy", "Utilities"),
    ("DUK", "Duke Energy", "Utilities"),
    ("SO", "Southern Co.", "Utilities"),
    ("D", "Dominion Energy", "Utilities"),
    ("AEP", "American Electric Power", "Utilities"),
    ("EXC", "Exelon Corp.", "Utilities"),
    ("SRE", "Sempra", "Utilities"),
    ("XEL", "Xcel Energy", "Utilities"),
    # Materials
    ("LIN", "Linde plc", "Materials"),
    ("SHW", "Sherwin-Williams", "Materials"),
    ("APD", "Air Products & Chemicals", "Materials"),
    ("ECL", "Ecolab Inc.", "Materials"),
    ("FCX", "Freeport-McMoRan", "Materials"),
    ("NEM", "Newmont Corp.", "Materials"),
    ("NUE", "Nucor Corp.", "Materials"),
    ("DOW", "Dow Inc.", "Materials"),
    # Real Estate
    ("PLD", "Prologis Inc.", "Real Estate"),
    ("AMT", "American Tower", "Real Estate"),
    ("EQIX", "Equinix Inc.", "Real Estate"),
    ("WELL", "Welltower Inc.", "Real Estate"),
    ("SPG", "Simon Property Group", "Real Estate"),
    ("PSA", "Public Storage", "Real Estate"),
    ("O", "Realty Income", "Real Estate"),
    ("CCI", "Crown Castle", "Real Estate"),
]


def main():
    out = Path(__file__).resolve().parents[1] / "screener" / "data" / "sp500.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ticker", "name", "sector", "industry"])
        for t, n, s in ROWS:
            w.writerow([t.replace(".", "-"), n, s, ""])
    print(f"Wrote {len(ROWS)} tickers to {out}")


if __name__ == "__main__":
    main()
