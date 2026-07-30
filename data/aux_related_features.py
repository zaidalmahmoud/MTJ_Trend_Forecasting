import datetime
import time
import pandas as pd
import requests

# 1. ENTER YOUR ELSEVIER CREDENTIALS HERE
ELSEVIER_API_KEY = "4443119cdab9fda477ffdd67fb4fd114"
ELSEVIER_INST_TOKEN = ""  # Leave as empty string "" if running inside campus network/VPN

# Scopus advanced query syntax using explicit Boolean structures
# TITLE-ABS-KEY searches across Title, Abstract, and Keywords.
AUXILIARY_QUERIES = {
    # 1. Global Compute Crisis / Software Pull
    "Aux_GenAI_LLM_Volume": 'TITLE-ABS-KEY("Large Language Model" OR "LLM" OR "Generative AI" OR "Transformer network" OR "GPT")', 
    
    # 2. Upstream Material Breakthroughs (Excludes 'neuromorphic' to prevent multi-counting)
    "Aux_Pure_Material_Spintronics": 'TITLE-ABS-KEY("spintronics" OR "MRAM" OR "perpendicular magnetic anisotropy" OR "MTJ") AND NOT TITLE-ABS-KEY("neuromorphic")',
    
    # 3. Market Application / Hardware Constraints Demand Pull
    "Aux_Application_Demand_BCI_TinyML": 'TITLE-ABS-KEY("Brain-Computer Interface" OR "BCI" OR "TinyML" OR "wearable medical device" OR "implantable")', 
    
    # 4. Pure Mathematical / Algorithmic Foundations (Excludes 'hardware' to capture pure theoretical drivers)
    "Aux_Pure_Algorithmic_Plasticity": 'TITLE-ABS-KEY("continual learning" OR "lifelong learning" OR "vector symbolic architecture" OR "hyperdimensional computing") AND NOT TITLE-ABS-KEY("hardware")'
}

def generate_months(start_year, start_month, end_year, end_month):
    """Generates an ordered list of (year, month) tuples."""
    start_date = datetime.date(start_year, start_month, 1)
    end_date = datetime.date(end_year, end_month, 1)
    current = start_date
    months = []
    while current <= end_date:
        months.append((current.year, current.month))
        if current.month == 12:
            current = datetime.date(current.year + 1, 1, 1)
        else:
            current = datetime.date(current.year, current.month + 1, 1)
    return months

def query_scopus_count(search_phrase, year, month, retries=3):
    """Queries Elsevier Scopus API to pull metadata count for a specific month/year."""
    # FIXED: Updated to the correct Scopus Search API endpoint URL
    base_url = "https://api.elsevier.com/content/search/scopus"
    
    # Format the explicit date bounds for Scopus (e.g., PUBYEAR IS 2011)
    # To get month-level precision in Scopus, we combine the year constraint 
    # with a cover-date range parameter.
    last_day = 31 if month in [1, 3, 5, 7, 8, 10, 12] else (30 if month != 2 else 28)
    
    # FIXED: Corrected the broken syntax for the leap year check
    if month == 2 and ((year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)):
        last_day = 29
        
    # Format dates to clean YYYYMMDD strings
    start_date_str = f"{year}{month:02d}01"
    end_date_str = f"{year}{month:02d}{last_day}"
    
    # 1. Enclose your core phrase query securely in its own parentheses
    # 2. Bind it strictly to the specific publication year
    # 3. Apply the internal server load window limits cleanly
    full_query = (
        f"({search_phrase}) "
        f"AND PUBYEAR IS {year} "
        f"AND ORIG-LOAD-DATE AFT {start_date_str} "
        f"AND NOT ORIG-LOAD-DATE AFT {end_date_str}"
    )
    
    # Construct headers required by Elsevier
    headers = {
        "X-ELS-APIKey": ELSEVIER_API_KEY,
        "Accept": "application/json"
    }
    if ELSEVIER_INST_TOKEN:
        headers["X-ELS-Insttoken"] = ELSEVIER_INST_TOKEN

    params = {
        "query": full_query,
        "count": 0  # 0 items requested because we only need the total results count metadata
    }

    for attempt in range(retries):
        try:
            response = requests.get(base_url, headers=headers, params=params, timeout=20)
            
            if response.status_code == 200:
                data = response.json()
                total_results = data.get("search-results", {}).get("opensearch:totalResults", 0)
                return int(total_results)
            
            elif response.status_code == 429:
                print(f"\n[Elsevier 429] Rate limit hit. Waiting {10 * (attempt + 1)}s...")
                time.sleep(10 * (attempt + 1))
            elif response.status_code == 401:
                print("\n[Elsevier 401 Authentication Error] Check your API key or VPN connection status.")
                return 0
            else:
                print(f"\n[HTTP Error {response.status_code}] Attempt {attempt + 1} failed.")
                time.sleep(2)
                
        except Exception as e:
            print(f"\n[Network Error] Connection failed: {e}")
            time.sleep(2)
            
    return 0

def main():
    # Targets your window: Jul-2011 to Dec-2025
    timeline = generate_months(2004, 10, 2025, 12)
    data_rows = []

    print(f"Connecting to Elsevier Scopus API... Processing {len(timeline)} months.")

    for year, month in timeline:
        month_label = datetime.date(year, month, 1).strftime("%b-%Y")
        row = {"Date Month-Year": month_label}
        print(f"Scraping Scopus metrics for: {month_label} ... ", end="", flush=True)

        for column_name, query_string in AUXILIARY_QUERIES.items():
            count = query_scopus_count(query_string, year, month)
            row[column_name] = count
            # Mild polite delay (0.5s) prevents server side spike rejection
            time.sleep(0.5)
            
        print("Done.")
        data_rows.append(row)

    # Output construction
    df = pd.DataFrame(data_rows)
    output_filename = "scopus_neuromorphic_auxiliary_features.csv"
    df.to_csv(output_filename, index=False)
    print(f"\nSuccess! Peer-reviewed Scopus metrics saved to: {output_filename}")

if __name__ == "__main__":
    main()