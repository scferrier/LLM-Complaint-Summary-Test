"""
#TODOs 

1. Set-up Scrapper to Pull a List of All Secuities Litgation Cases from CourtListener
2. Check GovInfo for Opinions on Motions to Dismiss - if so pull that opinion
3. Check CourtListener for Compliant that was at-issue in the Motion to Dismiss
4. Check CourtListener for Briefing on the Motion to Dismiss

"""
import requests
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

CL_TOKEN = os.getenv("COURT_LISTNER")
url = "https://www.courtlistener.com/api/rest/v4/dockets/"
headers = {
    "Authorization": f"Token {CL_TOKEN}"
    }

params = {
    "nature_of_suit": "850 Securities/Commodities", 
}

r = requests.get(url, headers=headers, params=params)
def fetch_all_results(url, headers, params, stop_date="2020-01-01"):
    all_results = []

    # First request with params
    r = requests.get(url, headers=headers, params=params)
    if r.status_code != 200:
        print(f"Error: Status {r.status_code}")
        return all_results, 0

    data = r.json()
    total_count = data.get('count', 0)
    all_results.extend(data.get('results', []))

    page = 1
    print(f"Fetched page {page}... ({len(all_results)} results so far)")

    # Follow 'next' until None
    next_url = data.get('next')
    while next_url is not None:
        page += 1
        r = requests.get(next_url, headers=headers)
        if r.status_code != 200:
            print(f"Error on page {page}: Status {r.status_code}")
            break

        data = r.json()
        results = data.get('results', [])
        all_results.extend(results)

        for d in results:
            #stop date format needs to be YYYY-MM-DD
            if d.get("date_filed") == stop_date:
                print("Matched date — stopping early.")
                return all_results, total_count

        print(f"Fetched page {page}... ({len(all_results)} results so far)")
        next_url = data.get('next')


def results_to_dataframe(results):
    #Convert results to a pandas DataFrame.
    if not results:
        return pd.DataFrame()

    rows = []
    for result in results:
        rows.append({
            'Case Name': result.get('case_name', 'No name'),
            'Filed': result.get('date_filed', 'N/A'),
            'Court': result.get('court_id', 'N/A'),
            'Docket No.': result.get('docket_number', 'N/A'),
            'Link': f"www.courtlistener.com{result.get('absolute_url', '')}"
        })

    return pd.DataFrame(rows)


case_no = "1_22-cv-01138"
court_id = 'ded'
oppinion_check = f'https://www.govinfo.gov/wssearch/getContentDetail?packageId=USCOURTS-{court_id}-{case_no}'

