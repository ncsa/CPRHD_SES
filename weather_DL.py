import urllib.request
import json
from datetime import datetime
from datetime import date


def download_weather_climate_division(county: str, date_to_start: str):
    base_url = "http://data.rcc-acis.org/GridData"
    start_date = datetime.strptime(date_to_start, '%Y%m%d').strftime("%Y%m%d")
    end_date = datetime.today().strftime('%Y%m%d')

    params_mint = {"state": "IL",
                     "sdate": start_date,
                     "edate": end_date,
                     "grid": 21,
                     "elems": [{"name": "mint", "area_reduce": "county_mean"}]}

    headers = {"Content-Type": 'application/json'}

    data = json.dumps(params_mint).encode('utf-8')

    try:
        req = urllib.request.Request(base_url, data, headers)
        with urllib.request.urlopen(req) as response:
            the_page = response.read()
        print(the_page.decode())
    except Exception as e:
        print(e)


download_weather_climate_division('20190101')
