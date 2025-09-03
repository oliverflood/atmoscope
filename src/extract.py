from datetime import datetime
from os.path import isfile
from urllib.request import urlopen
from shutil import copyfileobj
from contextlib import closing
from config import DATA_RAW

def download_globe_sky_data() -> str:
    start_date = datetime(2025,8,1)
    end_date = datetime(2025,9,1)

    # GLOBE protocol for cloud data is sky conditions
    protocol = "sky_conditions"

    # Where to save the downloaded file
    dates = f'{start_date.strftime("%Y%m%d")}_{end_date.strftime("%Y%m%d")}'
    download_dest = f'{DATA_RAW}/{protocol}_{dates}.json'

    # Check if file already exists at the destination
    if isfile(download_dest):
        print("--  Download will not be attempted as the file already exists locally.")
        return download_dest

    # Create the full download link 
    url = (
        "https://api.globe.gov/search/v1/measurement/protocol/measureddate/"
        f"?protocols={protocol}&startdate={start_date:%Y-%m-%d}&enddate={end_date:%Y-%m-%d}"
        "&geojson=TRUE&sample=FALSE"
    )

    # Download from the API
    try:
        print("--  Downloading from API...")
        print(f"--  {url}")

        # Open the target URL, open the local file, and copy
        with closing(urlopen(url)) as r:
            with open(download_dest, 'wb') as f:
                copyfileobj(r, f)
        
        print("--  Download successful. Saved to:")
        print(f"--  {download_dest}")

    # In the event of a failure, print the error
    except Exception as e:
        print("(x) Download failed:")
        print(e)

    return download_dest

import json
import pandas as pd
from typing import Optional
from tqdm import tqdm
import requests
from os.path import join
from config import DATA_IMAGES
from config import DATA_PROCESSED


def create_main_csv(file_path: str) -> None:

    with open(file_path, 'r') as f:
        data = json.load(f)

    observations = [feat['properties'] for feat in data['features']]

    PHOTO_KEYS = [
        "skyconditionsNorthPhotoUrl",
        "skyconditionsEastPhotoUrl",
        "skyconditionsSouthPhotoUrl",
        "skyconditionsWestPhotoUrl",
        "skyconditionsUpwardPhotoUrl"
    ]

    DIRECTIONS = ['N', 'E', 'S', 'W', 'U']

    CLOUD_KEYS = [
        "skyconditionsAltocumulus",
        "skyconditionsAltostratus",
        "skyconditionsCirrocumulus",
        "skyconditionsCirrostratus",
        "skyconditionsCirrus",
        "skyconditionsCumulonimbus",
        "skyconditionsCumulus",
        "skyconditionsNimbostratus",
        "skyconditionsStratocumulus",
        "skyconditionsStratus"
    ]

    def is_jpg(url: Optional[str]) -> bool:
        return url != None and url[-4:] == '.jpg'
        
    def has_jpg_photo(obs) -> bool:
        return any(is_jpg(obs.get(pk)) for pk in PHOTO_KEYS)

    rows = []
    for obs in observations:
        if not (has_jpg_photo(obs)):
            continue

        one_hot = {ck[13:].lower(): int(obs[ck] == 'true') for ck in CLOUD_KEYS}
        obs_id = obs['skyconditionsObservationId']
        for pk, direction in zip(PHOTO_KEYS, DIRECTIONS):
            photo_url = obs.get(pk)
            if photo_url[-3:] != 'jpg':
                continue
            if photo_url and photo_url != 'null':
                row = {'photo_url': photo_url, 'direction': direction, 'obs_id': obs_id}
                row.update(one_hot)
                rows.append(row)

    df = pd.DataFrame(rows)
    subset = df.head(30).copy()
    image_paths = []
    for _, row in tqdm(subset.iterrows(), total=len(subset)):
        url = row['photo_url']
        obs_id = row['obs_id']
        direction = row['direction']
        filename = f'{obs_id}_{direction}.jpg'
        filepath = join(DATA_IMAGES, filename)
        if isfile(filepath):
            image_paths.append(filepath)
            continue
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                image_paths.append(filepath)
            else:
                image_paths.append(None)
        except Exception as e:
            image_paths.append(None)
    
    subset['local_path'] = image_paths
    subset.to_csv(f'{DATA_PROCESSED}/main.csv')

if __name__ == '__main__':
    path = download_globe_sky_data()
    create_main_csv(path)