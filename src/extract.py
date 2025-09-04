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
from os.path import join, relpath
from config import DATA_IMAGES, DATA_PROCESSED, BASE_DIR


def create_main_csv(file_path: str) -> None:

    with open(file_path, 'r') as f:
        data = json.load(f)

    observations = [feat['properties'] for feat in data['features']]

    DIRECTIONS = ['N', 'E', 'S', 'W', 'U']
    PHOTO_KEYS = [
        "skyconditionsNorthPhotoUrl",
        "skyconditionsEastPhotoUrl",
        "skyconditionsSouthPhotoUrl",
        "skyconditionsWestPhotoUrl",
        "skyconditionsUpwardPhotoUrl"
    ]
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

    # create rows with feature columns for all jpg photos
    rows = []
    for obs in observations:
        if not (has_jpg_photo(obs)):
            continue
        
        # grab relevant observation-level features
        one_hot = {ck[13:].lower(): int(obs[ck] == 'true') for ck in CLOUD_KEYS}
        obs_id = obs['skyconditionsObservationId']
        user_id = int(obs['skyconditionsUserid'])

        # add rows for valid jpg photos
        for pk, direction in zip(PHOTO_KEYS, DIRECTIONS):
            photo_url = obs.get(pk)
            if is_jpg(photo_url):
                row = {'photo_url': photo_url, 
                       'direction': direction, 
                       'obs_id': obs_id, 
                       'user_id': user_id}
                row.update(one_hot)
                rows.append(row)

    # create DataFrame and a small sample
    df = pd.DataFrame(rows)
    subset = df.head(1000).copy()
    image_paths = []

    # download photo urls to local storage
    for _, row in tqdm(subset.iterrows(), total=len(subset)):
        url = row['photo_url']
        obs_id = row['obs_id']
        direction = row['direction']
        filename = f'{obs_id}_{direction}.jpg'
        filepath = join(DATA_IMAGES, filename)
        relative_path = relpath(filepath, start=BASE_DIR)

        # avoid re-downloading photos
        if isfile(filepath):
            image_paths.append(relative_path)
            continue

        # handle exceptions when downloading photos
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                image_paths.append(relative_path)
            else:
                image_paths.append(None)
        except Exception as e:
            image_paths.append(None)
    
    # save DataFrame as csv
    subset['local_path'] = image_paths
    subset.to_csv(f'{DATA_PROCESSED}/main.csv')

if __name__ == '__main__':
    # path = download_globe_sky_data()
    path = '/content/drive/MyDrive/clouds-ml/data/raw/sky_conditions_20250801_20250901.json'
    create_main_csv(path)