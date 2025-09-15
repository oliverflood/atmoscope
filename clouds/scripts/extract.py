from datetime import datetime
from os.path import isfile
from urllib.request import urlopen
from shutil import copyfileobj
from contextlib import closing
from config import DATA_RAW

def download_globe_sky_data() -> str:
    start_date = datetime(2025,8,1)
    end_date = datetime(2025,9,1)

    protocol = "sky_conditions"

    dates = f'{start_date.strftime("%Y%m%d")}_{end_date.strftime("%Y%m%d")}'
    download_dest = f'{DATA_RAW}/{protocol}_{dates}.json'

    if isfile(download_dest):
        print("--  Download will not be attempted as the file already exists locally.")
        return download_dest

    url = (
        "https://api.globe.gov/search/v1/measurement/protocol/measureddate/"
        f"?protocols={protocol}&startdate={start_date:%Y-%m-%d}&enddate={end_date:%Y-%m-%d}"
        "&geojson=TRUE&sample=FALSE"
    )

    try:
        print("--  Downloading from API...")
        print(f"--  {url}")

        with closing(urlopen(url)) as r:
            with open(download_dest, 'wb') as f:
                copyfileobj(r, f)
        
        print("--  Download successful. Saved to:")
        print(f"--  {download_dest}")

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

    rows = []
    for obs in observations:
        if not (has_jpg_photo(obs)):
            continue
        
        one_hot = {ck[13:].lower(): int(obs[ck] == 'true') for ck in CLOUD_KEYS}
        obs_id = obs['skyconditionsObservationId']
        user_id = int(obs['skyconditionsUserid'])
        globe_trained = 1 if obs['skyconditionsIsGlobeTrained'] == 'true' else 0
        citizen_science = 1 if obs['skyconditionsIsCitizenScience'] == 'true' else 0

        for pk, direction in zip(PHOTO_KEYS, DIRECTIONS):
            photo_url = obs.get(pk)
            if is_jpg(photo_url):
                row = {'photo_url': photo_url, 
                       'direction': direction, 
                       'obs_id': obs_id, 
                       'user_id': user_id,
                       'globe_trained': globe_trained,
                       'citizen_science': citizen_science}
                row.update(one_hot)
                rows.append(row)

    df = pd.DataFrame(rows)

    # TEMP: just pull whole df no images
    subset = df
    image_paths = []

    for _, row in tqdm(subset.iterrows(), total=len(subset)):
        url = row['photo_url']
        obs_id = row['obs_id']
        direction = row['direction']
        filename = f'{obs_id}_{direction}.jpg'
        filepath = join(DATA_IMAGES, filename)
        relative_path = relpath(filepath, start=BASE_DIR)
        
        image_paths.append(None)
    
    subset['local_path'] = image_paths
    subset.to_csv(f'{DATA_PROCESSED}/three_years.csv')

if __name__ == '__main__':
    # path = download_globe_sky_data()
    path = '/content/drive/MyDrive/clouds-ml/data/raw/sky_conditions_20220901_20250901.json'
    create_main_csv(path)