#------------------------------------------------------------------------------
# PURPOSE
# Retrieve and download all GLOBE observations that have reported dust during
# a specified period of time. Plot a timeseries of the observations and plot
# observation locations on a world map.
#
# INSTRUCTIONS
# The default start date is 01 July 2019. The default end date is today's date.
# To request a different date range, modify the start and end dates on lines
# 84 and 85.
#
# INSTALLATIONS
# Users need to install the Anaconda3 python distribution, https://www.anaconda.com/
#
# RESOURCES
# - GLOBE website: https://www.globe.gov/
# - GLOBE Data User Guide: https://www.globe.gov/globe-data/globe-data-user-guide
# - How to download GLOBE dust data: https://www.globe.gov/web/marile.colonrobles/home/blog/-/blogs/how-to-download-dust-observations-reported-through-globe
# - Download the GLOBE Observer app: https://observer.globe.gov/about/get-the-app
#
# MODIFICATION HISTORY
# 01 Aug 2019 - MJ Starke - v1.0 written
# 16 Aug 2019 - HM Amos   - Revised so the code stands alone and can be posted
#                           as a resource on the GLOBE website.
#
# This code comes as is and without any guarantees.
#------------------------------------------------------------------------------

# Import packages
from datetime import datetime, timedelta
from contextlib import closing
from os.path import isfile
from shutil import copyfileobj
from tqdm import tqdm
from urllib.request import urlopen
from cartopy.feature import NaturalEarthFeature
import matplotlib.pyplot as plt
import numpy as np
import json
import cartopy.crs as ccrs

class Observation:
    def __init__(self, feature: dict = None):
        self._raw = feature["properties"]
        self._raw["Latitude"] = feature["geometry"]["coordinates"][1]
        self._raw["Longitude"] = feature["geometry"]["coordinates"][0]
        
    def __getitem__(self, item: str):
        """
        Returns the value of key 'item'. None if item is not in dictionary.
        """
        try:
            return self._raw[item]
        except KeyError:
            return None
        
    @property
    def measured_dt(self):
        """
        :return: The measurement datetime of this observation, or none if the date and/or time are recorded incorrectly.
        Raises flag DX if the datetime is missing, and DI if the datetime is invalid or malformed.
        """
        # Find or construct the string representing the datetime.

        # sic: "Measurement" may be misspelled in the file.
        dtstring = self["skyconditionsMeasuredAt"]

        # Attempt to convert that string to a datetime.
        try:
            return datetime.strptime(dtstring, "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            # If that format string fails, try add in microseconds
            try:
                return datetime.strptime(dtstring, "%Y-%m-%dT%H:%M:%S.%f")
            except ValueError:
                return None

def main():
    #------------------------------------------------------------------------------
    # Datetime information
    #------------------------------------------------------------------------------
    
    # Set desired start and end dates
    graph_start_date     = datetime(2024,7,1)      # start date (yyyy,mm,dd)
    graph_end_date       = datetime.today().date() # end date, default is today's date
    
    # Convert to datetime objects
    graph_start_datetime = datetime.combine(graph_start_date, datetime.min.time())
    graph_end_datetime   = datetime.combine(graph_end_date, datetime.max.time())
    
    #------------------------------------------------------------------------------
    # Download and parse GLOBE data
    #
    # Source: GLOBE API, https://www.globe.gov/globe-data/globe-api
    #------------------------------------------------------------------------------
    
    # Download observations
    
    # Tell API the GLOBE protocol associated with the data you want
    # Note: dust data are part of sky conditions data
    protocols = ["sky_conditions"]
    
    # Create a string that represents the protocol part of the API query
    protocol_string = "".join(["protocols={}&".format(protocol) for protocol in protocols])
    
    # Create the full download link.
    download_src = "https://api.globe.gov/search/v1/measurement/protocol/measureddate/?{}startdate={}&enddate={" \
                   "}&geojson=TRUE&sample=FALSE "
    download_src = download_src.format(protocol_string, graph_start_date.strftime("%Y-%m-%d"), graph_end_date.strftime("%Y-%m-%d"))
    
    # Where to save the downloaded file
    download_dest = "%P_%S_%E.json"
     
    # Replace % indicators in the destination string with their respective variable values
    # %P will be replaced with protocol name(s), %S with the start date, and %E
    # with the end date
    download_dest = download_dest.replace("%P", "__".join(protocols))
    download_dest = download_dest.replace("%S", graph_start_date.strftime("%Y%m%d"))
    download_dest = download_dest.replace("%E", graph_end_date.strftime("%Y%m%d"))
    
    # Check if file already exists at the destination.  If a file by the target
    # name already exists, skip download.
    check_existing = True    # Default is True; change to False if you want to
                             # download data directly from the API every time
    if check_existing:
        if isfile(download_dest):
            print("--  Download will not be attempted as the file already exists locally.")
    
    # Download from the API
    try:
        print("--  Downloading from API...")
        print("--  {}".format(download_src))
        # Open the target URL, open the local file, and copy.
        with closing(urlopen(download_src)) as r:
            with open(download_dest, 'wb') as f:
                copyfileobj(r, f)
        print("--  Download successful.  Saved to:")
        print("--  {}".format(download_dest))
    # In the event of a failure, print the error.
    except Exception as e:
        print("(x) Download failed:")
        print(e)
    
    # Read data from JSON file
    print("--  Reading JSON from {}...".format(download_dest))
    try:
        with open(download_dest, "r") as f:
            g = f.read()
            print("--  Interpreting file as JSON...")
            raw = json.loads(g)
    except UnicodeDecodeError:
        with open(download_dest, "r", encoding="utf8") as f:
            g = f.read()
            print("--  Interpreting file as JSON...")
            raw = json.loads(g)
                
    # Parse JSON file and returns its features converted to observations
    # Display a progress bar         
    obs = []
    for o in tqdm(range(len(raw["features"])), desc="Parsing JSON as observations"):
        ob = raw["features"][o]
        obs.append(Observation(feature=ob))
    
    # Filter to observations reporting dust
    obs = [ob for ob in obs if ob["skyconditionsDust"] is not None and ob["skyconditionsDust"]=="true"]
    
    # Create a list of all dates between start date and end date
    dates = []
    d     = graph_start_datetime
    while d <= graph_end_datetime:
        dates.append(d)
        d += timedelta(days=1)
    
    #------------------------------------------------------------------------------
    # Figures
    #------------------------------------------------------------------------------
    
    #------------------------------------------------------------------------------
    # Plot timeseries of number of dust observations per day
    #------------------------------------------------------------------------------
    
    # Bins values by day
    ts = np.histogram([ob.measured_dt for ob in obs], dates)[0]
    
    # Create figure
    fig = plt.figure(figsize=(10, 3.5))
    ax  = fig.add_subplot(111)
    ax.bar(dates[:-1], ts)
    ax.set_xlim(graph_start_datetime - timedelta(days=0.5), graph_end_datetime + timedelta(days=0.5))
    ax.set_xlabel("Date (UTC)")
    ax.set_ylabel("Observations per day")
    ax.set_title("GLOBE observations reporting dust")
    plt.tight_layout()
    plt.show()
    
    #------------------------------------------------------------------------------
    # Plot dust observation locations on a world map
    #------------------------------------------------------------------------------
    
    # Create a figure and axis with Plate Carree projection
    fig = plt.figure(figsize=(10,6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(color="#444444")  # coastline color
    
    # Add and color background
    ax.add_feature(NaturalEarthFeature("physical", "land", "110m", facecolor="#999999", zorder=-1))
    ax.add_feature(NaturalEarthFeature("physical", "ocean", "110m", facecolor="#98B6E2", zorder=-1))
    
    # Set axes limits
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    
    # Overlay observation locations on the map as red dots
    x = []
    y = []
    for ob in tqdm(obs, desc="Preparing observations"):
        if ob["Latitude"] is not None and ob["Longitude"] is not None:
            x.append(ob["Longitude"])
            y.append(ob["Latitude"])
    
    ax.scatter(x, y, c="red", s=40)
    ax.set_title("Locations of GLOBE observations reporting dust",
                 fontdict={"fontsize": 18})
    plt.tight_layout()
    plt.show()
    return

#------------------------------------------------------------------------------
# Call main
#------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
 
