from skyfield.api import load,Loader, EarthSatellite
from skyfield.timelib import Time
from skyfield.api import wgs84
from datetime import datetime, date, timedelta, timezone
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import folium
import copy
import requests
from pyproj import CRS, Transformer
from folium.plugins import TimestampedGeoJson
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import re
import PyPDF2
import io

def chargement_constellations (file="satellites_tle.txt"):
    if not file:
        url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"
        data = requests.get(url)
        with open("satellites_tle.txt", "w") as f:
            f.write(data.content.decode("utf-8").replace('\r', ''))
        tle_constellations="satellites_tle.txt"
    else:
        tle_constellations=r"satellites_tle.txt"
    
    llignes=[]
    with open(tle_constellations, "r") as file:
       for line in file:
           llignes.append(line)
    
    dfConstellations=pd.DataFrame({"sat":llignes[0::3],"TLE":["".join([l0,l1]) for l0,l1 in zip(llignes[1::3],llignes[2::3])]})
    dfConstellations.sat=dfConstellations.sat.apply(lambda s : s.replace(" ","").replace("\n",""))

    return dfConstellations

def extractPositions(numsat,nomsat,tle): 
    TLE = tle 
    L1, L2 = TLE.splitlines()
    satellite = EarthSatellite(L1, L2)
    latssat=[]
    lonssat=[]
    heightssat=[]
    for t in times:
        possat=satellite.at(t)
        latsat, lonsat = wgs84.latlon_of(possat)
        height=wgs84.height_of(possat)
        latssat.append(latsat.degrees)
        lonssat.append(lonsat.degrees)
        heightssat.append(height.km)
    df_result=pd.DataFrame({"lat":latssat, "lon":lonssat,"height":heightssat, "ts":[t for t in times]})
    df_result["nom"]=nomsat
    df_result["numsat"]=numsat

    return df_result

def dates_observation (date_debut, nbr = 1080, step = 5):

    load = Loader('~/Documents/fishing/SkyData')
    data = load('de421.bsp')
    ts   = load.timescale()
    
    tsnow = datetime.timestamp(date_debut)
    tss=np.arange(0,nbr,1)*step
    tss=tss+tsnow
    datss=[datetime.fromtimestamp(timestamp) for timestamp in list(tss)]
    times=[ts.utc(d.year, d.month, d.day,d.hour, d.minute, d.second) for d in datss]
    return times

_crs_geodetic = CRS.from_epsg(4979)   
_crs_ecef    = CRS.from_epsg(4978)   
_transformer = Transformer.from_crs(_crs_geodetic, _crs_ecef, always_xy=True)

def compute_distance_and_elevation(row, iot_lon, iot_lat, iot_alt=0):

    sat_lon = float(row["lon"])
    sat_lat = float(row["lat"])
    sat_alt = float(row["height"]) * 1e3     

    sat_x, sat_y, sat_z = _transformer.transform(sat_lon, sat_lat, sat_alt)
    iot_x, iot_y, iot_z = _transformer.transform(iot_lon, iot_lat, iot_alt)

    v = np.array([sat_x - iot_x, sat_y - iot_y, sat_z - iot_z])
    dist_m = np.linalg.norm(v)
    distance_km = dist_m / 1e3

    r_iot = np.array([iot_x, iot_y, iot_z])
    up = r_iot / np.linalg.norm(r_iot)
    vertical_comp   = np.dot(v, up)
    horizontal_comp = np.linalg.norm(v - vertical_comp * up)
    elev_rad = np.arctan2(vertical_comp, horizontal_comp)
    elevation_deg = np.degrees(elev_rad)

    return distance_km, elevation_deg

def get_closest_satellite_per_nom(df):
    min_dists = df.groupby("nom")["distance_km"].min().reset_index()
    best_nom = min_dists.loc[min_dists["distance_km"].idxmin(), "nom"]
    row = df.loc[
        (df["nom"] == best_nom) & 
        (df["distance_km"] == min_dists["distance_km"].min())
    ].iloc[0]
    return row

def find_next_overhead_pass_skyfield(dfConstellation, times,iot_lat, iot_lon, iot_alt=0,threshold_deg=85):
   
    observer = wgs84.latlon(iot_lat, iot_lon, elevation_m=iot_alt)
    dt_list  = times.utc_datetime()
    best_pass = None
    best_time = None

    for _, row in dfConstellation.iterrows():
        L1, L2 = row["TLE"].splitlines()
        sat = EarthSatellite(L1, L2, row["sat"], ts)
        diff = sat - observer
        topo = diff.at(times)
        alt, az, dist = topo.altaz()

        mask = alt.degrees >= threshold_deg
        if not np.any(mask):
            continue

        idx = np.argmax(mask)
        pass_dt = dt_list[idx]
        elev    = float(alt.degrees[idx])
        azim    = float(az.degrees[idx])
        distance_km = float(dist.km[idx])

        if best_time is None or pass_dt < best_time:
            best_time = pass_dt
            best_pass = {
                "sat":         row["sat"],
                "time_utc":    pass_dt,
                "elevation°":  elev,
                "azimuth°":    azim,
                "distance_km": distance_km
            }

    return best_pass

def find_top_n_passes(dfConstellation: pd.DataFrame,times,iot_lat: float,iot_lon: float,iot_alt: float = 0,threshold_deg: float = 0,n: int = 3) -> list[dict]:
    ts = load.timescale()
    observer = wgs84.latlon(iot_lat, iot_lon, elevation_m=iot_alt)
    dt_list = times.utc_datetime()
    all_passes = []

    for _, row in dfConstellation.iterrows():

        L1, L2 = row["TLE"].splitlines()
        sat = EarthSatellite(L1, L2, row["sat"], ts)

        diff = sat - observer
        topoc = diff.at(times)
        alt, az, dist = topoc.altaz()
        elevs = alt.degrees
        dists = dist.km

        mask = elevs >= threshold_deg
        i = 0
        while i < len(mask):
            if mask[i]:
                start = i
                while i < len(mask) and mask[i]:
                    i += 1
                end = i
                segment = elevs[start:end]
                j = start + int(np.argmax(segment))
                all_passes.append({
                    "sat":         row["sat"],
                    "time_utc":    dt_list[j],
                    "elevation°":  float(elevs[j]),
                    "distance_km": float(dists[j])
                })
            else:
                i += 1

    top_n = sorted(all_passes, key=lambda x: x["elevation°"], reverse=True)[:n]
    return top_n

dfConstellations = chargement_constellations ()
times=dates_observation(datetime.now(),1080,5)
dfConstellation=dfConstellations[dfConstellations["sat"].str.contains("KINEIS")]
df_positions=pd.concat([extractPositions(i,dfConstellation.sat.values[i],dfConstellation.TLE.values[i]) for i in range(len(dfConstellation))])


for col in ["distance_km", "elevation_deg"]:
    if col in df_positions.columns:
        df_positions.drop(columns=col, inplace=True)

ts = load.timescale()
tomorrow = date.today() + timedelta(days=1)
hours   = np.repeat(np.arange(24), 60)
minutes = np.tile(np.arange(60), 24)
times   = ts.utc(tomorrow.year, tomorrow.month, tomorrow.day,hours, minutes, 0)

def compute_data(lat, lon, alt):
    
    df_positions[["distance_km", "elevation_deg"]] = (
    df_positions
    .apply(
        lambda row: compute_distance_and_elevation(row, lon, lat, alt),
        axis=1,
        result_type="expand"
    )
    )
    
    best3 = find_top_n_passes(dfConstellation, times, lat, lon, alt,threshold_deg=0, n=3)

    for idx, p in enumerate(best3, 1):
        print(f"Passage #{idx}:")
        print(f"  Satellite   : {p['sat']}")
        print(f"  Heure UTC   : {p['time_utc']}")
        print(f"  Élévation   : {p['elevation°']:.1f}°")
        print(f"  Distance    : {p['distance_km']:.1f} km\n")
    
    
    now = datetime.now(timezone.utc)
    deltaTime = []
    for idx, entry in enumerate(best3, start=1):
        delta_s = (entry['time_utc'] - now).total_seconds()
        deltaTime.append(delta_s)

    print(pd.DataFrame(deltaTime))
    return best3, deltaTime

#recup deltaTime pour Iot 
#recup best3 pour desc Iot
#recup df_positions pour position satellite

#charger pos_Iot

app = Flask(__name__)
CORS(app)  

@app.route('/')
def index():
    return render_template('Main.html')

@app.route('/realtime')
def realtime():
    return render_template('Realtime.html')

@app.route('/documentation')
def documentation():
    return render_template('Documentation.html')

@app.route('/satellite')
def satellite():
    return render_template('Satellite.html')

@app.route('/integration')
def integration():
    return render_template('Integration.html')

@app.route('/api/passes', methods=['GET'])
def api_passes():
    lat = float(request.args.get('lat', 48.85))
    lon = float(request.args.get('lon', 2.35))
    alt = float(request.args.get('alt', 35))

    best3, deltaTime = compute_data(lat, lon, alt)

    # Garder uniquement la dernière position par satellite
    result = (
    df_positions.sort_values("ts")
    .groupby("nom")
    .tail(1)  # ou .head(1) pour la première
    .copy()
    )

    # Conversion du temps
    result['ts'] = result['ts'].apply(lambda t: t.utc_iso())
    result.rename(columns={'lon': 'lng'}, inplace=True)

    # Convertir le champ 'ts' en texte ISO
    #df_serializable = df_positions.copy()
    #df_serializable['ts'] = df_serializable['ts'].apply(lambda t: t.utc_iso())

    return jsonify({
        'df_positions': result.to_dict(orient='records'),
        'best3':        best3,
        'deltaTime':    deltaTime
    })

# ----------- Protocole page -----------
@app.route('/protocol')
def protocol():
    # Read protocol.txt
    with open('protocol.txt', encoding='utf-8') as f:
        lines = f.readlines()

    sections = []
    current_section = None
    current_subsection = None
    buffer = []
    indent_stack = []

    def flush_section():
        nonlocal buffer, current_section, current_subsection
        if current_section and buffer:
            sections.append({
                'title': current_section,
                'subsection': current_subsection,
                'content': '\n'.join(buffer)
            })
            buffer = []
            current_subsection = None

    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Detect main numbered sections (1., 2., 3., etc.)
        main_section = re.match(r'^(\d+\.\s*[A-Z][^:]+)', line)
        if main_section:
            flush_section()
            current_section = main_section.group(1)
            continue
        
        # Detect subsections with numbers (1.1, 1.2, etc.)
        subsection = re.match(r'^(\d+\.\d+\s*[A-Z][^:]+)', line)
        if subsection:
            current_subsection = subsection.group(1)
            continue
            
        # Detect subsections with colons (Types d'orbites :, etc.)
        subsection_colon = re.match(r'^([A-Z][^:]+):', line)
        if subsection_colon and not current_section:
            flush_section()
            current_section = subsection_colon.group(1)
            continue
            
        # Detect other major titles (all caps or title case)
        major_title = re.match(r'^([A-Z][A-Z\s]+)$', line)
        if major_title and len(line) > 5 and not current_section:
            flush_section()
            current_section = major_title.group(1)
            continue
            
        # Detect title case sections (Différents types de satellites)
        title_case = re.match(r'^([A-Z][a-zéèêëàâäôöùûüÿç\s]+)$', line)
        if title_case and len(line) > 15 and not current_section:
            flush_section()
            current_section = title_case.group(1)
            continue
            
        # Otherwise, add to buffer
        buffer.append(line)
    flush_section()

    # Now, for each section, convert its content to HTML with indentation
    def text_to_html(text):
        html = ''
        lines = text.split('\n')
        for l in lines:
            if not l.strip():
                continue
            indent = len(l) - len(l.lstrip(' '))
            lstr = l.lstrip()
            # Bullets
            if lstr.startswith('•') or lstr.startswith('◦'):
                html += f'<div class="ml-{indent*2} mb-1"><span class="font-semibold">•</span> {lstr[1:].strip()}</div>'
            # Sub-bullets
            elif lstr.startswith('▪'):
                html += f'<div class="ml-{indent*2+4} mb-1"><span class="font-semibold">▪</span> {lstr[1:].strip()}</div>'
            # Numbered steps
            elif re.match(r'\d+\.', lstr):
                html += f'<div class="ml-{indent*2+2} mb-1"><span class="font-semibold">{lstr.split()[0]}</span> {" ".join(lstr.split()[1:])}</div>'
            else:
                html += f'<div class="ml-{indent*2} mb-1">{lstr}</div>'
        return html

    for section in sections:
        section['content_html'] = text_to_html(section['content'])

    return render_template('Protocol.html', protocol_sections=sections)

@app.route('/pdf/<filename>')
def serve_pdf(filename):
    try:
        return send_file(f'{filename}', mimetype='application/pdf')
    except FileNotFoundError:
        return "PDF non trouvé", 404

if __name__ == '__main__':
    app.run(debug=True)
