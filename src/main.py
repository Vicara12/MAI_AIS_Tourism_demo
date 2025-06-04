from typing import Union, Tuple
import streamlit as st
from streamlit import session_state as ss
import pandas as pd
import pydeck as pdk
from geopy.distance import geodesic
from dataloader import readTourismData

def setup():
  ss.data = readTourismData()
  ss.tabs = {'Profile 1':}
  ss.setup = True

def renderMap(coords: pd.DataFrame,
              user_loc: Union[None, Tuple[float,float]]=None,
              radius: Union[None, float]=None):
  coords['distance_km'] = coords.apply(
    lambda row: geodesic(user_loc, (row['lat'], row['lon'])).km,
    axis=1
  )
  # User has not entered loc
  if user_loc is None or radius is None:
    coords['color'] = coords.apply(lambda row: [0x2F, 0xE8, 0x8d, 200], axis=1)
  else:
    coords['color'] = coords['distance_km'].apply(
      lambda d: [0x2F, 0xE8, 0x8d, 200] if d <= radius else [0xFF, 0x84, 0x7C, 100]
    )
    user_df = pd.DataFrame([{
      'name': 'You',
      'lat': user_loc[0],
      'lon': user_loc[1],
      'color': [0x45, 0xAD, 0xFF],
      'distance_km': 0
    }])
    user_layer = pdk.Layer(
      'ScatterplotLayer',
      data=user_df,
      get_position='[lon, lat]',
      get_color='color',
      get_radius='size',
      pickable=False,
      radiusMinPixels=8,
      radiusMaxPixels=30
    )
    circle_layer = pdk.Layer(
      "ScatterplotLayer",
      data=user_df,
      get_position='[lon, lat]',
      get_color='[0, 100, 255, 15]',  # Soft blue fill
      get_radius=radius * 1000,    # Radius in meters
      pickable=False
    )
  locations_layer = pdk.Layer(
    'ScatterplotLayer',
    data=coords,
    get_position='[lon, lat]',
    get_color='color',
    radiusMinPixels=5,
    radiusMaxPixels=20,
    pickable=True
  )
  if user_loc is None or radius is None:
    layers=[locations_layer]
    init_view = pdk.ViewState(
      latitude=41.4076,
      longitude=2.1744,
      zoom=4,
      pitch=0
    )
  else:
    layers=[circle_layer, locations_layer, user_layer]
    init_view = pdk.ViewState(
      latitude=user_loc[0],
      longitude=user_loc[1],
      zoom=4,
      pitch=0
    )
  st.pydeck_chart(pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v9",
    initial_view_state=init_view,
    layers=layers
  ))



st.title("Map of Selected Locations")
# st.map(coords)

if "setup" not in ss:
  # User data needed:
  # - mobility
  # - location
  # - max disp
  # - pref culture, nature, nightlife, local_imact, co2
  # - Type prefs
  setup()
  renderMap(pd.DataFrame(ss.data, columns=["lat","lon"]), user_loc=(41.4076,2.1744), radius=None)