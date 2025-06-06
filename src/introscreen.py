from typing import Union, Tuple, List
import streamlit as st
from streamlit import session_state as ss
import pandas as pd
import pydeck as pdk
from geopy.distance import geodesic
from userprof import Profile


def checkValidProfiles():
  valid = True
  for n,p in ss.profiles.items():
    # Check name
    if ss[f'name_field_{n}'] == '':
      st.write(f" - ERROR: User {n} has no name")
      valid = False
    else:
      ss.profiles[n].name = ss[f'name_field_{n}']
  return valid


def renderHeader():
  if st.button("Generate recommendation") and checkValidProfiles():
    ss.users_menu = False
    ss.rerun = True
  if st.button("Add new traveler"):
    ss.profiles[ss.proc_counter] = Profile()
    ss.proc_counter += 1


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
      latitude=41.6076,
      longitude=1.8044,
      zoom=6.8,
      pitch=0
    )
  else:
    layers=[circle_layer, locations_layer, user_layer]
    init_view = pdk.ViewState(
      latitude=user_loc[0],
      longitude=user_loc[1],
      zoom=6.8,
      pitch=0
    )
  st.pydeck_chart(pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v9",
    initial_view_state=init_view,
    layers=layers
  ))

def setRerun():
  ss.rerun = True

def renderTabs():
  tabs = st.tabs([(p.name if p.name is not None else f"User {n}") for n,p in ss.profiles.items()])
  for n, tab in zip(ss.profiles.keys(), tabs):
    with tab:
      # First row: Name and Delete user
      cols_row1 = st.columns(3)
      with cols_row1[0]:
        name_inp = st.text_input("Traveler name:", on_change=setRerun, key=f"name_field_{n}")
        if name_inp != "":
          ss.profiles[n].name = name_inp
      with cols_row1[-1]:
        if st.button("Remove User", key=f"remove_user_{n}"):
          ss.profiles_to_del.append(n)
          st.rerun()
      # Mobility constraints
      ss.profiles[n].mobility_constr = st.checkbox("This user has mobility constraints", key=f"mob_constr_{n}")
      # Types of activities selector
      ss.profiles[n].avoid = st.multiselect("Which of these would you rather avoid?", options=ss.dest_types, key=f"city_ms_{n}")
      # Preferences slider
      # def changePrefs(name):
      #   sum_all = ss[f"cult_{n}"]+ss[f"nat_{n}"]+ss[f"nl_{n}"]+ss[f"li_{n}"]+ss[f"co2_{n}"]
      #   sum_others = sum_all - ss[name]
      st.text("Select your preferences:")
      cols_slider = st.columns(5)
      with cols_slider[0]:
        ss.profiles[n].culture = st.slider("Cultural activities", 0, 100, 50, key=f"cult_{n}")/100
      with cols_slider[1]:
        ss.profiles[n].nature = st.slider("Activities in nature", 0, 100, 50, key=f"nat_{n}")/100
      with cols_slider[2]:
        ss.profiles[n].nlife = st.slider("Night life activities", 0, 100, 50, key=f"nl_{n}")/100
      with cols_slider[3]:
        ss.profiles[n].local_imp = st.slider("Minimize local impact", 0, 100, 50, key=f"li_{n}")/100
      with cols_slider[4]:
        ss.profiles[n].co2 = st.slider("Minimize CO2", 0, 100, 50, key=f"co2_{n}")/100
      # Location selector
      st.text("Select coordinates or city")
      cols_loc = st.columns(4)
      with cols_loc[0]:
        lat = st.number_input("Latitude", key=f"latitude_{n}")
      with cols_loc[1]:
        lon = st.number_input("Longitude", key=f"longitude_{n}")
      with cols_loc[2]:
        def changeLocNumberInps():
          location = ss[f"sel_{n}"]
          lat, lon = ss.city_locs[location]["lat"], ss.city_locs[location]["lon"]
          ss[f"latitude_{n}"] = lat
          ss[f"longitude_{n}"] = lon
        st.selectbox("Get coords. of a city", list(ss.city_locs), key=f"sel_{n}", on_change=changeLocNumberInps)
      ss.profiles[n].location = (lat, lon)
      # Maximum disp.
      cols_dist = st.columns(4)
      with cols_dist[0]:
        dist = st.number_input("Maximum distance", key=f"disp_{n}")
      if dist != 0:
        ss.profiles[n].max_disp = dist
      # Map
      renderMap(pd.DataFrame(ss.data, columns=["lat","lon"]),
                user_loc=ss.profiles[n].location,
                radius=ss.profiles[n].max_disp)


def handleProfiles():
  for n in ss.profiles_to_del:
      del ss.profiles[n]
  ss.profiles_to_del = []
  if len(ss.profiles.items()) == 0:
    ss.profiles = {1: Profile()}
  st.rerun()
