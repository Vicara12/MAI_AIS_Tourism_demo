import streamlit as st
from streamlit import session_state as ss
from dataloader import readTourismData
from userprof import Profile
from introscreen import handleProfiles, renderHeader, renderTabs



def setup():
  ss.data = readTourismData()
  ss.city_locs = ss.data.groupby('municipality')[['lat', 'lon']].first().to_dict('index')
  ss.dest_types = set(ss.data["category"])
  ss.proc_counter = 2
  ss.profiles = {1: Profile()}
  ss.profiles_to_del = []
  ss.rerun = False
  ss.users_menu = True


def main():
  st.title("Map of Selected Locations")
  if "users_menu" not in ss:
    setup()
  # Select on wether to display input menu or recs. results
  if ss.users_menu:
    if len(ss.profiles_to_del) != 0:
      handleProfiles()
    renderHeader()
    renderTabs()
  else:
    with st.spinner("Loading recommendations"):
      results = runRecommender(ss.data, ss.profiles)
    displayResults(results)
  # Rerun if needed
  if ss.rerun:
    ss.rerun = False
    st.rerun()


main()