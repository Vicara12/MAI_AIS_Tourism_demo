# MAI_AIS_Tourism_demo/src/google_photos.py
from __future__ import annotations
import os, googlemaps
from functools import lru_cache
from typing import Tuple, Optional

_API_KEY = os.getenv("GOOGLE_MAPS_KEY")
_client  = googlemaps.Client(_API_KEY) if _API_KEY else None


@lru_cache(maxsize=2_000)
def place_photo_url(name: str,
                    coords: Tuple[float, float] | None = None,
                    max_w: int = 500) -> Optional[str]:
    """
    Return a public URL of the first Google photo for *name*.
    Falls back gracefully (returns None) if nothing is found or no API key.
    """
    if not _client:
        return None

    # ---------- 1) try Nearby Search (precise & cheap) ----------
    if coords is not None:
        try:
            nearby = _client.places_nearby(
                location=coords, radius=80, keyword=name, rank_by="distance"
            )
            if nearby["results"]:
                photos = nearby["results"][0].get("photos")
                if photos:
                    ref = photos[0]["photo_reference"]
                    return (
                        "https://maps.googleapis.com/maps/api/place/photo"
                        f"?maxwidth={max_w}&photoreference={ref}&key={_API_KEY}"
                    )
        except Exception:
            pass

    # ---------- 2) text search fallback (name-only) -------------
    try:
        search = _client.places(query=name)
        if search["results"]:
            photos = search["results"][0].get("photos")
            if photos:
                ref = photos[0]["photo_reference"]
                return (
                    "https://maps.googleapis.com/maps/api/place/photo"
                    f"?maxwidth={max_w}&photoreference={ref}&key={_API_KEY}"
                )
    except Exception:
        pass

    return None
