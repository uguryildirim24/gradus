"""US metro areas and place matching for the location rule.

A place is text as it appears on a card: "Cambridge, MA", "South San Francisco, CA",
"Massachusetts" (state only: anywhere in that state), "United States" or
"anywhere in the US" (the whole country). Two state names are also big cities: bare
"New York" and "Washington" mean New York City and Washington, DC, as real postings use
them; the whole state is written "New York State" / "Washington State" (`state_place`), and in a
list that names other whole states ("Oregon, Washington, Idaho") the bare name is the state
too (`posting_places`). Two places match when they share a metro
(Newton and Boston, Palo Alto and San Jose), when one is a state that holds the other,
or when one is the whole country and the other is in the US. Places outside these
metros still match on the exact city ("Toronto, ON" with "Toronto").
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .schema import norm

ANYWHERE_US = "anywhere in the US"

# metro id -> (display name, sampling weight, suburbs and nearby cities as "City, ST")
METROS: dict[str, tuple[str, int, tuple[str, ...]]] = {
    "nyc": ("New York, NY", 10, (
        "Brooklyn, NY", "Queens, NY", "Bronx, NY", "Long Island City, NY", "Jersey City, NJ", "Hoboken, NJ",
        "Newark, NJ", "Yonkers, NY", "White Plains, NY", "Stamford, CT")),
    "la": ("Los Angeles, CA", 8, (
        "Santa Monica, CA", "Pasadena, CA", "Burbank, CA", "Long Beach, CA", "Irvine, CA", "Glendale, CA",
        "Culver City, CA", "Torrance, CA", "El Segundo, CA", "Anaheim, CA")),
    "chicago": ("Chicago, IL", 7, ("Evanston, IL", "Naperville, IL", "Schaumburg, IL", "Oak Brook, IL", "Skokie, IL",
                                   "Deerfield, IL")),
    "houston": ("Houston, TX", 6, ("Sugar Land, TX", "The Woodlands, TX", "Katy, TX", "Pasadena, TX", "Pearland, TX")),
    "dallas": ("Dallas, TX", 6, ("Fort Worth, TX", "Plano, TX", "Irving, TX", "Arlington, TX", "Frisco, TX",
                                 "Richardson, TX")),
    "atlanta": ("Atlanta, GA", 5, ("Marietta, GA", "Alpharetta, GA", "Decatur, GA", "Sandy Springs, GA")),
    "boston": ("Boston, MA", 6, (
        "Cambridge, MA", "Newton, MA", "Waltham, MA", "Somerville, MA", "Burlington, MA", "Wilmington, MA",
        "Lexington, MA", "Woburn, MA", "Quincy, MA", "Brookline, MA", "Framingham, MA", "Needham, MA", "Bedford, MA",
        "Lowell, MA")),
    "bay_area": ("San Francisco, CA", 6, (
        "South San Francisco, CA", "Oakland, CA", "Berkeley, CA", "Emeryville, CA", "San Mateo, CA", "Foster City, CA",
        "Redwood City, CA", "Menlo Park, CA", "Palo Alto, CA", "Mountain View, CA", "Sunnyvale, CA", "Santa Clara, CA",
        "San Jose, CA", "Cupertino, CA", "Milpitas, CA", "Fremont, CA", "San Bruno, CA")),
    "seattle": ("Seattle, WA", 5, ("Bellevue, WA", "Redmond, WA", "Kirkland, WA", "Tacoma, WA", "Everett, WA",
                                   "Renton, WA")),
    "dc": ("Washington, DC", 6, (
        "Arlington, VA", "Alexandria, VA", "Bethesda, MD", "Silver Spring, MD", "Reston, VA", "Tysons, VA",
        "McLean, VA", "Rockville, MD", "Herndon, VA", "Fairfax, VA")),
    "philadelphia": ("Philadelphia, PA", 4, ("King of Prussia, PA", "Conshohocken, PA", "Camden, NJ",
                                             "Wilmington, DE")),
    "miami": ("Miami, FL", 4, ("Fort Lauderdale, FL", "Miami Beach, FL", "Coral Gables, FL", "Boca Raton, FL",
                               "Doral, FL")),
    "phoenix": ("Phoenix, AZ", 4, ("Scottsdale, AZ", "Tempe, AZ", "Mesa, AZ", "Chandler, AZ", "Glendale, AZ")),
    "denver": ("Denver, CO", 4, ("Boulder, CO", "Aurora, CO", "Lakewood, CO", "Englewood, CO", "Broomfield, CO")),
    "minneapolis": ("Minneapolis, MN", 3, ("St. Paul, MN", "Bloomington, MN", "Eden Prairie, MN", "Plymouth, MN")),
    "san_diego": ("San Diego, CA", 3, ("La Jolla, CA", "Carlsbad, CA", "Chula Vista, CA", "Escondido, CA")),
    "austin": ("Austin, TX", 4, ("Round Rock, TX", "Cedar Park, TX", "Georgetown, TX")),
    "charlotte": ("Charlotte, NC", 3, ("Concord, NC", "Gastonia, NC", "Rock Hill, SC")),
    "nashville": ("Nashville, TN", 2, ("Franklin, TN", "Brentwood, TN", "Murfreesboro, TN")),
    "pittsburgh": ("Pittsburgh, PA", 2, ("Cranberry Township, PA", "Monroeville, PA")),
    "raleigh": ("Raleigh, NC", 3, ("Durham, NC", "Cary, NC", "Chapel Hill, NC", "Morrisville, NC",
                                   "Research Triangle Park, NC")),
    "salt_lake": ("Salt Lake City, UT", 2, ("Lehi, UT", "Draper, UT", "Sandy, UT", "Provo, UT")),
    "detroit": ("Detroit, MI", 2, ("Dearborn, MI", "Troy, MI", "Southfield, MI", "Auburn Hills, MI", "Ann Arbor, MI")),
    "st_louis": ("St. Louis, MO", 2, ("Clayton, MO", "Chesterfield, MO", "St. Charles, MO")),
    "baltimore": ("Baltimore, MD", 2, ("Columbia, MD", "Towson, MD", "Annapolis, MD")),
    "portland": ("Portland, OR", 2, ("Beaverton, OR", "Hillsboro, OR", "Lake Oswego, OR", "Vancouver, WA")),
    "columbus": ("Columbus, OH", 2, ("Dublin, OH", "Westerville, OH")),
    "indianapolis": ("Indianapolis, IN", 2, ("Carmel, IN", "Fishers, IN")),
    "kansas_city": ("Kansas City, MO", 2, ("Overland Park, KS", "Kansas City, KS", "Lenexa, KS")),
    "tampa": ("Tampa, FL", 2, ("St. Petersburg, FL", "Clearwater, FL")),
    "orlando": ("Orlando, FL", 2, ("Kissimmee, FL", "Lake Mary, FL")),
    "san_antonio": ("San Antonio, TX", 2, ("New Braunfels, TX",)),
    "cleveland": ("Cleveland, OH", 1, ("Beachwood, OH", "Lakewood, OH")),
    "cincinnati": ("Cincinnati, OH", 1, ("Mason, OH", "Covington, KY")),
    "milwaukee": ("Milwaukee, WI", 1, ("Waukesha, WI",)),
    "madison": ("Madison, WI", 1, ("Middleton, WI", "Verona, WI")),
    "sacramento": ("Sacramento, CA", 1, ("Roseville, CA", "Folsom, CA", "Davis, CA")),
    "las_vegas": ("Las Vegas, NV", 1, ("Henderson, NV",)),
    "albuquerque": ("Albuquerque, NM", 1, ("Rio Rancho, NM",)),
    "hartford": ("Hartford, CT", 1, ("West Hartford, CT",)),
    "richmond": ("Richmond, VA", 1, ("Glen Allen, VA",)),
    "new_orleans": ("New Orleans, LA", 1, ("Metairie, LA",)),
    "huntsville": ("Huntsville, AL", 1, ("Madison, AL",)),
    "boise": ("Boise, ID", 1, ("Meridian, ID",)),
}

METRO_IDS = tuple(METROS)
METRO_CENTERS = tuple(v[0] for v in METROS.values())
METRO_WEIGHTS = tuple(v[1] for v in METROS.values())

# City aliases, keyed by the normalized city part of "City, ST".
CITY_ALIASES = {
    "nyc": "new york", "new york city": "new york", "manhattan": "new york",
    "sf": "san francisco", "san fran": "san francisco",
    "dc": "washington", "washington dc": "washington", "washington d c": "washington",
    "district of columbia": "washington",
    "la": "los angeles", "philly": "philadelphia",
    "saint louis": "st louis",
    "saint paul": "st paul", "vegas": "las vegas",
    "saint petersburg": "st petersburg", "saint charles": "st charles",
}
# Names that mean a whole metro rather than one city.
METRO_ALIASES = {
    "san francisco bay area": "bay_area", "bay area": "bay_area", "sf bay area": "bay_area",
    "silicon valley": "bay_area", "greater boston": "boston", "boston area": "boston",
    "greater new york": "nyc", "new york metro": "nyc", "greater seattle": "seattle",
    "dc metro": "dc", "dc metro area": "dc", "washington dc metro area": "dc", "research triangle": "raleigh",
    "dallas fort worth": "dallas", "dfw": "dallas", "greater los angeles": "la", "greater chicago": "chicago",
}
STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv", "new hampshire": "nh",
    "new jersey": "nj", "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn",
    "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy", "d c": "dc",
    "district of columbia": "dc",
}
STATE_NAMES = {v: k.title() for k, v in STATES.items() if v != "dc"}
# State names that alone read as a city ("New York", "Washington"): written with "State" when they
# mean the whole state.
CITY_STATES = frozenset({"new york", "washington"})
_US_ABBR = set(STATES.values())
_COUNTRY_US = {"united states", "usa", "us", "u s", "u s a", "united states of america", "anywhere in the us",
               "anywhere in the united states", "nationwide", "us nationwide"}
_COUNTRY_TAIL = {"usa", "us", "u s", "united states", "united states of america"}


def place_key(text: str) -> tuple[str, str]:
    """("city", "st") with aliases applied; st is "" when the text has no state."""
    parts = [" ".join(p.split()) for p in norm(text.replace(".", " ")).split(",")]
    parts = [p for p in parts if p]
    while len(parts) > 1 and parts[-1] in _COUNTRY_TAIL:
        parts.pop()
    city = parts[0] if parts else ""
    state = parts[1] if len(parts) > 1 else ""
    city = CITY_ALIASES.get(city, city)
    state = STATES.get(state, state)
    return city, state


# More places one commute from a metro, named by the real postings (the real-posting test and the demo
# pool: SpaceX in Hawthorne, Ford in Warren, GE Aerospace in Evendale). The rule matches them like the
# suburbs above; the samplers never draw them (suburbs() and all_places() read METROS only), so the
# synthetic cards and the pair manifest do not change.
NEARBY: dict[str, tuple[str, ...]] = {
    "nyc": ("Florham Park, NJ",),
    "la": ("Hawthorne, CA", "Costa Mesa, CA", "Fullerton, CA"),
    "chicago": ("Rosemont, IL", "Des Plaines, IL"),
    "austin": ("Bastrop, TX",),
    "boston": ("Lynn, MA",),
    "bay_area": ("Walnut Creek, CA",),
    "dc": ("Arlington County, VA", "Clarksburg, MD"),
    "miami": ("Plantation, FL",),
    "denver": ("Westminster, CO",),
    "minneapolis": ("Dayton, MN", "Eagan, MN"),
    "detroit": ("Warren, MI",),
    "orlando": ("Cape Canaveral, FL",),
    "cincinnati": ("Evendale, OH",),
    "milwaukee": ("New Berlin, WI",),
}


def _index() -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for mid, (center, _, subs) in METROS.items():
        for place in (center, *subs, *NEARBY.get(mid, ())):
            out[place_key(place)] = mid
    return out


_METRO_OF = _index()


@dataclass(frozen=True)
class Place:
    """kind: "city", "state", "country" or "metro". state is a two-letter code or ""."""

    kind: str
    city: str = ""
    state: str = ""
    metro: str | None = None

    @property
    def in_us(self) -> bool:
        return self.kind in ("country", "metro") or self.state in _US_ABBR or self.metro is not None


def state_place(code: str) -> str:
    """How a card writes "anywhere in this state": "California", but "New York State"."""
    name = STATE_NAMES[code]
    return f"{name} State" if name.lower() in CITY_STATES else name


_STATE_WORDS = re.compile(r"^(?:anywhere in |anywhere within )?(?:the )?(?:state of )?(.+?)(?: state)?$")


def _whole_state(flat: str) -> str | None:
    """Two-letter code when the text names a whole state: "California", "Washington State",
    "State of New York", "anywhere in Texas". Bare "New York" / "Washington" are cities."""
    m = _STATE_WORDS.match(flat)
    name = m.group(1) if m else flat
    code = STATES.get(name)
    if code not in STATE_NAMES:
        return None
    explicit = name != flat
    return code if explicit or name not in CITY_STATES else None


def parse_place(text: str) -> Place:
    t = " ".join(norm(text.replace(".", " ")).split())
    if t in _COUNTRY_US:
        return Place("country")
    flat = " ".join(t.replace(",", " ").split())
    if flat in METRO_ALIASES:
        return Place("metro", metro=METRO_ALIASES[flat])
    city, state = place_key(text)
    code = _whole_state(flat) or (_whole_state(city) if not state else None)
    if code:
        return Place("state", state=code)
    if not state and city in STATE_NAMES and len(city) == 2:
        return Place("state", state=city)
    if city in METRO_ALIASES:
        return Place("metro", metro=METRO_ALIASES[city])
    metro = _METRO_OF.get((city, state))
    if metro is None and not state:
        hits = {m for (c, _), m in _METRO_OF.items() if c == city}
        metro = hits.pop() if len(hits) == 1 else None
    return Place("city", city=city, state=state, metro=metro)


def metro_of(text: str) -> str | None:
    """Metro id of a place, or None (a state, the country, or a city outside the table)."""
    return parse_place(text).metro


def _state_of(p: Place) -> str:
    if p.state:
        return p.state
    if p.metro:
        return place_key(METROS[p.metro][0])[1]
    return ""


def places_match(a: str, b: str) -> bool:
    """Can someone in (or willing to move to) place a work at place b, or the other way round."""
    pa, pb = parse_place(a), parse_place(b)
    if pa.kind == "country" or pb.kind == "country":
        other = pb if pa.kind == "country" else pa
        return other.in_us
    if pa.metro and pb.metro:
        return pa.metro == pb.metro
    if pa.kind == "state" or pb.kind == "state":
        return bool(_state_of(pa)) and _state_of(pa) == _state_of(pb)
    if pa.kind == "metro" or pb.kind == "metro":
        return False  # a named metro against a city outside the table
    return pa.city == pb.city and (not pa.state or not pb.state or pa.state == pb.state)


def posting_places(locations: tuple[str, ...]) -> tuple[str, ...]:
    """A posting's places as the rule reads them: in a list that also names a whole state, a bare
    "Washington" or "New York" is that state ("Washington State"), not the city."""
    if not any(parse_place(x).kind == "state" for x in locations):
        return locations
    out = []
    for x in locations:
        flat = " ".join(norm(x.replace(".", " ")).split())
        out.append(state_place(STATES[flat]) if flat in CITY_STATES else x)
    return tuple(out)


def join_places(places: tuple[str, ...]) -> str:
    """"Boston or Denver"; "Portland, OR, or Columbus, OH" when a place has a state."""
    return (", or " if any("," in x for x in places) else " or ").join(places)


# A city to live in for states without a metro in the table (the capital).
STATE_CAPITALS = {
    "al": "Montgomery, AL", "ak": "Juneau, AK", "az": "Phoenix, AZ", "ar": "Little Rock, AR", "ca": "Sacramento, CA",
    "co": "Denver, CO", "ct": "Hartford, CT", "de": "Dover, DE", "fl": "Tallahassee, FL", "ga": "Atlanta, GA",
    "hi": "Honolulu, HI", "id": "Boise, ID", "il": "Springfield, IL", "in": "Indianapolis, IN", "ia": "Des Moines, IA",
    "ks": "Topeka, KS", "ky": "Frankfort, KY", "la": "Baton Rouge, LA", "me": "Augusta, ME", "md": "Annapolis, MD",
    "ma": "Boston, MA", "mi": "Lansing, MI", "mn": "St. Paul, MN", "ms": "Jackson, MS", "mo": "Jefferson City, MO",
    "mt": "Helena, MT", "ne": "Lincoln, NE", "nv": "Carson City, NV", "nh": "Concord, NH", "nj": "Trenton, NJ",
    "nm": "Santa Fe, NM", "ny": "Albany, NY", "nc": "Raleigh, NC", "nd": "Bismarck, ND", "oh": "Columbus, OH",
    "ok": "Oklahoma City, OK", "or": "Salem, OR", "pa": "Harrisburg, PA", "ri": "Providence, RI", "sc": "Columbia, SC",
    "sd": "Pierre, SD", "tn": "Nashville, TN", "tx": "Austin, TX", "ut": "Salt Lake City, UT", "vt": "Montpelier, VT",
    "va": "Richmond, VA", "wa": "Olympia, WA", "wv": "Charleston, WV", "wi": "Madison, WI", "wy": "Cheyenne, WY",
}


def cities_in(state: str) -> tuple[str, ...]:
    """Places to live inside a state: the table's metro places there, else its capital."""
    found = tuple(x for m in METROS for x in all_places(m) if place_key(x)[1] == state)
    return found or ((STATE_CAPITALS[state],) if state in STATE_CAPITALS else ())


def suburbs(metro: str) -> tuple[str, ...]:
    return METROS[metro][2]


def all_places(metro: str) -> tuple[str, ...]:
    return (METROS[metro][0], *METROS[metro][2])
