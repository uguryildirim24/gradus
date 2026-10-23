import pytest

from matcher.metros import ANYWHERE_US, METROS, metro_of, parse_place, places_match


@pytest.mark.parametrize("a,b", [
    ("Newton, MA", "Boston, MA"), ("Cambridge, MA", "Boston"), ("Palo Alto, CA", "San Jose, CA"),
    ("San Francisco Bay Area", "Mountain View, CA"), ("Jersey City, NJ", "NYC"), ("Bethesda, MD", "Washington, D.C."),
    ("Texas", "Austin, TX"), ("Austin, TX", "Texas"), ("United States", "Boise, ID"), (ANYWHERE_US, "Toledo, OH"),
    ("Cambridge, MA, USA", "Boston, MA"), ("Portland, ME", "Portland, ME"),
])
def test_matches(a, b):
    assert places_match(a, b) and places_match(b, a)


@pytest.mark.parametrize("a,b", [
    ("Boston, MA", "New York, NY"), ("Pasadena, TX", "Pasadena, CA"), ("Portland, ME", "Portland, OR"),
    ("Texas", "Boston, MA"), (ANYWHERE_US, "London, UK"), ("Arlington, VA", "Arlington, TX"),
])
def test_mismatches(a, b):
    assert not places_match(a, b)


def test_metro_table():
    assert len(METROS) >= 40
    assert metro_of("Newton, MA") == metro_of("Waltham, MA") == "boston"
    assert metro_of("South San Francisco, CA") == metro_of("Sunnyvale, CA") == "bay_area"
    assert parse_place("Massachusetts").kind == "state" and parse_place("Washington, DC").metro == "dc"
    assert metro_of("Madison") is None  # Madison, WI and Madison, AL: ambiguous without a state


@pytest.mark.parametrize("state,inside,outside", [
    ("Washington State", ("Seattle, WA", "Bellevue, WA", "Olympia, WA"), ("Washington, DC", "Silver Spring, MD",
                                                                          "Reston, VA")),
    ("New York State", ("Brooklyn, NY", "New York, NY", "Albany, NY"), ("Jersey City, NJ", "Stamford, CT")),
])
def test_city_named_states_are_written_with_state(state, inside, outside):
    assert parse_place(state).kind == "state"
    for x in inside:
        assert places_match(state, x) and places_match(x, state)
    for x in outside:
        assert not places_match(state, x) and not places_match(x, state)


def test_bare_new_york_and_washington_are_the_cities():
    assert parse_place("New York").metro == "nyc" and parse_place("Washington").metro == "dc"
    assert places_match("New York", "Jersey City, NJ") and places_match("Washington", "Reston, VA")
    assert not places_match("Washington", "Seattle, WA")


def test_sampled_state_only_places_are_unambiguous():
    import random

    from matcher.metros import STATE_NAMES, cities_in, state_place
    from matcher.sample import state_name

    assert state_name("Bellevue, WA") == "Washington State" and state_name("Brooklyn, NY") == "New York State"
    assert state_name("Austin, TX") == "Texas"
    for code in STATE_NAMES:
        text = state_place(code)
        assert parse_place(text) == parse_place(text) and parse_place(text).state == code
        homes = cities_in(code)
        assert homes and all(places_match(h, text) and parse_place(h).kind == "city" for h in homes)
    rng = random.Random(0)
    from matcher.sample import _locations

    for _ in range(3000):
        for loc in _locations(rng, "onsite"):
            p = parse_place(loc)
            assert p.kind != "state" or loc == state_place(p.state)


def test_bare_city_state_in_a_list_of_states_is_the_state():
    from dataclasses import replace

    from matcher import full_match, label
    from matcher.metros import posting_places
    from matcher.schema import RequirementCard

    states = ("Oregon", "Washington", "Idaho", "New York")
    assert posting_places(states) == ("Oregon", "Washington State", "Idaho", "New York State")
    assert posting_places(("Washington", "Boston, MA")) == ("Washington", "Boston, MA")  # no state named: DC
    card = RequirementCard(source_posting_id="p", title="Analyst", locations=states)
    seattle = replace(full_match(card), current_location="Seattle, WA", willing_locations=())
    b, reasons, _ = label(seattle, card)
    assert b == "For you" and "location works (Washington)" in reasons
    dc = replace(seattle, current_location="Washington, DC")
    assert label(dc, card)[0] == "Excluded"
    assert label(dc, replace(card, locations=("Washington", "Boston, MA")))[0] == "For you"
