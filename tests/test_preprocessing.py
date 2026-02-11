from src.preprocessing import get_cast, get_director, robust_parse


def test_robust_parse_json():
    result = robust_parse('[{"name": "Action"}, {"name": "Drama"}]')
    assert len(result) == 2
    assert result[0]["name"] == "Action"


def test_robust_parse_invalid():
    result = robust_parse("not valid json")
    assert result == []


def test_get_director():
    crew = (
        '[{"name": "Christopher Nolan", "job": "Director"}, {"name": "Someone", "job": "Producer"}]'
    )
    assert get_director(crew) == "Christopher Nolan"


def test_get_director_missing():
    crew = '[{"name": "Someone", "job": "Producer"}]'
    assert get_director(crew) == ""


def test_get_cast():
    cast = '[{"name": "Actor1"}, {"name": "Actor2"}, {"name": "Actor3"}]'
    result = get_cast(cast, limit=2)
    assert result == "Actor1, Actor2"
