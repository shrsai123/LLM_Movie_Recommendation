from src.router import Intent, classify_intent, extract_search_query, extract_title


def test_preference_request_routes_to_recommendation():
    assert (
        classify_intent("Recommend funny science fiction movies from the 1990s")
        == Intent.RECOMMENDATION
    )


def test_natural_movie_request_phrases_route_to_recommendation():
    queries = [
        "Give me all movies that has a bit of comedy and romance",
        "Give me movies with a feel-good tone",
        "Movies that have comedy and romance",
        "I want movies with adventure and comedy",
    ]

    assert all(classify_intent(query) == Intent.RECOMMENDATION for query in queries)


def test_similar_movie_request_takes_priority_over_general_recommendation():
    assert (
        classify_intent("Recommend movies similar to Interstellar")
        == Intent.SIMILAR_MOVIES
    )


def test_unclassified_question_routes_to_general_movie_search():
    assert classify_intent("Explain film noir") == Intent.GENERAL


def test_general_search_query_removes_instruction_and_year():
    assert extract_search_query("Tell me about Inception (2010)?") == "Inception"


def test_plot_question_with_command_prefix_routes_to_movie_details():
    query = "/Whats the plot of Inception"

    assert classify_intent(query) == Intent.MOVIE_DETAILS
    assert extract_title(query, Intent.MOVIE_DETAILS) == "Inception"


def test_ending_question_extracts_movie_title():
    query = "What's the ending of Inception?"

    assert classify_intent(query) == Intent.MOVIE_ENDING
    assert extract_title(query, Intent.MOVIE_ENDING) == "Inception"
