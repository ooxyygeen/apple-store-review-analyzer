from collections import Counter


def extract_keywords(negative_reviews: list[dict], preprocessor, top_n: int = 10) -> list[str]:
    """
    Extract the most frequent keywords from negative reviews.

    Accepts a preprocessor function as an argument to keep this module
    decoupled from the preprocessing implementation. Flattens all tokens
    across negative reviews into a single pool and returns the top_n
    most frequent terms after stopword removal and lemmatization.

    Returns an empty list if there are no negative reviews.
    """
    if not negative_reviews:
        return []

    all_tokens = [
        token
        for review in negative_reviews
        for token in preprocessor(review)
    ]

    return [word for word, _ in Counter(all_tokens).most_common(top_n)]