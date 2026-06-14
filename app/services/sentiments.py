from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


def analyze_sentiment(text: str) -> str:
    """
    Classify text sentiment using VADER.

    Returns 'positive', 'negative', or 'neutral' based on the compound score.
    VADER is optimized for short social text and uses casing and punctuation
    as sentiment signals, so raw unprocessed text should be passed in.
    """
    compound = _analyzer.polarity_scores(text)["compound"]
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    return "neutral"


def run_sentiment_analysis(reviews: list[dict], preprocessed_texts: list[str]) -> list[dict]:
    """
    Annotate each review with a sentiment label.

    Accepts preprocessed texts as a separate argument to keep preprocessing
    and analysis concerns decoupled. Returns a new list of review dicts
    with a 'sentiment' key added to each.
    """
    return [
        {**review, "sentiment": analyze_sentiment(text)}
        for review, text in zip(reviews, preprocessed_texts)
    ]