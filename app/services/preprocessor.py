import re

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

_lemmatizer = WordNetLemmatizer()

CUSTOM_STOPWORDS = {
    "app", "get", "got", "dont", "really", "keep", "even", "one", "use",
    "used", "using", "would", "could", "also", "make", "made", "going",
    "give", "given", "want", "wanted", "need", "needed", "thing", "things",
    "time", "good", "great", "bad", "say", "said", "way", "back", "well",
    "still", "never", "always", "every", "much", "many", "lot", "know",
    "think", "thought", "come", "came", "see", "look", "feel", "felt",
}

_stopwords = set(stopwords.words("english")) | CUSTOM_STOPWORDS


def _get_wordnet_pos(tag: str) -> str:
    """Map a Penn Treebank POS tag to a WordNet POS tag for lemmatization."""
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def preprocess_for_sentiment(review: dict) -> str:
    """
    Prepare review text for VADER sentiment analysis.

    Preserves original casing and punctuation since VADER uses them
    as sentiment signals. Only strips URLs and extra whitespace.
    Truncates to 500 words to stay within model limits.
    """
    text = f"{review.get('title', '')} {review.get('body', '')}".strip()
    text = re.sub(r"https?://\S+", "", text)
    words = text.split()
    if len(words) > 500:
        text = " ".join(words[:500])
    return text


def preprocess_for_keywords(review: dict) -> list[str]:
    """
    Prepare review text for keyword extraction.

    Lowercases, removes URLs and non-alphabetic characters, tokenizes,
    filters stopwords, and lemmatizes using POS-aware lemmatization
    so morphological variants like 'charged' and 'charging' collapse to 'charge'.
    """
    text = f"{review.get('title', '')} {review.get('body', '')}".lower()
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in _stopwords and len(t) > 2]
    tagged = pos_tag(tokens)
    return [
        _lemmatizer.lemmatize(token, pos=_get_wordnet_pos(tag))
        for token, tag in tagged
    ]