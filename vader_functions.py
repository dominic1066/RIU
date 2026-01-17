import nltk
nltk.download('vader_lexicon')
import music_criticism_lexicon

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

def missing_from_vader(corpus: str) -> set:
    """returns a set of words that are in the corpus but not in the VADER lexicon"""
    vader_lexicon = set(sid.lexicon.keys())
    tokens = nltk.word_tokenize(corpus)
    corpus_words = set(tokens)
    missing_words = corpus_words - vader_lexicon
    return missing_words
  
def get_vader_sentiment(sentence):
    """uses VADER to get sentiment for a given sentence"""
    scores = sid.polarity_scores(sentence)
    return scores['compound']

def add_music_criticism_lexicon():
    """adds the music criticism lexicon to the VADER lexicon"""
    sid.lexicon.update(music_criticism_lexicon.missing_adjectives)
    sid.lexicon.update(music_criticism_lexicon.other_missing_words)
    sid.lexicon.update(music_criticism_lexicon.updates_to_vader_values)
