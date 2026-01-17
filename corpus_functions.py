from ast import pattern
import nltk
import string
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re

from enum import Enum

class CorpusIDs(Enum):
    RIU_1977 = 0
    RIU_1997 = 1
    RS_1997 = 2

punctuation_set = set(string.punctuation)

# additional punctuation are pieces of text that I found were appearing in the results 
# that I would like to have been removed with other punctuation
additional_punctuation = set(["'s", "'m", "'", '.', '"', "n't","'ve",
                              '’', '“', '”', '’;', '’’', '.)', '("',
                              '–', '—', '...', '``', "''", '‘', ',"',
                              '•', '”.', '",', '”,', '.):', '":', '’)',
                              '.”', ".'", '’,','’),', '–,', '—,', '’.', 
                              '(‘', '".', ').', '."', "')", '!!',
                              '")', '"),', '").', '";', '.’',
                              "',", '’).', "-'", '--', '?"'])

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

nltk.download('wordnet')
nltk.download('omw-1.4') 

lemmatizer = WordNetLemmatizer()
corpus_root = '.'  # Directory containing text files
file_pattern = r'.*\.txt'  # Pattern to match .txt files
tokeniser = nltk.tokenize.WordPunctTokenizer()


# define a few functions.

def do_scrubbing(tokens: list, remove_punctuation=True, remove_additional_puctuation=True, remove_stopwords=True, lowercase=True, lemmatize=True) -> list:
    # Most of the cleaning out of punctuation, stopwords, plural forms etc, all in one place.
    # print("Initial token count:", len(tokens))
    if lowercase:
        tokens = [token.lower() for token in tokens]
    if (remove_punctuation):
        tokens = [word for word in tokens if word not in punctuation_set]
        # tokens = [token.translate(str.maketrans('', '', string.punctuation)) for token in tokens]

    if remove_additional_puctuation:
        tokens = [word for word in tokens if word not in additional_punctuation]

    if remove_stopwords:
        tokens = [word for word in tokens if word not in stop_words]

    if lemmatize:
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # print("Final token count after scrubbing:", len(tokens))
    return tokens

# for analysing the text of the reviews, I prefer to exclude the headers that contain album and artist names, record labels and reviewer names
def remove_header_lines(body: str, pattern: str) -> str:
    lines = body.split('\n')
    reg_exp = re.compile(pattern)
    cleaned_lines = [line for line in lines if not reg_exp.search(line)]
    return '\n'.join(cleaned_lines)

def draw_wordcloud(frequencies: FreqDist, title: str):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(frequencies.most_common(20)))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off") # Turn off the axis labels
    plt.title(title)
    plt.show()

def normalise_frequencies(frequencies: FreqDist, token_count: int) -> FreqDist:
    normalized_frequencies = FreqDist()
    for word, freq in frequencies.items():
        norm_freq = freq / token_count * 100
        normalized_frequencies[word] = norm_freq
    return normalized_frequencies

import os
import re
from collections import Counter
import numpy as np

# Path to your folder containing Hansard .txt files
folder_path = 'hansard'

def get_hansard_frequencies(folder):
    word_counts = Counter()
    total_words = 0

    for filename in os.listdir(folder):
        if filename.endswith('.txt'):
            with open(os.path.join(folder, filename), 'r', encoding='utf-8') as f:
                # Basic cleaning: lowercase and keep only alphanumeric characters
                text = f.read().lower()
                words = re.findall(r'\b\w+\b', text)
            
                word_counts.update(words)
                total_words += len(words)

    # Convert to DataFrame
    df = pd.DataFrame(word_counts.items(), columns=['word', 'raw_count'])

    # Normalize to Frequency Per Million Words (FPMW)
    df['fpmw'] = (df['raw_count'] / total_words) * 1000000

    # Apply Log10 transformation
    df['log10_fpmw'] = np.log10(df['fpmw'] + 1)

    return df, total_words

def get_hansard_dict()->dict:
    word_counts = Counter()
    total_words = 0
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                # Basic cleaning: lowercase and keep only alphanumeric characters
                text = f.read().lower()
                words = re.findall(r'\b\w+\b', text)
                
                word_counts.update(words)
                total_words += len(words)
    
    # Convert to DataFrame
    df = pd.DataFrame(word_counts.items(), columns=['word', 'raw_count'])
    
    # Normalize to Frequency Per Million Words (FPMW)
    df['fpmw'] = (df['raw_count'] / total_words) * 1000000
    
    # Apply Log10 transformation
    df['log10_fpmw'] = np.log10(df['fpmw'] + 1)
    
    # return df, total_words

    hansard_df, grand_total = get_hansard_frequencies(folder_path)
    hansard_dict = dict(zip(hansard_df['word'], hansard_df['log10_fpmw']))
    return hansard_dict

import kagglehub
from borrowed_functions import get_word_rating_resource

# Download latest version
path = kagglehub.dataset_download("rtatman/english-word-frequency")


def get_SUBTLEXUS_dict()->dict:
    SUBTLEXUS_url = 'https://raw.githubusercontent.com/scskalicky/LING-226-vuw/main/lexical-resources/subtlxus_frequency.txt'
    SUBTLEXUS_dict = get_word_rating_resource(SUBTLEXUS_url)
    return SUBTLEXUS_dict

import pandas as pd
import numpy as np

def get_google_dict()->dict:
    import kagglehub

    # Download latest version
    path = kagglehub.dataset_download("rtatman/english-word-frequency")

    # Read the CSV file
    df = pd.read_csv(f"{path}/unigram_freq.csv")

    total_count = df['count'].sum()
    TOTAL_GOOGLE_CORPUS_WORDS = 100000000000

    df['fpmw'] = df['count'] / TOTAL_GOOGLE_CORPUS_WORDS * 1000000
    df['log10_fpmw'] = round(np.log10(df['fpmw'] + 1), 3)

    # put word and log10_frequency into a dictionary
    google_dict = dict(zip(df['word'], df['log10_fpmw']))
    return google_dict

