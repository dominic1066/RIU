import string
import nltk

import re
import matplotlib.pyplot as plt
from nltk import FreqDist
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
import numpy as np

lemmatizer = WordNetLemmatizer()
from wordcloud import WordCloud

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

# define a few functions.
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

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

# BORROWED FROM COURSE MATERIAL!
# make a helper function to create dictionaries.

# to grab the resource by url, we'll import requests
# could also use !wget or other URL libraries
import requests

# create a function to read in resource and output a dictionary.
def get_word_rating_resource(url):
  """helper function to get lexical resources
  resources are hosted on github as .txt files in the form of Word\tValue\n
  """
  # read the raw text and split on newlines
  raw = requests.get(url).text.split('\n')

  # split each pair and convert value to rounded float
  # the if statement is there to avoid indexing errors when a row in a resource doesn't have complete data
  raw_list = [(pair.split('\t')[0], round(float(pair.split('\t')[1]), 3)) for pair in raw if len(pair.split('\t')) == 2]

  # create a dictionary and return it
  return dict(raw_list)


from collections import Counter
import os

from splits import *



def build_frequencies(folder):
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

def build_hansard_dict():
    folder_path = 'hansard'
    hansard_df, grand_total = build_frequencies(folder_path)
    # print(f"Processed {grand_total} total words from Hansard.")
    # print(hansard_df.sort_values(by='fpmw', ascending=False).head(10))
    hansard_dict = dict(zip(hansard_df['word'], hansard_df['log10_fpmw']))
    return hansard_dict

# def get_SUBTLEXUS_dict():
#     SUBTLEXUS_url = 'https://raw.githubusercontent.com/scskalicky/LING-226-vuw/main/lexical-resources/subtlxus_frequency.txt'
#     SUBTLEXUS_dict = get_word_rating_resource(SUBTLEXUS_url)
#     return SUBTLEXUS_dict

# def get_google_dict():
#     import kagglehub

#     # Download latest version
#     path = kagglehub.dataset_download("rtatman/english-word-frequency")

#     print("Path to dataset files:", path)

#     # SUBTLEXUS_url = 'https://raw.githubusercontent.com/scskalicky/LING-226-vuw/main/lexical-resources/subtlxus_frequency.txt'
#     # SUBTLEXUS_dict = get_word_rating_resource(SUBTLEXUS_url)

#     # import pandas as pd
#     # import numpy as np


#     # Read the CSV file
#     df = pd.read_csv(f"{path}/unigram_freq.csv")

#     total_count = df['count'].sum()
#     # print(f"Total count: {total_count}")
#     TOTAL_GOOGLE_CORPUS_WORDS = 100000000000    # apparently...

#     df['fpmw'] = df['count'] / TOTAL_GOOGLE_CORPUS_WORDS * 1000000
#     df['log10_fpmw'] = round(np.log10(df['fpmw'] + 1), 3)
#     # print(df.head(10))

#     # put word and log10_frequency into a dictionary
#     google_dict = dict(zip(df['word'], df['log10_fpmw']))
#     return google_dict

def calculate_metric_tokenised(tokens, dictionary):
  # create empty output container
  metric = []

  # check if token is in dictionary and append to metric output if so
  for token in tokens:
    if token in dictionary.keys():
      metric.append(dictionary[token])

  misses = [token for token in tokens if token not in dictionary.keys()]
  return metric, misses