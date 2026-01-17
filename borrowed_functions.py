# BORROWED FROM COURSE MATERIAL!
# make a helper function to create dictionaries.

import requests
from nltk import word_tokenize

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

def calculate_metric(text, dictionary):
  # create empty output container
  metric = []

  # tokenize the text (and any other preprocessing you might need)
  tokens = word_tokenize(text)

  # check if token is in dictionary and append to metric output if so
  for token in tokens:
    if token in dictionary.keys():
      metric.append(dictionary[token])
  return metric

def calculate_metric_tokenised(tokens, dictionary):
  # create empty output container
  metric = []

  # check if token is in dictionary and append to metric output if so
  for token in tokens:
    if token in dictionary.keys():
      metric.append(dictionary[token])

  misses = [token for token in tokens if token not in dictionary.keys()]
  return metric, misses