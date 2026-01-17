from splits import *
import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.corpus import PlaintextCorpusReader
from helpers import do_scrubbing, remove_header_lines, split_corpus_by_critic, calculate_metric_tokenised, build_hansard_dict
import pandas as pd  # Ensure pandas is imported  
from corpus_functions import *  

import re
from enum import Enum

from sentimentality import CriticSentiment

from vader_functions import get_vader_sentiment, missing_from_vader, add_music_criticism_lexicon

class CorpusIDs(Enum):
    RIU_1977 = 0
    RIU_1997 = 1
    RS_1997 = 2

paths = ['RECORDS 77.*', 'ALBUMS 97.*', 'RS97.*']

# raw_review_corpus = [None, None, None]
# review_corpus = [None, None, None]
# review_tokens = [None, None, None]
# scrubbed_tokens = [None, None, None]
# frequencies = [None, None, None]
# normalised_frequencies = [None, None, None]

test_corpus = "Section:1\nabcde\nSection:2\nfghij\nSection:1\nABCDE\n"
# raw_review_corpus = corpus.raw(['ALBUMS 97.06.01.txt', 'ALBUMS 97.07.01.txt', 'ALBUMS 97.08.01.txt', 'ALBUMS 97.09.01.txt', 'ALBUMS 97.10.01.txt', 'ALBUMS 97.11.01.txt', 'ALBUMS 97.12.01.txt'])

# class CriticSentiment:
#     def __init__(self, name: str):
#         self.name = name
#         self.sentences = []
#         self.sentiment_scores = []
#         self.absolute_average = 0.0
#         self.variance = 0.0
#         self.mean = 0.0

#     def calculate_statistics(self):
#         if len(self.sentiment_scores) == 0:
#             self.absolute_average = 0.0
#             self.mean = 0.0
#             self.variance = 0.0
#             return
#         self.absolute_average = sum(abs(s) for s in self.sentiment_scores) / len(self.sentiment_scores)
#         self.mean = sum(s for s in self.sentiment_scores) / len(self.sentiment_scores)
#         self.variance = sum((s - self.mean) ** 2 for s in self.sentiment_scores) / len(self.sentiment_scores)

#     def plot_sentiment_distribution(self):
#         import matplotlib.pyplot as plt
#         plt.hist(self.sentiment_scores, bins=20, alpha=0.7)
#         plt.title(f'Sentiment Distribution for {self.name}')
#         plt.xlabel('Sentiment Score')
#         plt.ylabel('Frequency')
#         plt.show()

sentiment_by_critic = {}
corpus_sentiment = {}
most_missing_words = {}
add_music_criticism_lexicon()

def calculate_frequencies():
    global most_missing_words
    corpus_root = '.'  # Directory containing text files

    for CORPUS in [CorpusIDs.RIU_1977, CorpusIDs.RIU_1997, CorpusIDs.RS_1997]:
        file_pattern = paths[CORPUS.value]
        corpus = PlaintextCorpusReader(corpus_root, file_pattern)
        raw_review_corpus = corpus.raw()
        corpus_by_critic = split_corpus_by_critic(raw_review_corpus)
        tokenised_corpus_by_critic = {}

        for critic in corpus_by_critic.keys():
            corpus_by_critic[critic] = remove_header_lines(corpus_by_critic[critic], r'^(Reviewed By:|Artist:|Album:|Record Label:).*$')
            
            tokenised_corpus_by_critic[critic] = nltk.word_tokenize(corpus_by_critic[critic])
            tokenised_corpus_by_critic[critic] = do_scrubbing(tokenised_corpus_by_critic[critic], remove_stopwords=False)

        hansard_dict = build_hansard_dict()
        google_dict = get_google_dict()
        SUBTLEXUS_dict = get_SUBTLEXUS_dict()
        most_missing_words = dict(sorted(most_missing_words.items(), key=lambda item: item[1], reverse=True))

        # put results into a dataframe
        frequency_results = []
        
        for critic in sorted(tokenised_corpus_by_critic.keys()):
            g_metric, misses = calculate_metric_tokenised(tokenised_corpus_by_critic[critic], google_dict)
            s_metric, misses = calculate_metric_tokenised(tokenised_corpus_by_critic[critic], SUBTLEXUS_dict)
            h_metric, misses = calculate_metric_tokenised(tokenised_corpus_by_critic[critic], hansard_dict)
            frequency_results.append([critic,
                            round(sum(g_metric)/len(g_metric), 2) if len(g_metric) > 0 else 0,
                            f"{len(g_metric)/len(corpus_by_critic[critic])*100:.2f}%" if len(corpus_by_critic[critic]) > 0 else "0.00%",
                            round(sum(s_metric)/len(s_metric), 2) if len(s_metric) > 0 else 0,
                            f"{len(s_metric)/len(corpus_by_critic  [critic])*100:.2f}%" if len(corpus_by_critic[critic]) > 0 else "0.00%",
                            round(sum(h_metric)/len(h_metric), 2) if len(h_metric) > 0 else 0,
                            f"{len(h_metric)/len(corpus_by_critic[critic])*100:.2f}%" if len(corpus_by_critic[critic]) > 0 else "0.00%"])
        df = pd.DataFrame(data=frequency_results, columns=['Critic', 'log10 Freq (google)', '% Coverage (google)', 'log10 Freq (SUBTLEXUS)', '% Coverage (SUBTLEXUS)', 'log10 Freq (Hansard)', '% Coverage (Hansard)'])
        print(f"Frequency results for corpus {file_pattern}:")
        print(df)
        print(df.describe())

def calculate_sentiments():
    global most_missing_words
    nltk.download(['punkt_tab', 'averaged_perceptron_tagger_eng', 'tagsets_json', 'book'])
    corpus_root = '.'  # Directory containing text files
    corpus_sentiment_results = []

    for CORPUS in [CorpusIDs.RIU_1977, CorpusIDs.RIU_1997, CorpusIDs.RS_1997]:
        file_pattern = paths[CORPUS.value]
        corpus = PlaintextCorpusReader(corpus_root, file_pattern)
        raw_review_corpus = corpus.raw()
        corpus_by_critic = split_corpus_by_critic(raw_review_corpus)
        tokenised_corpus_by_critic = {}
        paragraphs_by_critic = {}
        sentences_by_critic = {}
        corpus_sentiment[CORPUS] = CriticSentiment(str(CORPUS))

        critic_sentiment_results = []

        for critic in corpus_by_critic.keys():
            corpus_by_critic[critic] = remove_header_lines(corpus_by_critic[critic], r'^(Reviewed By:|Artist:|Album:|Record Label:).*$')
            paragraphs_by_critic[critic] = corpus_by_critic[critic].split('\n')
            sentences_by_critic[critic] = re.split(r'[\n.]', corpus_by_critic[critic])
            sentences_by_critic[critic] = [s for s in sentences_by_critic[critic] if s.strip()]
            sentiment_by_critic[critic] = CriticSentiment(critic)
            # for para in paragraphs_by_critic[critic]:
            for para in sentences_by_critic[critic]:
                if len(para.strip()) == 0:
                    continue
                score = get_vader_sentiment(para.lower())
                if (score < -0.95) or (score > 0.95):
                    print(f"Extreme score {score} for critic {critic}: {para}")
                sentiment_by_critic[critic].sentiment_scores.append(score)
                missing = missing_from_vader(para.lower())
                for word in missing:
                    if (word in most_missing_words):
                        most_missing_words[word] = most_missing_words.get(word, 0) + 1
                    else:
                        most_missing_words[word] = 1
            corpus_sentiment[CORPUS].sentiment_scores.extend(sentiment_by_critic[critic].sentiment_scores)
            sentiment_by_critic[critic].calculate_statistics()
            print(f"Critic: {critic}, VADER mean sentiment {sentiment_by_critic[critic].mean:.4f}, mean of absolute values: {sentiment_by_critic[critic].absolute_average:.4f}, variance: {sentiment_by_critic[critic].variance:.4f}") 
            critic_sentiment_results.append([critic,
                                      f"{sentiment_by_critic[critic].mean:.4f}",
                                      f"{sentiment_by_critic[critic].absolute_average:.4f}",
                                      f"{sentiment_by_critic[critic].variance:.4f}"])
            
            tokenised_corpus_by_critic[critic] = nltk.word_tokenize(corpus_by_critic[critic])
            tokenised_corpus_by_critic[critic] = do_scrubbing(tokenised_corpus_by_critic[critic], remove_stopwords=False)

        corpus_sentiment[CORPUS].calculate_statistics()
        corpus_sentiment[CORPUS].variance = sum((s - corpus_sentiment[CORPUS].mean) ** 2 for s in corpus_sentiment[CORPUS].sentiment_scores) / len(corpus_sentiment[CORPUS].sentiment_scores) if len(corpus_sentiment[CORPUS].sentiment_scores) > 0 else 0.0
        print(f"Corpus: {file_pattern}, VADER mean sentiment {corpus_sentiment[CORPUS].mean:.4f}, mean of absolute values: {corpus_sentiment[CORPUS].absolute_average:.4f}, variance: {corpus_sentiment[CORPUS].variance:.4f}")
        corpus_sentiment_results.append([str(CORPUS),
                                        f"{corpus_sentiment[CORPUS].mean:.4f}",
                                        f"{corpus_sentiment[CORPUS].absolute_average:.4f}",
                                        f"{corpus_sentiment[CORPUS].variance:.4f}"])
        most_missing_words = dict(sorted(most_missing_words.items(), key=lambda item: item[1], reverse=True))

        
        df2 = pd.DataFrame(data=critic_sentiment_results, columns=['Critic', 'Mean Sentiment', 'Mean Absolute Sentiment', 'Sentiment Variance'])
        print(f"Frequency results for corpus {file_pattern}:")
        print(f"Sentiment results for corpus {file_pattern}:")
        print(df2)
        # plot_all_sentiment_distributions(sentiment_by_critic)
    df3 = pd.DataFrame(data=corpus_sentiment_results, columns=['Corpus', 'Mean Sentiment', 'Mean Absolute Sentiment', 'Sentiment Variance'])
    print(f"Overall corpus sentiment statistics for corpora:")
    print(df3)
    # plot_sentiment_boxplots(corpus_sentiment)
    plot_sentiment_violins_grid(sentiment_by_critic, n_rows=5, n_cols=8)
    plot_sentiment_violins(corpus_sentiment)
    # plot_all_sentiment_distributions(corpus_sentiment, batch_size=3)

def plot_all_sentiment_distributions(sentiment_dict, batch_size=15):
    import matplotlib.pyplot as plt
    critics = list(sentiment_dict.values())
    
    for i in range(0, len(critics), batch_size):
        batch = critics[i:i+batch_size]
        n_plots = len(batch)
        n_cols = 3
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 5*n_rows))
        axes = axes.flatten() if n_plots > 1 else [axes]
        
        for idx, critic in enumerate(batch):
            axes[idx].hist(critic.sentiment_scores, bins=5, alpha=0.7)
            axes[idx].set_title(f'{critic.name}')
            axes[idx].set_xlabel('Sentiment Score')
            axes[idx].set_ylabel('Frequency')
        
        # Hide empty subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout(pad=8.0)
        plt.show()

def plot_sentiment_boxplots(sentiment_dict):
    import matplotlib.pyplot as plt
    names = [c.name for c in sentiment_dict.values()]
    scores = [c.sentiment_scores for c in sentiment_dict.values()]
    
    plt.figure(figsize=(12, 6))
    plt.boxplot(scores, labels=names)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Sentiment Score')
    plt.title('Sentiment Distribution Comparison')
    plt.tight_layout()
    plt.show()

def plot_sentiment_violins(sentiment_dict):
    import matplotlib.pyplot as plt
    
    names = [c.name for c in sentiment_dict.values()]
    scores = [c.sentiment_scores for c in sentiment_dict.values()]
    
    plt.figure(figsize=(12, 6))
    parts = plt.violinplot(scores, positions=range(len(names)), showmeans=True, showmedians=True)
    plt.xticks(range(len(names)), names, rotation=45, ha='right')
    plt.ylabel('Sentiment Score')
    plt.title('Sentiment Distribution (Violin Plot)')
    plt.tight_layout()
    plt.show()

def plot_sentiment_violins_grid(sentiment_dict, n_rows=3, n_cols=3, min_sentences=30):
    import matplotlib.pyplot as plt
    
    # Filter critics by minimum sentence count
    critics = [c for c in sentiment_dict.values() if len(c.sentiment_scores) >= min_sentences]
    plots_per_page = n_rows * n_cols
    
    # Scale figure size inversely with number of plots
    base_width = min(4, 20 / n_cols)
    base_height = min(3.5, 18 / n_rows)
    
    for page in range(0, len(critics), plots_per_page):
        batch = critics[page:page + plots_per_page]
        n_plots = len(batch)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(base_width*n_cols, base_height*n_rows))
        axes = axes.flatten() if n_plots > 1 else [axes]
        
        for idx, critic in enumerate(batch):
            parts = axes[idx].violinplot([critic.sentiment_scores], positions=[0], 
                                          showmeans=True, showmedians=True)
            axes[idx].set_title(f'{critic.name}', fontsize=8)
            axes[idx].set_ylabel('Sentiment Score', fontsize=7)
            axes[idx].tick_params(labelsize=6)
            axes[idx].set_xticks([])
        
        # Hide empty subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout(pad=3.0, h_pad=4.0, w_pad=2.0)
        page_num = page // plots_per_page + 1
        total_pages = (len(critics) + plots_per_page - 1) // plots_per_page
        plt.suptitle(f'Page {page_num} of {total_pages}', y=0.998, fontsize=10)
        plt.show()

def plot_sentiment_violins_tabbed(sentiment_dict, n_rows=3, n_cols=3, min_sentences=0):
    """
    Interactive tabbed version for Jupyter notebooks.
    Requires: pip install ipywidgets
    """
    try:
        import matplotlib.pyplot as plt
        from ipywidgets import widgets, interact, Output
        from IPython.display import display, clear_output
    except ImportError:
        print("This function requires ipywidgets. Install with: pip install ipywidgets")
        print("Falling back to multi-page version...")
        plot_sentiment_violins_grid(sentiment_dict, n_rows, n_cols, min_sentences)
        return
    
    # Filter critics by minimum sentence count
    critics = [c for c in sentiment_dict.values() if len(c.sentiment_scores) >= min_sentences]
    plots_per_page = n_rows * n_cols
    total_pages = (len(critics) + plots_per_page - 1) // plots_per_page
    
    # Scale figure size inversely with number of plots
    base_width = min(4, 20 / n_cols)
    base_height = min(3.5, 18 / n_rows)
    
    def show_page(page):
        start_idx = page * plots_per_page
        batch = critics[start_idx:start_idx + plots_per_page]
        n_plots = len(batch)
        
        plt.close('all')  # Close previous figures
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(base_width*n_cols, base_height*n_rows))
        axes = axes.flatten() if n_plots > 1 else [axes]
        
        for idx, critic in enumerate(batch):
            parts = axes[idx].violinplot([critic.sentiment_scores], positions=[0], 
                                          showmeans=True, showmedians=True)
            axes[idx].set_title(f'{critic.name}', fontsize=8)
            axes[idx].set_ylabel('Sentiment Score', fontsize=7)
            axes[idx].tick_params(labelsize=6)
            axes[idx].set_xticks([])
        
        # Hide empty subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout(pad=3.0, h_pad=4.0, w_pad=2.0)
        plt.suptitle(f'Page {page + 1} of {total_pages}', y=0.998, fontsize=10)
        plt.show()
    
    # Create interactive slider
    interact(show_page, page=widgets.IntSlider(
        min=0, 
        max=total_pages-1, 
        step=1, 
        value=0, 
        description='Page:',
        continuous_update=False
    ))

def main():
    calculate_frequencies()
    #calculate_sentiments()

if __name__ == "__main__":
    main()