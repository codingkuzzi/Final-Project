# Helper functions for EDA, data preprocessing, the models performance visualization  and comparing the results.

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# Functions for EDA
def display_distribution_of_labels(dataframe):
    """
    Displays the distribution of labels (real or fake) in the dataset using a count plot
    Parameters:
    - dataframe (pandas.DataFrame): The dataset containing the labels column ('label') 
                                  with class values (e.g., 'real', 'fake').
    """
    sns.countplot(x = 'label', data = dataframe)
    plt.title('Distribution of Labels in the Dataset')
    plt.show()

def plot_top_words(articles, is_stopwords, number_to_display):
    """
    Plots the top words in a collection of articles, either including or excluding stopwords, 
    based on the user's choice. The function also rotates the x-axis labels for better readability.
    
    Parameters:
    - articles (pd.Series): A pandas Series containing the text of the articles.
    - is_stopwords (bool): A flag to indicate whether to include stopwords. 
        True to include stopwords, False to exclude them.
    - number_to_display (int): The number of top words to display in the plot.
    """
    # Create a set of English stopwords from 'stopwords' module of nltk library
    stopwords_set = set(stopwords.words('english'))
    
    # Tokenize the articles into lists of words
    tokenized_articles = articles.str.split()
    
    # Flatten the list of words from all articles
    all_words = [word for articles_words in tokenized_articles for word in articles_words]
    
    # Count the occurrences of either stopwords or non-stopwords
    word_counts = Counter(all_words)
    
    # Filter the 'word_counts' dictionary based on the specified is_stopwords parameter.
    # If 'is_stopwords' is True, keep only words in the 'stopwords_set'. Otherwise, keep only words not in the 'stopwords_set'.
    word_counts = {word: count for word, count in word_counts.items() if word in stopwords_set} if is_stopwords else {word: count for word, count in word_counts.items() if word not in stopwords_set}
    
    # Sort the words by their occurrences in descending order
    top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:number_to_display]
    
    # Unpack the sorted words and counts into separate lists for plotting
    top_words, counts = zip(*top_words)
    
    # Plot the top words
    plt.figure(figsize=(15, 6))
    plt.bar(top_words, counts)
    plt.ylabel('Frequency')
    plt.xlabel('Words')
    
    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    if is_stopwords:
        plt.title(f'Top {number_to_display} Stopwords in Articles')
    else:
        plt.title(f'Top {number_to_display} Non-Stopwords in Articles')
    
    plt.tight_layout()  # Adjust layout to make sure everything fits without overlapping
    plt.show()

# Functions for data preprocessing and cleaning stopwords out of articles subset

# BeautifulSoup library for HTML parsing.
from bs4 import BeautifulSoup
# re (regular expression) module for text pattern matching.
import re

# Use BeautifulSoup to parse reviews subset and remove HTML tags from it. 
def remove_html_tags(articles):
    soup = BeautifulSoup(articles, "html.parser")
    return soup.get_text()

# Use a regular expression (re.sub()) to replace URLs (strings starting with 'http' followed by non-whitespace characters) with an empty string, effectively removing URLs from the articles content.
def remove_urls(articles):
    return re.sub(r'http\S+', '', articles)

# Use a regular expression to replace any character that is not an alphabet letter with a space. This helps remove special characters and numeric values, leaving only alphabetic characters.
def remove_spec_numer_char(articles):
    articles = re.sub(r'[^a-zA-Z]', ' ', articles)
    return articles

# Remove the stopwords from articles
def remove_stopwords(articles):
    # Create a set of English stopwords from 'stopwords' module of nltk library.
    stop_words = set(stopwords.words('english'))

    final_articles_text = [i.strip() for i in articles.split() if i.strip().lower() not in stop_words and i.strip().lower().isalpha()]
    return " ".join(final_articles_text)

# Helper function to remove the noisy text and stopwords
def denoise_text(text):
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_spec_numer_char(text)
    text = remove_stopwords(text)
    return text

# Helper function for processing data but keeping stopwords in the dataset
def denoise_text_keep_stopwords(text):
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_spec_numer_char(text)
    return text

# Function to remove the words 'Reuters' or 'reuters' from articles
def remove_reuters(articles):
    return re.sub(r'\b[Rr]euters\b', '', articles)

# Helper function to denoise text excluding 'Reuters' or 'reuters' words from articles
def denoise_text_and_remove_reuters(text):
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_spec_numer_char(text)
    text = remove_reuters(text)  # Include the new function here
    text = remove_stopwords(text)
    return text

def display_number_of_words_in_text(dataframe):
    """
    Displays a histogram showing the distribution of the number of words in text for true and fake news articles.
    Parameters:
    - dataframe (pandas.DataFrame): The dataset containing the news articles in a 'text' column and the corresponding
      'label' column, where 1 represents true articles and 0 represents fake articles.
    """

    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,4))
    text_len = dataframe[dataframe['label'] == 1]['text'].str.split().map(lambda x: len(x))
    ax1.hist(text_len, color = 'green')
    ax1.set_title('Text in True Articles')
    text_len = dataframe[dataframe['label'] == 0]['text'].str.split().map(lambda x: len(x))
    ax2.hist(text_len, color = 'red')
    ax2.set_title('Text in Fake Articles')
    fig.suptitle('Words in texts')
    plt.show()

def get_top_text_ngrams(news_text, n, gram_value):
    """
    This function uses CountVectorizer to compute the frequency of n-grams (based on the specified gram_value) 
    in the provided news articles and returns a dictionary of the top N most frequent n-grams along with their counts.
    Parameters:
    - news_text (iterable): A collection (e.g., list or pandas Series) of news articles (strings) from which the n-grams 
      will be extracted.
    - n (int): The number of top n-grams to return based on their frequency.
    - gram_value (int): The value of n for the n-grams (e.g., 1 for unigrams, 2 for bigrams, 3 for trigrams).
    Returns:
    dict: A dictionary where the keys are the top N n-grams (strings) and the values are their respective counts (integers).
    """
    # Create an instance of CountVectorizer with specified n-gram range
    vectorizer = CountVectorizer(ngram_range=(gram_value, gram_value)).fit(news_text)
    # Transform the reviews into a sparse matrix of n-gram counts
    vectors = vectorizer.transform(news_text)
    # Sum the counts of each n-gram across all news
    sum_words = vectors.sum(axis=0)
    # Create a list of tuples containing each n-gram and its total count
    words_frequencies = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    # Sort the list in descending order based on n-gram counts
    words_frequencies = sorted(words_frequencies, key = lambda x: x[1], reverse=True)
    # Return a dictionary containing the top N n-grams and their counts
    return dict(words_frequencies[:n])

# The function to plot n-grams dictionary data
def plot_top_ngrams(ngrams_dict, title):
    """
    This function takes a dictionary of n-grams and their counts and visualizes them in a bar chart using Plotly Express.
    Parameters:
    - ngrams_dict (dict): A dictionary where the keys are n-grams (strings) and the values are their respective counts (integers).
    - title (str): The title of the plot, providing context for the data being visualized.
    """
    ngrams_df = pd.DataFrame(columns = ["Common_words" , 'Count'])
    ngrams_df["Common_words"] = list(ngrams_dict.keys())
    ngrams_df["Count"] = list(ngrams_dict.values())
    # Create a horizontal bar chart using Plotly Express (px)
    fig = px.bar(ngrams_df, x = "Count", y = "Common_words", title = title, orientation='h', 
                width = 700, height = 700,color ='Common_words')
    # Show the plot
    fig.show()

# Functions to visualize the models performance and compare the results

# A function for the model evaluation based on accuracy, precision, recall and F1-score.
def evaluate_model(test_labels, predicted_labels):
  # Use the accuracy_score function from sklearn.metrics module to calculate the accuracy of the model by comparing true labels with predicted labels.
  model_accuracy = accuracy_score(test_labels, predicted_labels)

  # Use the precision_recall_fscore_support function to calculate precision, recall, and F1-score.
  # The average="weighted" parameter specifies that these metrics should be computed using weighted averaging, which takes class imbalance into account (just to be safe as we know we have balansed sets).
  model_precision, model_recall, model_f1,_= precision_recall_fscore_support(test_labels, predicted_labels, average = "weighted")

  # Create a dictionary containing the calculated accuracy, precision, recall, and F1-score.
  model_results = {"accuracy": model_accuracy,
                 "precision": model_precision,
                 "recall": model_recall,
                 "f1": model_f1}
  return model_results

# A function to evaluate models performance on both training and test datasets.
def get_score(model, train_x, test_x, train_y, test_y):
    """
    This function fits the model to the provided training data and calculates its accuracy score on both the training 
    and test datasets. It then prints the training and testing accuracy scores.
    Parameters:
    - model: A machine learning model object that implements the fit and score methods.
    - train_x (array-like): Training data features.
    - test_x (array-like): Test data features.
    - train_y (array-like): Labels for the training data.
    - test_y (array-like): Labels for the test data.
    """
    model.fit(train_x, train_y)
    train_score = model.score(train_x, train_y)
    test_score = model.score(test_x, test_y)
    print('training score:' + str(train_score))
    print('testing score:' + str(test_score))

# A function to plot confusion matrix
def plot_confusion_matrix(cm):
    """
    This function takes a confusion matrix as input and visualizes it using a heatmap. 
    Parameters:
    - cm (array-like): Confusion matrix 
    """
    print(cm)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

    # labels, title and ticks
    ax.set_xlabel('Predicted labels');
    ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['real', 'fake']);
    ax.yaxis.set_ticklabels(['real','fake']);

def plot_accuracy_vs_efficiency(dataframe, classifier_name, hue_param, style_param):
    """
    The function creates a scatter plot showing the relationship between the mean fit time (training time) 
    and the mean test score (accuracy). Different colors and markers represent different parameter values 
    (e.g., n-gram range or regularization strength). 

    Parameters:
    - dataframe: pandas DataFrame
      A DataFrame containing the results of the grid search, with columns such as 'mean_fit_time', 'mean_test_score', and the hyperparameters.
    - classifier_name: str
      The name of the classifier being evaluated (used in the plot title).
    - hue_param: str
      The column name in the DataFrame that will be used for coloring the points based on the hyperparameter values.
    - style_param: str
      The column name in the DataFrame that will be used for changing the marker style based on the hyperparameter values.
    """
    plt.figure(figsize=(6, 4))
    # Define a color palette (modify colors as needed)
    color_palette = {0: 'purple', 0.1: 'blue', 1: 'green', 10: 'purple', (1, 1): 'blue', (1, 2): 'green', (1, 3): 'purple', 0.0001: 'purple', 0.001: 'green'}
    legend_title =  ''
    if (hue_param == style_param):
        sns.scatterplot(data=dataframe, x='mean_fit_time', y='mean_test_score', hue=hue_param, palette=color_palette, s=100)
        legend_title=hue_param
    else:  
        sns.scatterplot(data=dataframe, x='mean_fit_time', y='mean_test_score', hue=hue_param, style=style_param, palette=color_palette, s=100)

    # Additional plot settings
    plt.title('Trade-off Between Accuracy and Training Efficiency -  {}'.format(classifier_name))
    plt.xlabel('Mean Fit Time (seconds)')
    plt.ylabel('Mean Test Score')
    plt.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc='upper left')
    #title='Regularization Strength (C)',
    plt.grid(True)
    plt.show()

def plot_comparison_by_classifier_and_feature_extraction(dataframe, param_to_compare):
    """
    Creates a bar plot that visually compares the performance or training efficiency of different classifiers,
    segmented by various feature extraction methods.

    Parameters:
    dataframe (pd.DataFrame): A pandas DataFrame containing the results of classifiers. It should include the following columns:
                              - 'classifier': Name of the classifier (e.g., 'Naive Bayes', 'Logistic Regression', etc.).
                              - 'feature_extraction': The feature extraction method used (e.g., 'TF-IDF', 'CountVectorizer').
                              - 'mean_test_score': Mean test score of the model.
                              - 'mean_fit_time': Mean fit time of the model.
    param_to_compare (str): The parameter to compare, either 'mean_test_score' for performance comparison or 'mean_fit_time' for
                            training efficiency comparison.
    """
    plt.figure(figsize=(10, 8))

    algo_model_figure = sns.barplot(y='classifier', x=param_to_compare, hue='feature_extraction', data=dataframe, width=0.5)

    if param_to_compare == 'mean_test_score':
        plt.title('Performance Comparison by Classifier and Feature Extraction')
        plt.xlabel('Mean Test Score')
    else:
        plt.title('Training Efficiency Comparison by Classifier and Feature Extraction')
        plt.xlabel('Mean Fit Time')    
    plt.ylabel('Classifier')
    plt.legend(title='Feature Extraction Method', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Annotating each bar with the mean test score
    for p in algo_model_figure.patches:
        width = p.get_width()
        # Adjust annotation threshold to skip near-zero values
        if width > 0.001:  # Increase or decrease this threshold based on your data range
            annotation = f'{width:.3f}' if width > 0.001 else ''
            plt.text(p.get_x() + width + 0.01,  # Slight right offset to avoid clutter
                     p.get_y() + p.get_height() / 2,
                     annotation,
                     va='center')

    plt.show()

def visualize_classifier_results_with_params(df, classifier_name, hue_param, size_param, style_param):
    """
    Visualizes the performance and training efficiency of a classifier using different
    vectorization techniques, parameter configurations, and stopword settings.

    Parameters:
    df (DataFrame): A pandas DataFrame containing the results of classifier models.
                    The DataFrame should have the following columns:
                    - Vectorizer: The vectorization technique used (TF-IDF or CountVectorizer).
                    - use_idf: Whether IDF was used (True/False or None) for TF-IDF models.
                    - C: The regularization parameter for models like Logistic Regression or SVC (if applicable, or can be None).
                    - alpha: The regularization parameter for SGD (if applicable, or can be None).
                    - n-gram range: The n-gram range used for the vectorizer (e.g., (1, 1), (1, 2), etc.).
                    - With Stopwords: Flag indicating whether stopwords were included (True for included, False for excluded).
                    - Mean Test Score: The mean test accuracy of the model.
                    - Mean Fit Time: The mean training time of the model in seconds.                
    classifier_name (str): The name of the classifier to display in the plot titles (e.g., 'Logistic Regression', 'Naive Bayes').
    hue_param (str): The column name to use for coloring points in the scatter plot (e.g., 'C', 'alpha').
    size_param (str): The column name to use for adjusting the size of points in the scatter plot (e.g., 'With Stopwords').
    style_param (str): The column name to use for adjusting the style of points in the scatter plot (e.g., 'n-gram range').
    """
    
    sns.set(style="whitegrid")
    
    tfidf_df = df[df['Vectorizer'] == 'TF-IDF']
    count_vect_df = df[df['Vectorizer'] == 'CountVectorizer']
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    
    # Define a color palette
    color_palette = {0.1: 'blue', 1: 'green', 10: 'red', 0.0001: 'purple', 0.001: 'orange', 0: 'red', (1, 1): 'blue', (1, 2): 'green', (1, 3): 'purple'}

    sns.scatterplot(
        ax=axes[0],
        data=tfidf_df,
        x='Mean Fit Time',
        y='Mean Test Score',
        hue=hue_param, 
        style=style_param,
        size=size_param, 
        sizes=(50, 150),  # Set size range
        palette=color_palette,
        legend='full'
    )
    
    sns.scatterplot(
        ax=axes[1],
        data=count_vect_df,
        x='Mean Fit Time',
        y='Mean Test Score',
        hue=hue_param,  
        style=style_param,
        size=size_param,  
        sizes=(50, 150),  # Set size range
        palette=color_palette,
        legend='full'
    )
    
    axes[0].set_title(f'{classifier_name} Performance with TF-IDF Vectorizer')
    axes[0].set_xlabel('Mean Fit Time (seconds)')
    axes[0].set_ylabel('Mean Test Score')
    
    axes[1].set_title(f'{classifier_name} Performance with CountVectorizer')
    axes[1].set_xlabel('Mean Fit Time (seconds)')
    
    plt.tight_layout()
    plt.show()

# Function to predict user-input text using the full model pipeline
def predict_user_input_text(user_input_text, model):
    """
    Predict whether a given news article text is real or fake using the trained model.

    Parameters:
    - user_input_text: str
      The input text (news article) provided by the user.
    - model: Trained model object (Pipeline)
      The trained classification model (Pipeline) used for making predictions.
    """
    # Ensure the input is in the form of a list since the model pipeline expects a list of strings
    user_input_text = [user_input_text]  # Wrapping the input string in a list
    
    # Use the entire model pipeline to transform the input and make a prediction
    prediction = model.predict(user_input_text)[0]
    
    # Interpret the prediction
    predicted_label = "Real" if prediction == 1 else "Fake"
    
    # Output the prediction
    print(f"Predicted Label for the given text: {predicted_label}")

def evaluate_model_predictions(test_text_data, test_label_data, model):
    """
    Function to test the selected model on the test dataset and print the output when predictions are incorrect,
    along with the number and percentage of incorrect predictions out of the total.

    Parameters:
    - test_text_data: The text data to use for testing.
    - test_label_data: The actual labels corresponding to the test data.
    - model: Trained model to use for predictions.
    """
    # Model Predictions
    predictions = model.predict(test_text_data)

    # Initialize counters for incorrect predictions
    incorrect_count = 0
    total_count = len(predictions)

    # For each prediction, determine the predicted and actual sentiment
    for i, pred in enumerate(predictions):
        # Get actual label value
        actual_label_value = test_label_data.iloc[i]
        predicted_label = "Real" if pred == 1 else "Fake"
        actual_label = "Real" if actual_label_value == 1 else "Fake"
        
        # Check if the prediction is correct
        if pred != actual_label_value:
            print(f"Review {i}: Predicted Label - {predicted_label}, Actual Label - {actual_label} (Incorrect)")
            incorrect_count += 1

    # Calculate percentage of incorrect predictions
    incorrect_percentage = (incorrect_count / total_count) * 100

    # Print the final count of incorrect predictions with percentage
    print(f"\nTotal Incorrect Predictions: {incorrect_count} out of {total_count} ({incorrect_percentage:.2f}%)")