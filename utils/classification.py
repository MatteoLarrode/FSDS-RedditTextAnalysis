from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

sns.set()

def train_and_display_confusion_matrix(model, X_train, X_test, y_train, y_test, model_name):
    '''
    function that trains our data on a defined classification algorithm, 
    then creates a confusion matrix and classification report
    @params: 
        model: defined classification model (MNB, Log Reg, SVM)
        X_train,X_test,y_train,y_test: data split on 80/20 train test split 
        model_name: string indicating name of the model
    @returns: 
        classification report and confusion matrix for the model

    '''
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # generate classification report and confusion matrix
    class_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(conf_matrix, display_labels=model.classes_)
    disp.plot(cmap='Oranges')
    plt.title(f'{model_name} Confusion Matrix')
    plt.grid(False)
    plt.show()
    return class_report, conf_matrix

def get_feature_names(model, vectorizer, top_n):
    '''
    function that takes a model with log probability (only MNB in our case but can be extended to RFC etc.)
    and displays the top n words for each class and their log prob
    creates a dataframe of all probabilities categorized by class 

    @params: 
        model: model that we want to investigate
        vectorizer: vectorizer (TFIDF in our use case)
        top_n: how many words we want to extract
    
    @returns: pd.DataFrame of ranked words for each class 
    '''
    feature_names = np.array(vectorizer.get_feature_names_out())
    log_prob = model.feature_log_prob_
    num_classes = len(model.classes_)
    top_words_per_class = {}

    for i, class_label in enumerate(model.classes_):
        # Get the indices of the top words for this class
        top_word_indices = log_prob[i].argsort()[-top_n:][::-1]
        # Map indices to actual words
        top_words = feature_names[top_word_indices]
        # Map words to their log probabilities
        top_words_per_class[class_label] = list(zip(top_words, log_prob[i][top_word_indices]))
    
    data = {class_label: [] for class_label in model.classes_}

    for class_label, top_words in top_words_per_class.items():
        for rank, (word, log_prob) in enumerate(top_words, start=1):
            data[class_label].append((word, np.exp(log_prob)))  # Convert log prob to probability

    # Create a DataFrame with ranked words and their probabilities for each class
    max_rows = max(len(words) for words in data.values())
    for class_label, words in data.items():
        data[class_label] = words + [('', np.nan)] * (max_rows - len(words))  # Pad with empty values

    # Create a DataFrame and format
    df_ranked_words = pd.DataFrame({
        class_label: [f"{word} ({prob:.4f})" for word, prob in words]
        for class_label, words in data.items()
    })

    return df_ranked_words