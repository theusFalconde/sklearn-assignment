import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    movie_reviews_data_folder = r"./data"
    dataset = load_files(movie_reviews_data_folder, shuffle=False)
    n_samples = len(dataset)

    x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.30, random_state=0)

    # ------------------------------------------------------------------------------------------------
    # Ultilizando MultinomialNB
    # ------------------------------------------------------------------------------------------------
    # text_clf = Pipeline([
    #     ('vect', CountVectorizer()),
    #     ('tfidf', TfidfTransformer()),
    #     ('clf', MultinomialNB())
    # ])
    #
    # parameters = {
    #     'vect__ngram_range': [(1, 1), (1, 2)],
    #     'tfidf__use_idf': (True, False),
    #     'clf__alpha': (1e-2, 1e-3),
    # }
    # ------------------------------------------------------------------------------------------------
    # Ultilizando SGDClassifier
    # ------------------------------------------------------------------------------------------------
    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(loss='hinge', penalty='l2', random_state=42, max_iter=5, tol=None))
    ])

    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2)],
        'tfidf__use_idf': (True, False),
        'clf__alpha': (1e-2, 1e-3)
    }
    # ------------------------------------------------------------------------------------------------
    gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)

    gs_clf = gs_clf.fit(x_train, y_train)
    predicted = gs_clf.predict(x_test)

    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

    print(metrics.classification_report(y_test, predicted, target_names=dataset.target_names))

    cm = metrics.confusion_matrix(y_test, predicted)
    print(cm)

    print("Média: %s" % np.mean(predicted == y_test))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    fig.colorbar(cax)

    ax.set_xticklabels([''] + dataset.target_names)
    ax.set_yticklabels([''] + dataset.target_names)

    plt.show()
