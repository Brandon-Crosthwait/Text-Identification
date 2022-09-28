from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline, make_pipeline
import sklearn
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import io

print("Reading in training text... ")
twenty_train = sklearn.datasets.load_files('Training', description=None, categories=None, load_content=True, shuffle=True, encoding=None, decode_error='strict', random_state=0)
print("Done.\n")

print("Training... ")
logistic_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('lr', LogisticRegression(random_state=0, solver="lbfgs", multi_class="auto"))
                         ])
logistic_clf.fit(twenty_train.data, twenty_train.target)

Naive_clf = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('mnb', MultinomialNB())
                      ])

Naive_clf.fit(twenty_train.data, twenty_train.target)

svm_clf = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('lr', LinearSVC())
                    ])
svm_clf.fit(twenty_train.data, twenty_train.target)

randomForest_clf = Pipeline([('vect', CountVectorizer()),
                             ('tfidf', TfidfTransformer()),
                             ('rf', RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0))
                             ])
randomForest_clf.fit(twenty_train.data, twenty_train.target)

print("Done")
def Naive(input):
    text_file = io.open(input, mode="r", encoding="utf-8")
    all_categories_names = np.array(twenty_train.target_names)
    prediction = Naive_clf.predict([text_file.read()])
    return all_categories_names[prediction][0]

def SVM(input):
    text_file = io.open(input, mode="r", encoding="utf-8")
    all_categories_names = np.array(twenty_train.target_names)
    prediction = svm_clf.predict([text_file.read()])
    return all_categories_names[prediction][0]

def RandomForest(input):
    text_file = io.open(input, mode="r", encoding="utf-8")
    all_categories_names = np.array(twenty_train.target_names)
    prediction = randomForest_clf.predict([text_file.read()])
    return all_categories_names[prediction][0]


def LogisticR(input):
    text_file = io.open(input, mode="r", encoding="utf-8")
    all_categories_names = np.array(twenty_train.target_names)
    prediction = logistic_clf.predict([text_file.read()])
    return all_categories_names[prediction][0]


while True:
    endloop = input("What text would you like to identify? (Type 'exit' to quit.) ")
    if endloop == "exit":
        break
    try:
        open(endloop, encoding='utf8')
    except:
        print("Please enter valid file name\n")
        continue

    answers = [Naive(endloop), SVM(endloop), RandomForest(endloop), LogisticR(endloop)]

    print("Naïve Bayes: {}".format(Naive(endloop)))
    print("SVM: {}".format(SVM(endloop)))
    print("Random Forest: {}".format(RandomForest(endloop)))
    print("Logistic Regression: {}".format(LogisticR(endloop)))
    print("My Answer: {}".format(max(set(answers), key=answers.count)))

    answers = []
