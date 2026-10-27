import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(URL, sep="\t", header=None, names=["label", "text"])
print("Messages loaded:", len(df))
df["target"] = (df["label"] == "spam").astype(int)

X_train, X_test, y_train, y_test = train_test_split(df["text"], df["target"], test_size=0.2, random_state=42, stratify=df["target"])

model = make_pipeline(TfidfVectorizer(stop_words="english"), MultinomialNB())
model.fit(X_train, y_train)

pred = model.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, pred) * 100:.2f}%\n")
print(classification_report(y_test, pred, target_names=["ham", "spam"]))
print("Confusion matrix:\n", confusion_matrix(y_test, pred))

samples = ["Congratulations! You have won a free iPhone, click the link now", "Hey, are you coming to class tomorrow?", "URGENT! Your bank account is blocked, verify immediately"]
for s, p in zip(samples, model.predict(samples)):
    print(f"\n'{s}'\n -> {'SPAM' if p else 'NOT SPAM'}")