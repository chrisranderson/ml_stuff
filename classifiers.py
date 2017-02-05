from sklearn import svm as sk_svm
from sklearn.ensemble import RandomForestClassifier

def svm(data, labels):
  classifier = sk_svm.SVC()
  classifier.fit(data, labels)
  return classifier

def forest_classify(data, labels, max_depth=None):
  classifier = RandomForestClassifier(n_estimators=10, max_depth=max_depth)
  classifier = classifier.fit(data, labels)
  return classifier
