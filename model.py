from sklearn import svm
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier


def random_forest(feat_train, level_train, feat_test, n_estimators=6):

  clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=None, max_features='auto')
  print("before fit")
  clf.fit(feat_train, level_train)
  del feat_train

  return clf.predict(feat_test)

def kernel_svm(feat_train, level_train, feat_test):

  clf = svm.SVC(kernel='poly', class_weight='auto')
  clf.fit(feat_train, level_train)
  del feat_train

  return clf.predict(feat_test)