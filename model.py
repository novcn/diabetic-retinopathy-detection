from sklearn import svm
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier


def random_forest(train_set, feat_test, n_estimators=6):

  feat_train = train_set[0]
  level_train = train_set[1]
  clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=None, max_features='auto')
  # print("before fit")
  clf.fit(feat_train, level_train)
  del feat_train

  return clf.predict(feat_test)

def kernel_svm(train_set, feat_test):

  feat_train = train_set[0]
  level_train = train_set[1]
  clf = svm.SVC(kernel='poly', class_weight='auto')
  clf.fit(feat_train, level_train)
  del feat_train

  return clf.predict(feat_test)