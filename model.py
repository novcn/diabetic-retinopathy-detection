from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier


def random_forest(feat_train, level_train, feat_test):
  # print(feat_train)
  # print(level_train)

  # filew = open("output.txt", 'w')
  # for image in feat_train:
  #   filew.write("\nSep\n")
  #   for line in image:
  #     filew.write("%s\n" % line)

  rfc = RandomForestClassifier(n_estimators=10, max_depth=None, max_features='auto')
  print("before fit")
  rfc.fit(feat_train, level_train)
  del feat_train

  ret = rfc.predict(feat_test)
  return ret
#  print("ret: ")
 # print(ret)
