import theano
import theano.tensor as T

import os
import util
import model
import csv
import numpy as np
np.set_printoptions(threshold=1000)
import time
import sys
import convnet
import kappa

def import_image(_name, _phase, _set, lcn=False):
  try:
    img = util.im_load(_name, _phase, _set)
  except IOError:
    print("warning %s not found " % (_name))
    return []
  mid = util.find_mid(img)
  img = util.im_crop(img, mid)
  img = util.im_resize(img)
  im_len = np.prod(img.shape)
  img = np.reshape(img, im_len)
  nan_check = True
  if(lcn == True):
    tmp_img = util.im_lcn(img)
    nan_check = any(np.isnan(x) for x in tmp_img.flatten())
  if (nan_check == True):
    return img
  else:
    return tmp_img

test_files = 0

def get_data_set(_phase, _set, save=False, lcn=False):
  imgs = []
  lvls = []

  if(_phase == "test"):
    global test_files
    test_files = os.listdir("data/test/%s" % (_set))
    for image in test_files:
      name = image.replace(".jpeg", "")
      img = import_image(name, _phase, _set, lcn=lcn)
      if img != []:
        imgs.append(img)

    return imgs
  
  if(_phase == "train" or _phase == "valid"):
    with open('data/csv/%s.csv' % (_set + "_" + _phase)) as csvfile:
      reader = csv.DictReader(csvfile)
      for row in reader:
        img = import_image(row["image"], _phase, _set, lcn=lcn)
        if img != []:
          imgs.append(img)
          lvls.append(row["level"])

        if(save == True):
          print("saving image...")
          util.im_save(row["image"], img, _set)


    imgs = np.vstack(imgs)
    set = (imgs, lvls)
    return set
  
  else:
    print("Invalid phase: %s " % (_phase))

def get_stats(_predictions, _csv_test, _label):
  if(len(_predictions.shape) == 1):
    _predictions = np.array([_predictions])

  stats_file = open("statistics.txt", 'a')  
  human_ratings = []
  machine_ratings = []
  with open('data/csv/%s' % (_csv_test)) as csvfile:
    thresh = (1. / 2.) + .0001
    reader = csv.DictReader(csvfile)
    correct = 0
    votes = np.array([0, 0, 0, 0, 0])
    k = 0
    for row in reader:
      for predict in _predictions:
        votes[predict[k]] += 1
      human_vote = int(row["level"])

      _max = max(votes)
      best = np.where(votes==_max)
      confidence = float(_max) / float(len(votes))
      if(confidence < thresh):
        machine_vote = 0
      else:
        machine_vote = int(best[0][0])

      machine_ratings.append(machine_vote)
      human_ratings.append(human_vote)
      if(machine_vote == human_vote):
        correct += 1
      k += 1

    kap = kappa.quadratic_weighted_kappa(human_ratings, machine_ratings)
    acc = (float(correct) / float(k)) * 100

    stats_file.write(_label + "\n")
    stats_file.write("human ratings" + "\n")
    stats_file.write(str(human_ratings) +  "\n")
    stats_file.write("machine ratings" + "\n")
    stats_file.write(str(machine_ratings) + "\n")
    stats_file.write(str(kap) + "\n")
    stats_file.write(str(acc) + "\n\n\n")

    print(_label)
    print("kappa: %s" % kap)
    print("acc: %%%.2f\n" % acc)

def main():

  if(len(sys.argv) == 2):
    if str(sys.argv[1]) == "subset":
      _set = "subset"
      _csv_test = "test_subset.csv"
    elif str(sys.argv[1]) == "sample":
      _set = str(sys.argv[1])
      _csv_test = "test_sample.csv"
  else:
    _set = "train"
    _csv_test = "train.csv"

  start = time.time()

  train_set = get_data_set("train", _set)
  test_set = get_data_set("test", _set)
  alt_train_set = get_data_set("train", _set, lcn=True)
  alt_test_set = get_data_set("test", _set, lcn=True)

  for k in range(0, 3):
    #RANDOM FOREST
    rf_predictions = model.random_forest(train_set, test_set)
    rf_alt_predictions = model.random_forest(alt_train_set, alt_test_set)
    get_stats(rf_predictions, _csv_test, "Random Forest")
    get_stats(rf_alt_predictions, _csv_test, "Altered Random Forest")
    get_stats(np.array([rf_predictions, rf_alt_predictions]), _csv_test, "Voting Random Forest")

    #EXTRA TREES
    et_predictions = model.extra_trees(train_set, test_set)
    et_alt_predictions = model.extra_trees(alt_train_set, alt_test_set)
    get_stats(et_predictions, _csv_test, "Extra Trees")
    get_stats(et_alt_predictions, _csv_test, "Altered Extra Trees")
    get_stats(np.array([et_predictions, et_alt_predictions]), _csv_test, "Voting Extra Forest")

    #SUPPORT VECTOR MACHINE
    sv_predictions = model.kernel_svm(train_set, test_set)
    sv_alt_predictions = model.kernel_svm(alt_train_set, alt_test_set)
    get_stats(np.array(sv_predictions), _csv_test, "SVM Forest")
    get_stats(np.array(sv_alt_predictions), _csv_test, "Altered SVM Forest")
    get_stats(np.array([sv_predictions, sv_alt_predictions]), _csv_test, "Voting SVM Forest")

    #NAIVE BAYES
    nb_predictions = model.naive_bayes(train_set, test_set)
    nb_alt_predictions = model.naive_bayes(alt_train_set, alt_test_set)
    get_stats(np.array(nb_predictions), _csv_test, "SVM Forest")
    get_stats(np.array(nb_alt_predictions), _csv_test, "Altered SVM Forest")
    get_stats(np.array([nb_predictions, nb_alt_predictions]), _csv_test, "Voting SVM Forest")

    #ALL TREES
    get_stats(np.array([rf_predictions, rf_alt_predictions, et_predictions, et_alt_predictions]), _csv_test, "Trees")
    get_stats(np.array([rf_predictions, rf_alt_predictions, et_predictions, et_alt_predictions, nb_predictions, nb_alt_predictions]), _csv_test, "Trees Bayes")
    get_stats(np.array([rf_predictions, rf_alt_predictions, et_predictions, et_alt_predictions, sv_predictions, sv_alt_predictions]), _csv_test, "Trees SVM")
    get_stats(np.array([sv_predictions, sv_alt_predictions, nb_predictions, nb_alt_predictions]), _csv_test, "SVM Bayes")
    get_stats(np.array([rf_predictions, rf_alt_predictions, et_predictions, et_alt_predictions, sv_predictions, sv_alt_predictions, nb_predictions, nb_alt_predictions]), _csv_test, "ALL")


  end = time.time()
  print("time elapsed: ")
  print(end - start)


if __name__ == '__main__':
  main()
