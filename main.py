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

def get_stats(_predictions, _csv_test, _label, denom=2.):
  write_results(_predictions, _csv_test, _label, denom)
  if(len(_predictions.shape) == 1):
    _predictions = np.array([_predictions])

  stats_file = open("statistics.txt", 'a')  
  human_ratings = []
  machine_ratings = []
  with open('data/csv/%s' % (_csv_test)) as csvfile:
    thresh = (1. / denom) + .0001
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
    return (acc, kap)
    
def write_results(_predictions, _csv_test, _label, denom=2., avg=False):
  filename = _csv_test.replace(".csv", "").replace("test_", "") + "_" + _label + ".csv"
  filename = filename.replace(" ", "_")
  print("Creating file: %s" % filename)
  csv_file = open(filename, 'w')
  results = "image,level\n"
  if(len(_predictions.shape) == 1):
    _predictions = np.array([_predictions])
  thresh = (1. / denom) + .0001
  votes = np.array([0, 0, 0, 0, 0])
  with open('data/csv/%s' % (_csv_test)) as csvfile:
    reader = csv.DictReader(csvfile)
    k = 0
    for row in reader:
      for predict in _predictions:
        votes[predict[k]] += 1
      _max = max(votes)
      best = np.where(votes==_max)
      confidence = float(_max) / float(len(votes))
      if(confidence < thresh):
        machine_vote = 0
      else:
        machine_vote = int(best[0][0])
      results += row["image"] + "," + str(machine_vote) + "\n"
      k += 1
  csv_file.write(results)



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

  max_a_rfu = -99
  max_k_rfu = -99
  max_a_rfa = -99
  max_k_rfa = -99
  max_a_rfv = -99
  max_k_rfv = -99
  max_a_etu = -99
  max_k_etu = -99
  max_a_eta = -99
  max_k_eta = -99
  max_a_etv = -99
  max_k_etv = -99
  max_a_svu = -99
  max_k_svu = -99
  max_a_sva = -99
  max_a_nbu = -99
  max_k_nbu = -99
  max_k_sva = -99
  max_a_svv = -99
  max_k_svv = -99
  max_a_nba = -99
  max_k_nba = -99
  max_a_nbv = -99
  max_k_nbv = -99
  max_a_am1 = -99
  max_k_am1 = -99
  max_a_am2 = -99
  max_k_am2 = -99
  max_a_am3 = -99
  max_k_am3 = -99
  max_a_am4 = -99
  max_k_am4 = -99
  max_a_am5 = -99
  max_k_am5 = -99

  denoms = []
  val = 1.
  for k in range(0, 31):
    denoms.append(round(val,2))
    val += .1

  for k in range(0, len(denoms)):
    #RANDOM FOREST
    rf_predictions = model.random_forest(train_set, test_set)
    rf_alt_predictions = model.random_forest(alt_train_set, alt_test_set)
    (a_rfu, k_rfu) = get_stats(rf_predictions, _csv_test, "Random Forest", denom=denoms[k])
    max_a_rfu = max(max_a_rfu, a_rfu)
    if(max_a_rfu == a_rfu):
      denom_max_a_rfu = denoms[k]
    max_k_rfu = max(max_k_rfu, k_rfu)
    if(max_k_rfu == k_rfu):
      denom_max_k_rfu = denoms[k]
    (a_rfa, k_rfa) = get_stats(rf_alt_predictions, _csv_test, "Altered Random Forest", denom=denoms[k])
    max_a_rfa = max(max_a_rfa, a_rfa)
    if(max_a_rfa == a_rfa):
      denom_max_a_rfa = denoms[k]
    max_k_rfa = max(max_k_rfa, k_rfa)
    if(max_k_rfa == k_rfa):
      denom_max_k_rfa = denoms[k]
    (a_rfv, k_rfv) = get_stats(np.array([rf_predictions, rf_alt_predictions]), _csv_test, "Voting Random Forest", denom=denoms[k])
    max_a_rfv = max(max_a_rfv, a_rfv)
    if(max_a_rfv == a_rfv):
      denom_max_a_rfv = denoms[k]
    max_k_rfv = max(max_k_rfv, k_rfv)
    if(max_k_rfv == k_rfv):
      denom_max_k_rfv = denoms[k]

    #EXTRA TREES
    et_predictions = model.extra_trees(train_set, test_set)
    et_alt_predictions = model.extra_trees(alt_train_set, alt_test_set)
    (a_etu, k_etu) = get_stats(et_predictions, _csv_test, "Extra Trees", denom=denoms[k])
    max_a_etu = max(max_a_etu, a_etu)
    if(max_a_etu == a_etu):
      denom_max_a_etu = denoms[k]
    max_k_etu = max(max_k_etu, k_etu)
    if(max_k_etu == k_etu):
      denom_max_k_etu = denoms[k]
    (a_eta, k_eta) = get_stats(et_alt_predictions, _csv_test, "Altered Extra Trees", denom=denoms[k])
    max_a_eta = max(max_a_eta, a_eta)
    if(max_a_eta == a_eta):
      denom_max_a_eta = denoms[k]
    max_k_eta = max(max_k_eta, k_eta)
    if(max_k_eta == k_eta):
      denom_max_k_eta = denoms[k]
    (a_etv, k_etv) = get_stats(np.array([et_predictions, et_alt_predictions]), _csv_test, "Voting Extra Forest", denom=denoms[k])
    max_a_etv = max(max_a_etv, a_etv)
    if(max_a_etv == a_etv):
      denom_max_a_etv = denoms[k]
    max_k_etv = max(max_k_etv, k_etv)
    if(max_k_etv == k_etv):
      denom_max_k_etv = denoms[k]

    #SUPPORT VECTOR MACHINE
    sv_predictions = model.kernel_svm(train_set, test_set)
    sv_alt_predictions = model.kernel_svm(alt_train_set, alt_test_set)
    (a_svu, k_svu) = get_stats(np.array(sv_predictions), _csv_test, "SVM Forest", denom=denoms[k])
    max_a_svu = max(max_a_svu, a_svu)
    if(max_a_svu == a_svu):
      denom_max_a_svu = denoms[k]
    max_k_svu = max(max_k_svu, k_svu)
    if(max_k_svu == k_svu):
      denom_max_k_svu = denoms[k]
    (a_sva, k_sva) = get_stats(np.array(sv_alt_predictions), _csv_test, "Altered SVM Forest", denom=denoms[k])
    max_a_sva = max(max_a_sva, a_sva)
    if(max_a_sva == a_sva):
      denom_max_a_sva = denoms[k]
    max_k_sva = max(max_k_sva, k_sva)
    if(max_k_sva == k_sva):
      denom_max_k_sva = denoms[k]
    (a_svv, k_svv) = get_stats(np.array([sv_predictions, sv_alt_predictions]), _csv_test, "Voting SVM Forest", denom=denoms[k])
    max_a_svv = max(max_a_svv, a_svv)
    if(max_a_svv == a_svv):
      denom_max_a_svv = denoms[k]
    max_k_svv = max(max_k_svv, k_svv)
    if(max_k_svv == k_svv):
      denom_max_k_svv = denoms[k]

    #NAIVE BAYES
    nb_predictions = model.naive_bayes(train_set, test_set)
    nb_alt_predictions = model.naive_bayes(alt_train_set, alt_test_set)
    (a_nbu, k_nbu) = get_stats(np.array(nb_predictions), _csv_test, "Naive Bayes", denom=denoms[k])
    max_a_nbu = max(max_a_nbu, a_nbu)
    if(max_a_nbu == a_nbu):
      denom_max_a_nbu = denoms[k]
    max_k_nbu = max(max_k_nbu, k_nbu)
    if(max_k_nbu == k_nbu):
      denom_max_k_nbu = denoms[k]
    (a_nba, k_nba) = get_stats(np.array(nb_alt_predictions), _csv_test, "Altered Naive Bayes", denom=denoms[k])
    max_a_nba = max(max_a_nba, a_nba)
    if(max_a_nba == a_nba):
      denom_max_a_nba = denoms[k]
    max_k_nba = max(max_k_nba, k_nba)
    if(max_k_nba == k_nba):
      denom_max_k_nba = denoms[k]
    (a_nbv, k_nbv) = get_stats(np.array([nb_predictions, nb_alt_predictions]), _csv_test, "Voting Naive Bayes", denom=denoms[k])
    max_a_nbv = max(max_a_nbv, a_nbv)
    if(max_a_nbv == a_nbv):
      denom_max_a_nbv = denoms[k]
    max_k_nbv = max(max_k_nbv, k_nbv)
    if(max_k_nbv == k_nbv):
      denom_max_k_nbv = denoms[k]

    #ALL TREES
    (a_am1, k_am1) = get_stats(np.array([rf_predictions, rf_alt_predictions, et_predictions, et_alt_predictions]), _csv_test, "Trees", denom=denoms[k])
    max_a_am1 = max(max_a_am1, a_am1)
    if(max_a_am1 == a_am1):
      denom_max_a_am1 = denoms[k]
    max_k_am1 = max(max_k_am1, k_am1)
    if(max_k_am1 == k_am1):
      denom_max_k_am1 = denoms[k]
    (a_am2, k_am2) = get_stats(np.array([rf_predictions, rf_alt_predictions, et_predictions, et_alt_predictions, nb_predictions, nb_alt_predictions]), _csv_test, "Trees Bayes", denom=denoms[k])
    max_a_am2 = max(max_a_am2, a_am2)
    if(max_a_am2 == a_am2):
      denom_max_a_am2 = denoms[k]
    max_k_am2 = max(max_k_am2, k_am2)
    if(max_k_am2 == k_am2):
      denom_max_k_am2 = denoms[k]
    (a_am3, k_am3) = get_stats(np.array([rf_predictions, rf_alt_predictions, et_predictions, et_alt_predictions, sv_predictions, sv_alt_predictions]), _csv_test, "Trees SVM", denom=denoms[k])
    max_a_am3 = max(max_a_am3, a_am3)
    if(max_a_am3 == a_am3):
      denom_max_a_am3 = denoms[k]
    max_k_am3 = max(max_k_am3, k_am3)
    if(max_k_am3 == k_am3):
      denom_max_k_am3 = denoms[k]
    (a_am4, k_am4) = get_stats(np.array([sv_predictions, sv_alt_predictions, nb_predictions, nb_alt_predictions]), _csv_test, "SVM Bayes", denom=denoms[k])
    max_a_am4 = max(max_a_am4, a_am4)
    if(max_a_am4 == a_am4):
      denom_max_a_am4 = denoms[k]
    max_k_am4 = max(max_k_am4, k_am4)
    if(max_k_am4 == k_am4):
      denom_max_k_am4 = denoms[k]
    (a_am5, k_am5) = get_stats(np.array([rf_predictions, rf_alt_predictions, et_predictions, et_alt_predictions, sv_predictions, sv_alt_predictions, nb_predictions, nb_alt_predictions]), _csv_test, "ALL", denom=denoms[k])
    max_a_am5 = max(max_a_am5, a_am5)
    if(max_a_am5 == a_am5):
      denom_max_a_am5 = denoms[k]
    max_k_am5 = max(max_k_am5, k_am5)
    if(max_k_am5 == k_am5):
      denom_max_k_am5 = denoms[k]


    max_file = open("maxs.txt", "w")
    max_file.write(
      "max_a_rfu: " + str(max_a_rfu) + "denom: " + str(denom_max_a_rfu) + "\n"
      "max_k_rfu: " + str(max_k_rfu) + "denom: " + str(denom_max_k_rfu) + "\n"
      "max_a_rfa: " + str(max_a_rfa) + "denom: " + str(denom_max_a_rfa) + "\n"
      "max_k_rfa: " + str(max_k_rfa) + "denom: " + str(denom_max_k_rfa) + "\n"
      "max_a_rfv: " + str(max_a_rfv) + "denom: " + str(denom_max_a_rfv) + "\n"
      "max_k_rfv: " + str(max_k_rfv) + "denom: " + str(denom_max_k_rfv) + "\n"
      "max_a_etu: " + str(max_a_etu) + "denom: " + str(denom_max_a_etu) + "\n"
      "max_k_etu: " + str(max_k_etu) + "denom: " + str(denom_max_k_etu) + "\n"
      "max_a_eta: " + str(max_a_eta) + "denom: " + str(denom_max_a_eta) + "\n"
      "max_k_eta: " + str(max_k_eta) + "denom: " + str(denom_max_k_eta) + "\n"
      "max_a_etv: " + str(max_a_etv) + "denom: " + str(denom_max_a_etv) + "\n"
      "max_k_etv: " + str(max_k_etv) + "denom: " + str(denom_max_k_etv) + "\n"
      "max_a_svu: " + str(max_a_svu) + "denom: " + str(denom_max_a_svu) + "\n"
      "max_k_svu: " + str(max_k_svu) + "denom: " + str(denom_max_k_svu) + "\n"
      "max_a_sva: " + str(max_a_sva) + "denom: " + str(denom_max_a_sva) + "\n"
      "max_a_nbu: " + str(max_a_nbu) + "denom: " + str(denom_max_a_nbu) + "\n"
      "max_k_nbu: " + str(max_k_nbu) + "denom: " + str(denom_max_k_nbu) + "\n"
      "max_k_sva: " + str(max_k_sva) + "denom: " + str(denom_max_k_sva) + "\n"
      "max_a_svv: " + str(max_a_svv) + "denom: " + str(denom_max_a_svv) + "\n"
      "max_k_svv: " + str(max_k_svv) + "denom: " + str(denom_max_k_svv) + "\n"
      "max_a_nba: " + str(max_a_nba) + "denom: " + str(denom_max_a_nba) + "\n"
      "max_k_nba: " + str(max_k_nba) + "denom: " + str(denom_max_k_nba) + "\n"
      "max_a_nbv: " + str(max_a_nbv) + "denom: " + str(denom_max_a_nbv) + "\n"
      "max_k_nbv: " + str(max_k_nbv) + "denom: " + str(denom_max_k_nbv) + "\n"
      "max_a_am1: " + str(max_a_am1) + "denom: " + str(denom_max_a_am1) + "\n"
      "max_k_am1: " + str(max_k_am1) + "denom: " + str(denom_max_k_am1) + "\n"
      "max_a_am2: " + str(max_a_am2) + "denom: " + str(denom_max_a_am2) + "\n"
      "max_k_am2: " + str(max_k_am2) + "denom: " + str(denom_max_k_am2) + "\n"
      "max_a_am3: " + str(max_a_am3) + "denom: " + str(denom_max_a_am3) + "\n"
      "max_k_am3: " + str(max_k_am3) + "denom: " + str(denom_max_k_am3) + "\n"
      "max_a_am4: " + str(max_a_am4) + "denom: " + str(denom_max_a_am4) + "\n"
      "max_k_am4: " + str(max_k_am4) + "denom: " + str(denom_max_k_am4) + "\n"
      "max_a_am5: " + str(max_a_am5) + "denom: " + str(denom_max_a_am5) + "\n"
      "max_k_am5: " + str(max_k_am5) + "denom: " + str(denom_max_k_am5) +"\n")




  end = time.time()
  print("time elapsed: ")
  print(end - start)


if __name__ == '__main__':
  main()
