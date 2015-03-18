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
    img = util.im_load(_name, _phase)
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

def get_stats(_predictions, _csv_test, _label, denom=2.7, avg=False):
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
      if avg == False:
        _max = max(votes)
        best = np.where(votes==_max)
        confidence = float(_max) / float(len(votes))
        if(confidence < thresh):
          machine_vote = 0
        else:
          machine_vote = int(best[0][0])
      else:
        machine_vote = int(round(reduce(lambda x, y: x + y, votes) / float(len(votes))))

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
    
def write_results(_predictions, _csv_test, _label, denom=2.7, avg=False):
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


  #EXTRA TREES
  et_predictions = model.extra_trees(train_set, test_set)
  et_alt_predictions = model.extra_trees(alt_train_set, alt_test_set)
  (a_etu, k_etu) = get_stats(et_predictions, _csv_test, "Extra Trees")



  end = time.time()
  print("time elapsed: ")
  print(end - start)


if __name__ == '__main__':
  main()
