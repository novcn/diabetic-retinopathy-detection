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
  # valid_set = get_data_set("valid", _set)

  # test_set_x, test_set_y = shared_dataset(test_set)
  # valid_set_x, valid_set_y = shared_dataset(valid_set)
  # train_set_x, train_set_y = shared_dataset(train_set)

  # datasets = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]

  # convnet.evaluate_lenet5(datasets)  
  max_ = -1
  avg = 0
  i = 1
  predictions = model.random_forest(train_set, test_set)
  alt_predictions = model.random_forest(alt_train_set, alt_test_set)

  print("predicts: ")
  print(predictions)
  human_rate = []
  auto_rate = []
  with open('data/csv/%s' % (_csv_test)) as csvfile:
    reader = csv.DictReader(csvfile)
    k = 0
    correct = 0
    w = 0
    # if(_set != "full"):
    #   for row in reader: 
    #     if(predictions[k] == alt_predictions[k]):
    #       p = predictions[k]
    #     else:
    #       p = 0
    #     human_rate.append(int(row["level"]))
    #     auto_rate.append(p)
    #     # if(p == int(row["levlel"])):
    #     if(p == int(row["level"])):
    #       correct += 1
    #     k += 1

    #   print("correct: %d" % correct)
    #   print("total: %d" % k)
    #   accuracy = (correct * 100) / k
    #   max_ = max(max_, accuracy)
    #   avg += accuracy
    #   print("accuracy: %%%.2f" % accuracy)
    #   quad_kappa = kappa.quadratic_weighted_kappa(human_rate, auto_rate)
    #   avg = avg / i
    #   print("kappa: %s" % quad_kappa)
    #   print("Max: %%%.2f" % max_)
    #   print("Avg: %%%.2f" % avg)

    full_test = "image,level\n"
    # test_files = os.listdir("data/test/%s" % (_set))
    for k in range(0, len(test_files)):
      if(predictions[k] == alt_predictions[k]):
        p = predictions[k]
      else:
        p = 0
      name = test_files[k].replace(".jpeg", "")
      full_test += name + "," + str(p) + "\n"
    test_file = open("test.csv", 'w')
    test_file.write(full_test)


  end = time.time()
  print("time elapsed: ")
  print(end - start)


if __name__ == '__main__':
  main()
