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

def import_image(_name, _phase, _set):
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
  return img


def get_data_set(_phase, _set, save=False):
  imgs = []
  lvls = []

  if(_phase == "test"):
    test_files = os.listdir("data/test/%s" % (_set))
    for image in test_files:
      name = image.replace(".jpeg", "")
      img = import_image(name, _phase, _set)
      if img != []:
        imgs.append(img)
    return imgs
  
  if(_phase == "train" or _phase == "valid"):
    with open('data/csv/%s.csv' % (_set + "_" + _phase)) as csvfile:
      reader = csv.DictReader(csvfile)
      for row in reader:
        img = import_image(row["image"], _phase, _set)
        if img != []:
          imgs.append(img)
        lvls.append(row["level"])

        if(save):
          print("saving image...")
          util.im_save(row["image"], img, _set)

    imgs = np.vstack(imgs)
    set = (imgs, lvls)
    return set
  
  else:
    print("Invalid phase: %s " % (_phase))


def shared_dataset(data_xy, borrow=True):
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')


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
  valid_set = get_data_set("valid", _set)

  test_set_x, test_set_y = shared_dataset(test_set)
  valid_set_x, valid_set_y = shared_dataset(valid_set)
  train_set_x, train_set_y = shared_dataset(train_set)

  datasets = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]


  convnet.evaluate_lenet5(datasets)  

  predictions = model.random_forest(train_set, test_set)

  print("predicts: ")
  print(predictions)

  with open('data/csv/%s' % (_csv_test)) as csvfile:
    reader = csv.DictReader(csvfile)
    k = 0
    correct = 0
    for row in reader: 
      if(int(predictions[k]) == int(row["level"])):
        correct += 1
      k += 1

    print("correct: %d" % correct)
    print("total: %d" % k)
    accuracy = (correct * 100) / k
    print("accuracy: %%%.2f" % accuracy)

  end = time.time()
  print("time elapsed: ")
  print(end - start)


if __name__ == '__main__':
  main()

  # briankrebs
  # rajgoel
  # thedarktangent
  # threatintel
  # _defcon_