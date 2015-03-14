import os
import util
import model
import csv
import numpy as np
np.set_printoptions(threshold=1000)
import time
import sys

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
  # samples = [b]
  train_set = []
  train_lvls = []
  with open('data/%s.csv' % (_set)) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      try:
        img = util.im_load(row["image"], "train", _set)
      except IOError:
        #print("warning %s not found " % (row["image"]))
        continue

      #print("rescaling image...")
      # img = util.im_rescale(img)
      
      #print("finding mid...")
      mid = util.find_mid(img)
      #print("cropping image...")
      img = util.im_crop(img, mid)
      img = util.im_resize(img)
      im_len = np.prod(img.shape)
      img = np.reshape(img, im_len)

      #print("feature count: %d" % len(img))
      train_set.append(img)
      train_lvls.append(row["label"])

    #print("saving image...")
    #   # util.im_save(row["image"], img, _set)
  test_set = []

  test_files = os.listdir("data/test/%s" % (_set))
  for image in test_files:
    try:
      image = image.replace(".jpeg", "")
      img = util.im_load(image, "test", _set)
    except IOError:
      print("warning %s THIS SHOULD NEVER HAPPEN" % (image))
      continue

    #print("finding mid...")
    mid = util.find_mid(img)
    #print("cropping image...")
    img = util.im_crop(img, mid)
    img = util.im_resize(img)
    im_len = np.prod(img.shape)
    img = np.reshape(img, im_len)

    #print("feature count: %d" % len(img))
    test_set.append(img)


  test_set = np.vstack(test_set)
  predictions = model.random_forest(train_set, train_lvls, test_set)

  with open('data/%s' % (_csv_test)) as csvfile:
    reader = csv.DictReader(csvfile)
    k = 0
    correct = 0
    #total = 0
    for row in reader: 
      print(row["image"])
      print(test_set[k])
      print("comparing: %d =? %s" % (int(predictions[k]), row["level"]))
      if(int(predictions[k]) == int(row["level"])):
        correct += 1
      k += 1

  print("predicts: ")
  print(predictions)
  print("correct: %d" % correct)
  print("total: %d" % k)
  accuracy = (correct / k) * 100
  print("accuracy: %.2f" % accuracy)
    #print("%s : %s" % (test_files[k].replace(".jpeg", ""), predictions[k]))

      #print("saving image...")
      # util.im_save(row["image"], img, _set)


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