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
      _train = "subset"
    elif str(sys.argv[1]) == "sample":
      _train = str(sys.argv[1])
  else:
    _train = "train"

  start = time.time()
  # samples = [b]
  features = []
  levels = []
  with open('data/%s.csv' % (_train)) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      try:
        img = util.im_load(row["image"], _train)
      except IOError:
        print("warning %s not found " % (row["image"]))
        continue

      # print("rescaling image...")
      # img = util.im_rescale(img)
      print("finding mid...")
      mid = util.find_mid(img)
      print("cropping image...")
      img = util.im_crop(img, mid)
      # img = util.im_rescale(img)
      img = util.im_resize(img)

      im_len = np.prod(img.shape)
      img = np.reshape(img, im_len)
      print("feature count: %d" % len(img))
      features.append(img)
      levels.append(row["label"])

    #   # print("saving image...")
    #   # util.im_save(row["image"], img, _train)


    # # print(features[0])
    features = np.vstack(features)
    model.random_forest(features, levels)



      # print("saving image...")
      # util.im_save(row["image"], img, _train)


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