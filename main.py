import util
import csv
import numpy as np
np.set_printoptions(threshold=1000)
import time

def main():


  start = time.time()
  # samples = [b]
  with open('data/subset_train.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      try:
        img = util.im_load(row["name"], test=True)
      except IOError:
        print("Warning %s not found " % (row["name"]))
      
      print("finding mid...")
      mid = util.find_mid(img)
      print("cropping image...")
      cropped = util.im_crop(img, mid)
      print("cropped image: ", cropped)
      print("saving image...")
      util.im_save(row["name"], cropped, test=True)


  end = time.time()
  print("time elapsed: ")
  print(end - start)


if __name__ == '__main__':
  main()