import skimage
import skimage.transform
import skimage.io

from scipy import ndimage

import numpy as np
np.set_printoptions(threshold=np.nan)

# img = skimage.io.imread("test.jpeg")

def im_load(img_name, test=False, normalise=True):
  if(test):
    path = "data/raw/subset/%s.jpeg" % (img_name)
  else:
    path = "data/raw/train/%s.jpeg" % (img_name)
  print "loading %s" % path
  img = skimage.io.imread(path)
  if normalise:
    img = img.astype('float32') / 255.0 # normalise and convert to float
  return img

def im_save(img_name, img, test=False):
  if(test):
    path = "data/altered/subset/%s.jpeg" % (img_name)
  else:
    path = "data/altrd/train/%s.jpeg" % (img_name)
  skimage.io.imsave(path, img)



def find_mid(img, thresh=.078):
  size_x = img.shape[0]
  size_y = img.shape[1]
  return size_y // 2
  # size_y = img.shape[1]
  # for h in range(0, size_y):
  #   for w in range(0, size_x):
  #     total = np.add.reduce(img[h][w])
  #     if(total > thresh*3):
  #       return w
  # return -1

def im_crop(img, mid, thresh=0.078):
  size_x = img.shape[0]
  size_y = img.shape[1]
  length = len(img[0])
  
  print("old mid: %d" % mid)

  for l in range(0, length):
    total = np.add.reduce(img[mid][l])
    mthresh = thresh * 3
    if( total > mthresh ):
      break

  for r in range(length-1, 0, -1):
    total = np.add.reduce(img[mid][r])
    if( total > thresh*3 ):
      break

  diff = l - r
  top = mid - (diff // 2)
  bot = mid + (diff // 2)

  # print("mid: %d size_x: %d size_y: %d t: %d b: %d l_side: %d r_side: %d" % (mid, size_x, size_y, t, b, l_side, r_side))
  return img[0:,l:r]


def im_lcn(img, sigma_mean=3, sigma_std=99):
  """
  based on matlab code by Guanglei Xiong, see http://www.mathworks.com/matlabcentral/fileexchange/8303-local-normalization
  """
  means = ndimage.gaussian_filter(img, sigma_mean)
  img_centered = img - means
  stds = np.sqrt(ndimage.gaussian_filter(img_centered**2, sigma_std))
  return img_centered / stds
