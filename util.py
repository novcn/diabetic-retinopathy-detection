import skimage
import skimage.transform
import skimage.io

from scipy import ndimage

import numpy as np
np.set_printoptions(threshold=np.nan)

# img = skimage.io.imread("test.jpeg")

"""
img_name: name of the image to load
mode: <test> or <train> or <validate>
data_set: the data_set to use
"""
def im_load(img_name, mode, data_set, normalise=True):
  path = "data/%s/%s/%s.jpeg" % (mode, data_set, img_name)
  print "loading %s" % path
  img = skimage.io.imread(path)
  if normalise:
    img = img.astype('float32') / 255.0 # normalise and convert to float
  return img

def im_save(img_name, img, data_set):
  path = "data/altered/%s/%s.jpeg" % (data_set, img_name)
  skimage.io.imsave(path, img)

def im_rescale(img):
  return skimage.transform.rescale(img, 0.1)
  # size_x = img.shape[0]
  # size_y = img.shape[1]

  # if(size_x > size_y):
  #   new_x = 128 #TODO, maybe if this value were larger it would increase accuracy?
  #   new_y = (new_x * size_y) / size_x
  # else:
  #   new_y = 128
  #   new_x = (new_y * size_x) / size_y

  #is this better or is it better to rescale?
  ## Also, what weould happen if you were to fist crop the image in one dimension and then
  ## resize to that size on both directions
  # return skimage.transform.resize(img, (size_x, size_y))
    # 128 / 3888 * 2592

  #print("size_x: %d size_y: %d new_x: %d new_y: %d" % (size_x, size_y, new_x, new_y))

def im_resize(img, dimension=128):
  return skimage.transform.resize(img, (256, 256))




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
  size_x = img.shape[1]
  size_y = img.shape[0]
  length = len(img[0])

  r = 0  
  for l in range(0, length):
    r = length - l
    if( (size_x - l*2) <= size_y):
      break
    # l_total = np.add.reduce(img[mid][l])
    # r_total = np.add.reduce(img[mid][r])
    # if( total > thresh*3 ):
    #   break

  # for r in range(length-1, 0, -1):
  #   total = np.add.reduce(img[mid][r])
  #   if( total > thresh*3 ):
  #     break


  # diff = l - r
  # top = mid - (diff // 2)
  # bot = mid + (diff // 2)

  # print("mid: %d size_x: %d size_y: %d t: %d b: %d l_side: %d r_side: %d" % (mid, size_x, size_y, t, b, l_side, r_side))
  #print("r: %d l: %d x: %d y: %d" % (r, l, size_x, size_y))
  return img[0:,l:r]

# def im_crop(img, thres=0.078):
#   size_x = img.shape[0]
#   size_y = img.shape[1]
#   mid = size_y // 2
#   length = len(img[0])




def im_lcn(img, sigma_mean=3, sigma_std=99):
  """
  based on matlab code by Guanglei Xiong, see http://www.mathworks.com/matlabcentral/fileexchange/8303-local-normalization
  """
  means = ndimage.gaussian_filter(img, sigma_mean)
  img_centered = img - means
  stds = np.sqrt(ndimage.gaussian_filter(img_centered**2, sigma_std))
  return img_centered / stds
