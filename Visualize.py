from __future__ import division

import scipy.misc
import numpy as np


def save_images(images, size, image_path):
  return imsave(images, size, image_path)


def imsave(images, size, path):
  image = np.squeeze(merge(images, size))
  print(path)
  return scipy.misc.imsave(path, image)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]

    if images.shape[3] in (3, 4):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
          i = idx % size[1]
          j = idx // size[1]
          img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img

    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
          i = idx % size[1]
          j = idx // size[1]
          img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
        return img

    else:
        raise ValueError('in merge(images,size) images parameter '
                     'must have dimensions: HxW or HxWx3 or HxWx4')


def image_manifold_size(num_images):
    manifold_h = int(np.floor(np.sqrt(num_images)))
    manifold_w = int(np.ceil(np.sqrt(num_images)))
    assert manifold_h * manifold_w == num_images
    return manifold_h, manifold_w


def closest_divisor(number):
    divisors = []
    t_num = int(number / 2)

    divisors.append(number)
    while t_num >= 1:
        if number % t_num == 0:
            divisors.append(t_num)
        t_num -= 1

    if np.mod(len(divisors), 2) == 1:
        w, h = divisors[len(divisors)//2], divisors[len(divisors)//2]
    else:
        w, h = divisors[len(divisors)//2-1], divisors[len(divisors)//2]
    return w, h
