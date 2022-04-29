import numpy as np


def erode(img, kernel):
    kernel_center = (kernel.shape[0] // 2, kernel.shape[1] // 2)
    kernel_ones_count = kernel.sum()
    eroded_img = np.zeros((img.shape[0] + kernel.shape[0] - 1, img.shape[1] + kernel.shape[1] - 1))
    img_shape = img.shape

    x_append = np.zeros((img.shape[0], kernel.shape[1] - 1))
    img = np.append(img, x_append, axis=1)

    y_append = np.zeros((kernel.shape[0] - 1, img.shape[1]))
    img = np.append(img, y_append, axis=0)

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            i_ = i + kernel.shape[0]
            j_ = j + kernel.shape[1]
            if kernel_ones_count == (kernel * img[i:i_, j:j_]).sum() / 255:
                eroded_img[i + kernel_center[0], j + kernel_center[1]] = 1

    return eroded_img[:img_shape[0], :img_shape[1]]


def dilate(img, kernel):
    kernel_center = (kernel.shape[0] // 2, kernel.shape[1] // 2)
    kernel_ones_count = kernel.sum()
    dilated_img = np.zeros((img.shape[0] + kernel.shape[0] - 1, img.shape[1] + kernel.shape[1] - 1))
    img_shape = img.shape

    x_append = np.zeros((img.shape[0], kernel.shape[1] - 1))
    img = np.append(img, x_append, axis=1)

    y_append = np.zeros((kernel.shape[0] - 1, img.shape[1]))
    img = np.append(img, y_append, axis=0)

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            i_ = i + kernel.shape[0]
            j_ = j + kernel.shape[1]
            if img[i + kernel_center[0], j + kernel_center[1]] == 255:
                dilated_img[i:i_, j:j_] = 1

    return dilated_img[:img_shape[0], :img_shape[1]]

def open(img, kernel):
    return dilate(erode(img,kernel), kernel)

def close(img, kernel):
    return erode(dilate(img,kernel), kernel)

def bitwise_and(X, Y):
    return np.bitwise_and(np.uint8(X),np.uint8(Y))

def hitmiss(img, kernel):
    kernel_hit = np.zeros((kernel.shape[0], kernel.shape[1]))
    kernel_hit[kernel == 1] = 1

    kernel_miss = np.zeros((kernel.shape[0], kernel.shape[1]))
    kernel_miss[kernel == -1] = 1

    e1 = erode(img,kernel_hit)
    e2 = erode(25 - img,kernel_miss)
    return bitwise_and(e1,e2)
