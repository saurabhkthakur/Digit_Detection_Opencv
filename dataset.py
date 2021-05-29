import imutils
import numpy as np
import cv2
import mahotas

def load_digits(datasetPath):
    data = np.genfromtxt(datasetPath, delimiter=',', dtype='uint8')

    target = data[:,0]
    data = data[:,1:].reshape(data.shape[0], 28,28)

    return (data, target)

def deskew(image, width):
    (h,w) = image.shape[:2]
    moments = cv2.moments(image)
    print(moments)
    skew = moments['mu11']/ moments['mu02']
    M  = np.float32([
        [1,skew, -0.5*w*skew],
        [0,1,0]
    ])
    image=cv2.warpAffine(image,M,(w,h),flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)

    image = imutils.resize(image, width=width)

    return image

def center_extent(image, size):
    (ew, eh) = size

    if image.shape[1] > image.shape[0]:
        image = imutils.resize(image, width= ew)

    else:
        image = imutils.resize(image, height=eh)

    extent = np.zeros((eh,ew), dtype='uint8')

    offsetX = (ew - image.shape[1])//2
    offsetY = (eh - image.shape[0])//2

    extent[offsetY:offsetY + image.shape[0], offsetX:offsetX + image.shape[1] ] = image
    CM = mahotas.center_of_mass(extent)
    (CY, CX) = np.round(CM).astype('int32')
    (dX, dY) = ((size[0] // 2) - CX, (size[1] // 2) - CY)
    M = np.float32([[1, 0, dX], [0, 1, dY]])
    extent = cv2.warpAffine(extent, M, size)

    return extent





