import numpy as np
import matplotlib.pyplot as plt
from skimage.util import img_as_float
from skimage.io import imread
from contour_tracing import ContourTracing
from Global_Interpolation import GlobalInterpolation
import os
import cv2 as cv2

def SimpleRunner(image,save_as):

    #call image for contour tracing
    gambar = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    print(" start contourtracing ")
    contours = ContourTracing().findCountourCustom(gambar)
    print(" contour tracing done ")

    #putting image for matlib
    image_width = gambar.shape[1]
    image_height = gambar.shape[0]
    gambar = imread(image)
    img = img_as_float(gambar)

    #plotting contours
    translate_contour(contours, img, image_width, image_height, save_as + ".svg", 0)
    translate_contour(contours, img, image_width, image_height, save_as + "_withcontour.svg", 1)
    translate_contour(contours, img, image_width, image_height, save_as + "_withinterpolation.svg", 2)
    translate_contour(contours, img, image_width, image_height, save_as + "_withcurve.svg", 3)
    translate_contour(contours, img, image_width, image_height, save_as + "_justcontour.svg", 4)
    translate_contour(contours, img, image_width, image_height, save_as + "_justinterpolation.svg", 5)
    translate_contour(contours, img, image_width, image_height, save_as + "_justcurve.svg", 6)

def borderchecker(array, limit):
    for i in range(len(array)):
        if (array[i] > limit): i = limit

def translate_contour(contours, img, w, h, save_as, tipe):
    #type 0 = image with raw contour, interpolation, and curved
    #type 1 = image with raw contour
    #type 2 = image with interpolation
    #type 3 = image with curved
    #type 4 = just contour
    #type 5 = just interpolation
    #type 6 = just curve

    DPI = 90 # https://www.infobyip.com/detectmonitordpi.php

    fig = plt.figure(frameon=False)
    fig.set_size_inches(w/DPI,h/DPI)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    #putting image
    if(0 <= tipe < 4):
        ax.imshow(img, aspect='auto')
    #making line
    for i in range(len(contours)):
        if(len(contours[i]) > 50):
            print("penghitungan", i/len(contours))
            cX = np.empty(0)
            cY = np.empty(0)
            for ii in range(len(contours[i])):
                splitar = np.array_split(contours[i][ii], 2, axis=1)
                cX = np.append(cX, splitar[0])
                cY = np.append(cY, splitar[1])
            
            #plot contour tracing
            if(0 <= tipe <2 or tipe == 4):
                cX = np.append(cX, cX[0])
                cY = np.append(cY, cY[0])
                ax.plot(cX,cY,'k')
            if(tipe == 0 or 1 < tipe <= 3 or 4 < tipe <= 6):
                cXy = cX
                cxY = cY
                trigger = 1000
                if(len(cX) > trigger):
                    interval = 0.0
                    difference = len(cX)-trigger
                    interval = len(cX)/difference
                    for i in range(difference - 1, 1, -1):
                        cXy = np.delete(cXy, int(i*interval))
                        cxY = np.delete(cxY, int(i*interval))
                cXy = cXy.reshape(-1,1)
                cxY = cxY.reshape(-1,1)
                cXY = np.concatenate((cXy,cxY),axis = 1)
                print("start interpolation")
                interpolated = GlobalInterpolation.GlobalCurveInterpolation(cXY, 3)
                print("interpolation done!")
                if(tipe ==0 or tipe == 2 or tipe == 5):
                    nurbsplit = np.array_split(interpolated, 2, axis=1)
                    nurbsplit[0] = np.append(nurbsplit[0],nurbsplit[0][0])
                    nurbsplit[1] = np.append(nurbsplit[1],nurbsplit[1][0])
                    borderchecker(nurbsplit[0], w)
                    borderchecker(nurbsplit[1], h)
                    ax.plot(nurbsplit[0],nurbsplit[1],'k')
                if(tipe == 0 or tipe == 3 or tipe == 6):
                    curved = GlobalInterpolation.bezier(interpolated, 400)
                    curvesplit = np.array_split(curved, 2, axis=1)
                    curvesplit[0] = np.append(curvesplit[0],curvesplit[0][0])
                    curvesplit[1] = np.append(curvesplit[1],curvesplit[1][0])
                    borderchecker(curvesplit[0], w)
                    borderchecker(curvesplit[1], h)
                    ax.plot(curvesplit[0],curvesplit[1],'k')

    if(3 < tipe  <= 6):
        ax.plot([0, w, w, 0, 0],[0, 0 , h , h , 0], alpha = 0)
        ax.invert_yaxis()
    fig.savefig(save_as)
    plt.clf()
    print("file "+ str(tipe) + " saved!")

def batchcontour(directory, save_at, image_count):
    for i in range(image_count + 1):
        image = directory + "/" + str(i) + ".jpg"
        if os.path.exists(image):
            print("Start scanning : ", i, "/", image_count + 1)
            SimpleRunner(image, save_at + "/" + str(i))

def selectivebatchcontour(directory, save_at, imageindex):
    for i in range(len(imageindex)):
        image = directory + "/" + str(imageindex[i]) + ".jpg"
        if os.path.exists(image):
            print("Start scanning : ", i, "/", len(imageindex))
            SimpleRunner(image, save_at + "/" + str(i))