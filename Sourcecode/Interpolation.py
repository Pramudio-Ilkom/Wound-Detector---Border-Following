import numpy as np
import matplotlib.pyplot as plt
from skimage.util import img_as_float
from skimage.io import imread
from skimage.color import rgb2gray
from scipy.linalg import solve
from scipy.special import binom
from scipy.special import comb
from contour_tracing import ContourTracing
import os
import cv2 as cv2


#ngitung jarak koordinat
def JarakTitik(titikAwal = np.array([0,0]), TitikAkhir = np.array([0,0])):
    return np.linalg.norm(titikAwal - TitikAkhir)

#contoh
#bikin koordinat nya dengan [(x0,y0), (x1,y1), ..., (xn,yn)]
koordinatContoh = np.array([[0,0], [3,4], [-1,4], [-4,0], [-4,-3]])
# contohx = np.array([0, 3, -1, -4, -4])
# contohy = np.array([0, 4, 4, 0, -3])

def ChordLength(koordinat = np.array([0,0])):
    knot = np.empty([len(koordinat)],dtype=float)
    knot[0] = 0.0
    totalJarak = 0

    #nyari d / total jarak
    for i in range(1, len(koordinat)):
        jarak = np.linalg.norm(koordinat[i-1] - koordinat[i])
        totalJarak += jarak
        knot[i] = totalJarak
    
    #nyari knotkord
    for i in range(1, len(koordinat)):
        knot[i] = knot[i]/totalJarak


    #return as KnotVector, group of knots
    return np.array(knot)

def knotGlobalCurve(knots = np.array([0]), degree=0):
    
    #nyari knotVector
    KnotVector = np.empty([len(knots) + degree + 1])
    for i, val in enumerate(KnotVector):
        if(i <= degree): KnotVector[i] = 0 
        elif(i >= len(KnotVector) - degree-1): KnotVector[i] = 1
        else:
            knotvector = 0.0
            for ii in range(i-degree, i):
                knotvector += knots[ii]
            KnotVector[i] = knotvector/degree

    return KnotVector

# cek = ChordLength(koordinatContoh)

# KnotVectorTes = [0, 0, 0, 0, 28/51, 1, 1, 1, 1]

def N(knot = 0.0, i = 0 , degree = 0, KnotVector = np.array([0])):
    if (degree == 0):
        return 1.0 if (KnotVector[i] <= knot < KnotVector[i + 1]) else 0.0
    # elif (i > len(KnotVector)):
    #     return 0
    else:
        #rumusdasar = (((knot - KnotVector[i])/(KnotVector[i+degree] - KnotVector[i])) * N(knot, degree - 1, i)) + (((KnotVector[i + degree + 1] - knot) / (KnotVector[i + degree + 1] - KnotVector[i + 1])) * N(knot, degree - 1, i+1))

        kiri = 0.0 if(KnotVector[i + degree] == KnotVector[i]) else (knot - KnotVector[i]) / (KnotVector[i + degree] - KnotVector[i]) * N(knot, i, degree - 1, KnotVector)
        kanan = 0.0 if(KnotVector[i + degree + 1] == KnotVector[i + 1]) else (KnotVector[i + degree + 1] - knot) / (KnotVector[i + degree + 1] - KnotVector[i + 1]) * N(knot, i + 1, degree - 1, KnotVector)
    
        return kiri + kanan

def GlobalCurveInterpolation(Point = np.array([0,0]), degree = 0):
    Knot = ChordLength(Point)
    KnotVector = knotGlobalCurve(Knot, degree)
    nurb = np.empty([len(Point), len(Point)])
    for x in range(len(Point)):
        for y in range(len(Point)):
            # print("koordinate : ", x, y)
            # print("knot : ", Knot[x])
            nurb[x,y] = N(Knot[x], y, degree, KnotVector)
            # print("nurb [",x, ", ", y, "]", nurb[x,y])
    ## biar 1 di akhir
    nurb[len(Point)-1,len(Point)-1] = 1.0

    # ##plotting point awal
    # pointsplit = np.array_split(Point, 2, axis=1)
    # plt.subplot(2, 1, 1)
    # plt.plot(pointsplit[0],pointsplit[1]) 
    # plt.title("Original Spline") 
    # plt.xlabel("x") 
    # plt.ylabel("y") 


    # ## buat cek hasil rumus dasar
    # os.makedirs(f'SourceCode/nurbtest', exist_ok=True)
    # if os.path.exists('SourceCode/nurbtest/nurb.txt'):
    #     os.remove('SourceCode/nurbtest/nurb.txt')
    # with open(f'SourceCode/nurbtest/nurb.txt', "a") as f:
    #     f.write(
    #         f"{nurb} \n \n"
    #     )
    #     # print(contours.size)

    #hasil interpolasi
    hasil = np.dot(np.linalg.pinv(nurb), Point)

    # os.makedirs(f'SourceCode/nurbtest', exist_ok=True)
    # if os.path.exists('SourceCode/nurbtest/hasil.txt'):
    #     os.remove('SourceCode/nurbtest/hasil.txt')
    # with open(f'SourceCode/nurbtest/hasil.txt', "a") as f:
    #     f.write(
    #         f"{hasil} \n \n"
    #     )
    #     # print(contours.size)

    # ##plotting point terinteroloalsi
    # nurbsplit = np.array_split(hasil, 2, axis=1)
    # plt.subplot(2, 1, 2)
    # plt.plot(nurbsplit[0],nurbsplit[1]) 
    # plt.title("Interpolated Spline") 
    # plt.xlabel("x") 
    # plt.ylabel("y") 
    # plt.show()

    # # memperluas hasil array
    # lonjong = np.empty([len(hasil)*len(Knot),2])
    # #general loop
    # print("start pemanjangan")
    # for i in range(1,len(hasil)):
    #     print("pemanjangan : ", i/len(hasil))
    #     x_difference = hasil[i][0] - hasil[i-1][0]
    #     y_difference = hasil[i][1] - hasil[i-1][1]
    #     for ii in range(len(Knot)-1):
    #         x = hasil[i-1][0] + (x_difference * np.sqrt(Knot[ii]))
    #         y = hasil[i-1][1] + (y_difference * np.sqrt(Knot[ii]))
    #         lonjong[(i*len(hasil)+ii-len(hasil))] = [x,y]
    # #last loop
    # last_x = hasil[0][0] - hasil[len(hasil)-1][0]
    # last_y = hasil[0][1] - hasil[len(hasil)-1][1]
    # for i in range(len(Knot)):
    #     x = hasil[len(hasil)-1][0] + (last_x * Knot[ii])
    #     y = hasil[len(hasil)-1][1] + (last_y * Knot[ii])
    #     lonjong[(len(hasil)+i)] = [x,y]
        
    # return lonjong

    return hasil

# def TesGambar(self):
    
#     # # Load a sample image as floating-point values
#     # gambar = imread("Skripkating/Rizki_Wound_ACM/dataset_3/luka_merah/ready/44.jpg")
#     gambar = imread(self)
#     im_gray = rgb2gray(gambar)
#     img = img_as_float(im_gray)
#     # img = img_as_float(data.camera())

#     # Get the shape of the image
#     rows, cols= img.shape

#     # Convert image to a list of points and parameters
#     points = []

#     for i in range(rows):
#         for j in range(cols):
#             points.append([j, i, img[i, j]])  # Assuming x, y, intensity
            
#     points = np.array(points)

#     return GlobalCurveInterpolation(points, 3)


# d = N(5/17, 4, 3, KnotVectorTes)

# whatever = GlobalCurveInterpolation(koordinatContoh, 3)

# teskontor = np.array([[93, 26], [92, 27], [92, 28], [92, 29], [92, 30], [93, 31], [94, 31], [95, 31], [96, 31], [97, 31], [98, 31], [98, 30], [98, 29], [98, 28], [97, 27], [96, 26], [95, 26], [94, 26]])

bernstein = lambda n, k, t: binom(n,k)* t**k * (1.-t)**(n-k)

def bezier(points, num=400):
    N = len(points)
    t = np.linspace(0, 1, num=num)
    curve = np.zeros((num, 2))
    for i in range(N):
        curve += np.outer(bernstein(N - 1, i, t), points[i])
    return curve

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i


def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)
    
    xvals = xvals.reshape(-1,1)
    yvals = yvals.reshape(-1,1)
    result = np.concatenate((xvals,yvals),axis = 1)

    return result

def tes_contour(image,save_as):
    # delete existing file
    if os.path.exists(save_as):
        os.remove(save_as)


    # gambar = imread("Skripkating/Rizki_Wound_ACM/dataset_3/luka_merah/ready/44.jpg")
    # image read for contour tracing
    gambar = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    # im_gray = rgb2gray(gambar)
    # img = img_as_float(im_gray)
    print(" start contourtracing ")
    contours = ContourTracing().findCountourCustom(gambar)
    print(" contour tracing done ")
    # os.makedirs(f'SourceCode/nurbtest', exist_ok=True)
    # if os.path.exists('SourceCode/nurbtest/kontor.txt'):
    #     os.remove('SourceCode/nurbtest/kontor.txt')
    # with open(f'SourceCode/nurbtest/kontor.txt', "a") as f:
    #     f.write(
    #         f"{contours} \n \n"
    #     )
    # contour_index = np.zeros([len(contours), 2], dtype=int)
    # print(contour_index, "\n")

    #sort contours
    # for i in range(len(contours)):
    #     contour_index[i] = [str(i), len(contours[i])]
    # # print(contour_index, "\n")
    # split_sorted_contour = (contour_index[:, 0], contour_index[:, 1])
    # index_sorted_contour = np.lexsort(split_sorted_contour)
    # sorted_contour = contour_index[index_sorted_contour]
    # sorted_contour = sorted_contour[::-1]
    # print(sorted_contour)
    # for i in range(contour_count):
    #     print(sorted_contour[i][0])
    #tes swap
    # temp = contours[0]
    # contours[0] = contours[2]
    # contours[2] = temp

    #putting image
    image_width = gambar.shape[1]
    image_height = gambar.shape[0]
    gambar = imread(image)
    img = img_as_float(gambar)
    # plt.imshow(img)

    #plotting contours
    # for i in range(contour_count):
    # for i in range(len(contours)):
    #     if(len(contours[i]) > 50):
    #         print("penghitungan", i/len(contours))
    #         cX = np.empty(0)
    #         cY = np.empty(0)
    #         # cXY = np.empty([len(contours[i]),2])
    #         for ii in range(len(contours[i])):
    #             splitar = np.array_split(contours[i][ii], 2, axis=1)
    #             cX = np.append(cX, splitar[0])
    #             cY = np.append(cY, splitar[1])
    #         # for ii in range(len(contours[sorted_contour[i][0]])):
    #         #     splitar = np.array_split(contours[sorted_contour[i][0]][ii], 2, axis=1)
    #         #     cX = np.append(cX, splitar[0])
    #         #     cY = np.append(cY, splitar[1])
    #         #     # cXY = np.append(cXY, [cX, cY])

    #         #plot contour tracing
    #         plt.plot(cX,cY)     

    #         #plot interpolated
    #         cXy = cX.reshape(-1,1)
    #         cxY = cY.reshape(-1,1)
    #         cXY = np.concatenate((cXy,cxY),axis = 1)
    #         print(len(cXY))
    #         # print("cXY = " ,cXY, "\n")
    #         print("start interpolation")
    #         interpolated = GlobalCurveInterpolation(cXY, 3)
    #         # print(interpolated)
    #         print("interpolation done!")
    #         nurbsplit = np.array_split(interpolated, 2, axis=1)
    #         plt.plot(nurbsplit[0],nurbsplit[1]) 
    # # plt.show()
    # plt.savefig(save_as)
    # plt.clf()
    # print("file saved!")
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
                interpolated = GlobalCurveInterpolation(cXY, 3)
                print("interpolation done!")
                if(tipe ==0 or tipe == 2 or tipe == 5):
                    nurbsplit = np.array_split(interpolated, 2, axis=1)
                    nurbsplit[0] = np.append(nurbsplit[0],nurbsplit[0][0])
                    nurbsplit[1] = np.append(nurbsplit[1],nurbsplit[1][0])
                    borderchecker(nurbsplit[0], w)
                    borderchecker(nurbsplit[1], h)
                    ax.plot(nurbsplit[0],nurbsplit[1],'k')
                if(tipe == 0 or tipe == 3 or tipe == 6):
                    curved = bezier(interpolated, 400)
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
            tes_contour(image, save_at + "/" + str(i))

def selectivebatchcontour(directory, save_at, imageindex):
    for i in range(len(imageindex)):
        image = directory + "/" + str(imageindex[i]) + ".jpg"
        if os.path.exists(image):
            print("Start scanning : ", i, "/", len(imageindex))
            tes_contour(image, save_at + "/" + str(i))

def tesrawglobalinterpolation(Points, degree, save_as):
    splitpoint = np.array_split(Points, 2, axis=1)
    plt.plot(splitpoint[0],splitpoint[1])
    contour = GlobalCurveInterpolation(Points,degree)
    splitcontour = np.array_split(contour, 2, axis=1)
    plt.plot(splitcontour[0],splitcontour[1])
    curved = bezier(contour, 100)
    curvesplit = np.array_split(curved, 2, axis=1)
    plt.plot(curvesplit[0], curvesplit[1])
    plt.savefig(save_as)
    plt.clf()


# tesbatch = batchcontour("Skripkating/Rizki_Wound_ACM/dataset_3/luka_merah/ready", "SourceCode/imagerawest", 44)

# tesbatchmerah = batchcontour("SourceCode/dataset/luka_merah", "SourceCode/Imagetest6/luka_merah/tes", 44)

tesgambargembrot = tes_contour("SourceCode/dataset/luka_merah/33.jpg", "SourceCode/TesImagegembrot")

# tesbatchkuning = batchcontour("SourceCode/dataset/luka_kuning", "SourceCode/Imagetest6/luka_kuning", 42)

# tesbatchhitam = batchcontour("SourceCode/dataset/luka_hitam", "SourceCode/Imagetest6/luka_hitam", 41)

# tesbatch3 = selectivebatchcontour("SourceCode/dataset", "SourceCode/Imagetest3", [44, 12, 14, 17, 18, 20, 22, 23, 24, 30, 31, 33, 35, 36, 38, 39, 8])

# tesbatch4 = selectivebatchcontour("SourceCode/dataset", "SourceCode/Imagetest4", [44, 12, 14, 17, 18, 20, 22, 23, 24, 30, 31, 33, 35, 36, 38, 39, 8])

# teskontorr = tes_contour("SourceCode/dataset/luka_merah/44.jpg", "SourceCode/Imagetest3/44tes5")

# kontortes = tes_contour("SourceCode/TesImage.jpg", "SourceCode/TesImageT.jpg")

# tegambargembrotdipotong = tes_contour("SourceCode/gambargembrotdipotong.jpg", "SourceCode/TesImagegembrotdipotong.svg")

# tesraw = tesrawglobalinterpolation(koordinatContoh, 5, "SourceCode/teskoordinatsimpel/degree_5.svg")

tegambargembrot = tes_contour("SourceCode/dataset/luka_merah/2.jpg", "SourceCode/TesImagegembrot")




# def knotlocalsurface(self, koordinat = np.array([0,0,0])):

#     #split coordinate
#     splitcoordinate = np.array_split(koordinat, 3, axis=1)

#     #coordinate for u
#     coordinate_k = np.stack((splitcoordinate[0], splitcoordinate[2]), axis=1)
#     CLK = ChordLength(coordinate_k)
#     KnotVecotrK = np.empty([(len(CLK)*2)+4],dtype=float)
#     KnotVecotrK[0], KnotVecotrK[1], KnotVecotrK[2], KnotVecotrK[3] = 0
#     KnotVecotrK[(len(CLK)*2)+1], KnotVecotrK[(len(CLK)*2)+2], KnotVecotrK[(len(CLK)*2)+3], KnotVecotrK[(len(CLK)*2)+4] = 1
#     for k in range(CLK):
#         if(0 < k < len(CLK)):
#             KnotVecotrK[(2 * k) + 2], KnotVecotrK[(2 * k) + 3] = CLK[k]

#     #coordinate for v
#     coordinate_k = np.stack((splitcoordinate[1], splitcoordinate[2]), axis=1)

# print(d)

