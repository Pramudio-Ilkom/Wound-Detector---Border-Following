import numpy as np
from scipy.special import binom

class GlobalInterpolation(object):
    def __init__(self, point):
        self.point = point
        self.GlobalCurveInterpolation = GlobalInterpolation.GlobalCurveInterpolation(point)

    def ChordLength(koordinat = np.array([0,0])):
        knot = np.empty([len(koordinat)],dtype=float)
        knot[0] = 0
        totalJarak = 0

        #d / total jarak
        for i in range(len(koordinat)):
            if(i>0):
                jarak = np.linalg.norm(koordinat[i-1] - koordinat[i])
                totalJarak += jarak
                knot[i] = totalJarak
    
        #knotkord
        for i in range(len(koordinat)):
            knot[i] = knot[i]/totalJarak

        #return as KnotVector, group of knots, 1d array
        return np.array(knot)
    
    def knotGlobalCurve(knots = np.array([0]), degree=0):
    
        #knot Vector that used in global interpolation

        KnotVector = np.empty([len(knots) + degree + 1])
        for i, val in enumerate(KnotVector):
            if(i <= degree): KnotVector[i] = 0 
            elif(i >= len(KnotVector) - degree-1): KnotVector[i] = 1
            else:
                knotvector = 0.0
                for ii in range(i-degree, i):
                    knotvector += knots[ii]
                KnotVector[i] = knotvector/degree

        #return as 1d array
        return KnotVector
    
    def N(knot = 0.0, i = 0 , degree = 0, KnotVector = np.array([0])):

        #Basis function that used to count N in interpolation

        if (degree == 0):
            #
            return 1.0 if (KnotVector[i] <= knot < KnotVector[i + 1]) else 0.0
        else:
            
            #count the left side and right side of equation
            kiri = 0.0 if(KnotVector[i + degree] == KnotVector[i]) else (knot - KnotVector[i]) / (KnotVector[i + degree] - KnotVector[i]) * GlobalInterpolation.N(knot, i, degree - 1, KnotVector)
            kanan = 0.0 if(KnotVector[i + degree + 1] == KnotVector[i + 1]) else (KnotVector[i + degree + 1] - knot) / (KnotVector[i + degree + 1] - KnotVector[i + 1]) * GlobalInterpolation.N(knot, i + 1, degree - 1, KnotVector)
        
            return kiri + kanan
        
    def GlobalCurveInterpolation(Point = np.array([0,0]), degree = 0):
        Knot = GlobalInterpolation.ChordLength(Point)
        KnotVector = GlobalInterpolation.knotGlobalCurve(Knot, degree)
        nurb = np.empty([len(Point), len(Point)])
        for x in range(len(Point)):
            for y in range(len(Point)):
                nurb[x,y] = GlobalInterpolation.N(Knot[x], y, degree, KnotVector)

        #making sure the last array
        nurb[len(Point)-1,len(Point)-1] = 1.0

       
        hasil = np.dot(np.linalg.pinv(nurb), Point)

        return hasil
    
    bernstein = lambda n, k, t: binom(n,k)* t**k * (1.-t)**(n-k)

    def bezier(points, num=400):
        N = len(points)
        t = np.linspace(0, 1, num=num)
        curve = np.zeros((num, 2))
        for i in range(N):
            curve += np.outer(GlobalInterpolation.bernstein(N - 1, i, t), points[i])
        return curve