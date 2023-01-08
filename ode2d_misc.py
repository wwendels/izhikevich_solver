# misc functions
import numpy as np
import matplotlib.pyplot as plt

def diagramVI(modelclass,xmesh,label):

    # xmin, xmax, dx, xlab = xmesh
    # x = np.arange(xmin, xmax+dx, dx)

    xmin, xmax, _, nx, xlab = xmesh 
    # ymin, ymax, ny, ylab = ymesh
    x = np.linspace(xmin,xmax,nx,endpoint=True)

    # dx, dy = (xmax-xmin)/float(nx-1), (ymax-ymin)/float(ny-1)

    plt.figure()
    plt.plot(x, -modelclass.f(np.array([x,modelclass.ninf(x)]))[0]/modelclass.C)
    plt.xlim([xmin, xmax])
    plt.ylim([-200,200])
    plt.xlabel(xlab)
    plt.ylabel("$I$")
    plt.savefig("VI-diagram_"+label+".png")

def inList(x,X,tol=1e-2):

    n = len(X)
    # m = len(X[0])
    flag = False
    for i in range(n):
        # print(X[i])
        # print(x)
        if np.linalg.norm(np.array(X[i])-np.array(x),2) < tol:
            flag = True

    return flag