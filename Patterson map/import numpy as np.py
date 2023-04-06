import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# create some fake data
x = y = z = np.arange(-4.0, 4.0, 0.02)
# here are the x,y and respective z values
X, Y, Z= np.meshgrid(x, y, z)
#Z = np.sinc(np.sqrt(X*X+Y*Y))
# this is the value to use for the color
V = np.sin(Y)

# create the figure, add a 3d axis, set the viewing angle
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(45,60)

# here we create the surface plot, but pass V through a colormap
# to create a different color for each patch
#ax.plot_surface(X, Y, Z, facecolors=cm.Oranges(V))
ax.scatter(X,Y,Z, c=cm.Oranges(V))