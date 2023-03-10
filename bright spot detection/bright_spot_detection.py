from imutils import contours
from skimage import measure
import numpy as np
import imutils
import cv2
import math

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

name = input("Image name: ")
path = "Images/" + name
image = cv2.imread(path)
image = cv2.bitwise_not(image)
image_size_x, image_size_y, _ = image.shape
greyscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(greyscale, (11,11), 0)
cv2.imshow("Invert", image)

brightness = []
for x in range(image_size_x):
    col = []
    for y in range(image_size_y):
        col.append(image[y][x][0])
    brightness.append(col)

x = np.linspace(0,image_size_x, image_size_x)
y = np.linspace(0,image_size_y, image_size_y)
X,Y = np.meshgrid(x,y)
Z = np.array(brightness)

threshold = cv2.threshold(blurred,150,255,cv2.THRESH_BINARY)[1]
cv2.imshow("Blurred image", threshold)

labels = measure.label(threshold, background=0)
mask = np.zeros(threshold.shape, dtype="uint8")


for label in np.unique(labels):
    if label == 0:
        continue
    label_mask = np.zeros(threshold.shape, dtype="uint8")
    label_mask[labels == label] = 255
    num_pixels = cv2.countNonZero(label_mask)

    if num_pixels > 20 and num_pixels < 300:
        mask = cv2.add(mask,label_mask)

cont = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cont = imutils.grab_contours(cont)
cont = contours.sort_contours(cont)[0]

data_file = open("output/{}.csv".format(name), "w")
data_file.write("Bright spot, x , y, L, D, Theta\n")
for(i,c) in enumerate(cont):
    (x,y,w,h) = cv2.boundingRect(c)
    ((cX, cY), radius) = cv2.minEnclosingCircle(c)
    cv2.circle(image, (int(cX), int(cY)), int(radius),(0, 0, 255), 3)
    
    pos_x = x 
    pos_y = y 
    if(pos_x > (image_size_x)/2):
        pos_x = pos_x - ((image_size_x) / 2)
    else:
        pos_x = ((image_size_x) / 2) - pos_x
    
    if(pos_y > (image_size_y )/2):
        pos_y = pos_y - ((image_size_y)/2)
    else:
        pos_y = ((image_size_y)/2) - pos_y
    
    formula = "=0.5 * ATAN(D{}/E{})".format(i+2, i+2)
    cv2.putText(image, "#{}".format(i+1), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
    print("#{} : ({},{})".format(i+1,pos_x,pos_y))
    data_file.write("#{},{},{},{},,{},\n".format(i+1, pos_x, pos_y, math.sqrt((pos_x * 96E-9)**2 + (pos_y*96E-9)**2), formula))

cv2.imshow("Final Image", image)
data_file.close()
cv2.imwrite("output/{}".format(name), image)
cv2.destroyAllWindows()

#fig = plt.figure()
#ax = plt.axes(projection='3d')
#ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
#ax.view_init(90,0)
#plt.show()
#plt.savefig("output/Figures/{}".format(name))
