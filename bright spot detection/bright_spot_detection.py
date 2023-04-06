from imutils import contours
from skimage import measure
import numpy as np
import imutils
import cv2
import math
import os
from scipy import signal as spsg

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

max_brightness = None

def brightness_of_region(x,y,z_data,size):
    sample = []
    sample_y = z_data[y-size: y + size]
    for x_data in sample_y:
        sample.append(x_data[x-size: x + size])
    avg_brightness = 0
    for i in sample:
        avg_brightness = avg_brightness + sum(i)
    avg_brightness = avg_brightness / (size*2)**2

    for i in range(0,len(sample)):
        for e in range(0,len(sample)):
            if(sample[i][e] < avg_brightness):
                sample[i][e] = 0

    return sample

def image_threshold(size, z_data):
    image_brightness = np.zeros((500,500,3),dtype="uint8")

    size = 5
    for y in range(size,500,size):
        for x in range(size,500,size):
            sample = brightness_of_region(x,y,z_data,size)
            for i in range(0,len(sample)):
                for e in range(0,len(sample[i])):
                    image_brightness[y-size + i][x-size + e][0] = sample[i][e]
                    image_brightness[y-size + i][x-size + e][1] = sample[i][e]
                    image_brightness[y-size + i][x-size + e][2] = sample[i][e]

    return image_brightness

def get_relative_brightness(x,y,z_data, radius):
    global max_brightness
    max_x, min_x = x + radius, x - radius
    max_y, min_y = y + radius, y - radius

    max_x = math.ceil(max_x)
    max_y = math.ceil(max_y)
    min_x = math.floor(min_x)
    min_y = math.floor(min_y)

    sample = z_data[min_x:max_x]
    for i in range(len(sample)):
        sample[i] = sample[i][min_y:max_y]

    data = []

    for y in range(len(sample)):
        data.append([])
        for x in range(len(sample[y])):
            x_dist,y_dist = abs(x - radius), abs(y-radius)
            dist = (x_dist**2) + (y_dist**2)
            if (dist**0.5) > radius:
                continue
            data[y].append(sample[y][x])
    
    data_points = 0
    sum = 0
    for i in range(len(sample)):
        for e in range(len(sample[i])):
            sum = sum + sample[i][e]
            data_points = data_points + 1

    if(data_points == 0):
        return 0
    
    intensity = sum/data_points
    relative_intensity = 0 
    if(max_brightness == None):
        max_brightness = intensity
        relative_intensity = 1
    else:
        relative_intensity = intensity / max_brightness
    return relative_intensity


path = "Images"
possible_files = os.listdir(path)
images = []
for x in possible_files:
    if(os.path.splitext(x)[-1].lower() == ".png"):
        images.append(x)

for x in images:
    name = x
    path = "Images/" + name
    image = cv2.imread(path)
    image = cv2.bitwise_not(image)
    image_size_x, image_size_y, _ = image.shape
    #greyscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #denoised = cv2.fastNlMeansDenoising(greyscale, None,15,14,21)
    #cv2.imshow("Denoised", denoised)
   # blurred = cv2.GaussianBlur(greyscale, (11,11), 0)
    cv2.imshow("Invert", image)

    brightness = []
    for x in range(image_size_x):
        col = []
        for y in range(image_size_y):
            col.append(image[x][y][0])
        brightness.append(col)

    avg_brightness = 0
    for row in brightness:
        avg_brightness = avg_brightness + sum(row)
    avg_brightness = avg_brightness / (500*500)

    x = np.linspace(0,image_size_x, image_size_x)
    y = np.linspace(0,image_size_y, image_size_y)
    X,Y = np.meshgrid(x,y)
    Z = np.array(brightness)

    threshold = image_threshold(20,brightness)
    cv2.imshow("pass_1", threshold)
    #threshold = image_threshold(100, threshold)
    #cv2.imshow("pass_2", threshold)

    greyscale = cv2.cvtColor(threshold, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(greyscale, (11,11), 0)

    w_threshold = cv2.threshold(blurred, avg_brightness, 255, cv2.THRESH_BINARY_INV)[1]
    cv2.imshow("", w_threshold)
    threshold = cv2.bitwise_not(w_threshold)


    #w_threshold = cv2.threshold(blurred, avg_brightness, 255, cv2.THRESH_BINARY_INV)[1]
    #cv2.imshow("", w_threshold)
    #threshold = cv2.threshold(w_threshold,90,255,cv2.THRESH_BINARY)[1]
    #threshold = cv2.bitwise_not(w_threshold)

    cv2.imshow("Blurred image", blurred)

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

    data_file = open("output_2/{}.csv".format(name), "w")
    data_file.write("Bright spot, x , y, L, D, Theta, sin(theta), avg_intensity \n")

    print(get_relative_brightness(250,250, brightness,20))

    for(i,c) in enumerate(cont):
        (x,y,w,h) = cv2.boundingRect(c)
        ((cX, cY), radius) = cv2.minEnclosingCircle(c)
        cv2.circle(image, (int(cX), int(cY)), int(radius),(0, 0, 255), 3)
        r_brightness = get_relative_brightness(cX, cY, brightness, radius)

        pos_x = x 
        pos_y = y 
        if(pos_x > (image_size_x)/2):
            pos_x = pos_x - ((image_size_x) / 2)
        else:
            pos_x = -(((image_size_x) / 2) - pos_x)

        if(pos_y < (image_size_y )/2):
            pos_y = -(pos_y - ((image_size_y)/2))
        else:
            pos_y = (((image_size_y)/2) - pos_y)

        formula = "=0.5 * ATAN(D{}/E{})".format(i+2, i+2)
        formula_2 = "=SIN(F{})".format(i+2)
        cv2.putText(image, "#{}".format(i+1), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        print("#{} : ({},{})".format(i+1,pos_x,pos_y))
        data_file.write("#{},{},{},{},,{},{},{},\n".format(i+1, pos_x, pos_y, math.sqrt((pos_x * 96E-3)**2 + (pos_y*96E-3)**2), formula, formula_2, r_brightness))

    cv2.imshow("Final Image", image)
    data_file.close()
    cv2.imwrite("output/{}".format(name), image)
    cv2.destroyAllWindows()

    #fig = plt.figure()
    #ax = plt.axes(projection='3d')
    #ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='', edgecolor='none')
    #ax.view_init(90,0)
    #plt.show(block = False)
    #plt.savefig("output/Figures/{}".format(name))
