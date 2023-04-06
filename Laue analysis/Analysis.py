import math
import pandas as pd
from scipy import constants 

max_value = 44
max_m_indicie = 7
pixel_to_mm = 96e-3
voltage = 35e3
x_ray_lambda = 0
plane_seperation = 4.08E-10
name = ""

class plane:
    h = 0
    k = 0
    l = 0
    d = 0
    wavelength = 0.0
    theta = 0.0
    k_l = 0.0
    code = ''
    
    def __init__(self, H : int, K : int, L : int) -> None:
        self.h = H
        self.k = K
        self.l = L
        d = (H**2 + K**2 + L**2)
        self.d = plane_seperation / d
        
        sin_theta = 0
        try:
            sin_theta = abs(self.h) / math.sqrt(d)
            self.theta = math.asin(sin_theta)

            #if(self.k < 0 and self.l >= 0):
            #    self.theta = self.theta + (3*math.pi)/2
            #elif(self.k < 0 and self.l < 0 ):
            #    self.theta = self.theta + math.pi
            #elif(self.k >= 0 and self.l < 0):
            #    self.theta = self.theta + (math.pi)/2

        except:
            self.theta = 0
        try:
            self.k_l = K/L
        except:
            self.k_l = 0
        self.code = "{}{}{}".format(self.h, self.k, self.l)
        self.wavelength = 2*self.d*abs(math.sin(self.theta))


class bright_spot:
    y = 0
    z = 0
    l = 0
    d = 0
    theta = 0.0
    y_z = 0.0
    assigned_plane = ''

    def __init__(self, Y, Z, D) -> None:
        #self.y = Y * pixel_to_mm
        #self.z = Z * pixel_to_mm

        self.y = abs(Y * pixel_to_mm)
        self.z = abs(Z * pixel_to_mm)

        self.d = D
        self.l = math.sqrt((self.y**2 + self.z**2))
        self.theta = 0.5 * math.atan(self.l / self.d)
        #if(Y >= 0 and Z < 0):
            #self.theta = self.theta + (math.pi)/2
        #elif(Y < 0 and Z < 0):
            #self.theta = self.theta + (math.pi)
        #elif(Y < 0 and Z >=0):
            #self.theta = self.theta + 3*(math.pi)/2
        self.y_z = self.y / self.z



def find_all_planes():
    planes = []
    global x_ray_lambda
    x_ray_lambda = (constants.Planck * constants.c)/(1.602e-19 * voltage)
    for h in range(0,max_m_indicie):
        for k in range(0,max_m_indicie):
            for l in range(0,max_m_indicie):
                if((h%2 == 0 and k%2 == 0 and l%2 == 0) or (h%2 == 1 and k%2 == 1 and l%2 == 1)):
                    if(h**2 + k**2 + l**2 <= 44 and h**2 + k**2 + l**2 != 0):
                        temp = [h,k,l]
                        print(temp)
                        temp_plane = plane(h,k,l)
                        #if(temp_plane.wavelength > x_ray_lambda):
                            #if(abs(temp_plane.theta) > (math.pi/18)):
                               # planes.append(plane(h,k,l))
                        planes.append(temp_plane)
    print(len(planes))


    return planes

def load_bright_spots():
    global name
    name = input("Excel filename: ")
    filename = "Spots//" + name + ".csv"
    file = pd.read_csv(filename)
    spots_df = pd.DataFrame(file)
    data = []
    columns = ['x','y','D']
    for col in columns:
        data_2 = []
        col_data = spots_df[col]
        for i in range(0, len(col_data)):
                data_2.append(float(col_data[i]))
        data.append(data_2)

    spots = []
    for i in range(0,len(data[0])):
        spots.append(bright_spot(data[1][i], data[0][i], data[2][0]))
    
    return spots
    

def match_spots_to_planes(spots, planes):
    angle_plane = []
    k_l_plane = []
    planes_list = []
    plane_dict = {}

    spot_plane = []

    index = 0
    for p in planes:
        angle_plane.append([p.theta, p.code])
        k_l_plane.append([p.k_l, p.code])
        planes_list.append(p.code)
        plane_dict[p.code] = index
        index = index + 1

    for spot in spots:
        spot_angle = spot.theta
        residuals_2 = []
        for i in range(0,len(planes)):
            r = abs(angle_plane[i][0] - spot_angle)
            residuals_2.append([r,i])
        
        residuals_2 = sorted(residuals_2, key = lambda x: x[0])

        spot_kl = spot.y_z
        residuals = []
        for i in range(0,len(planes)):
            r = abs(k_l_plane[i][0] - spot_kl)
            residuals.append([r,i])
        
        residuals = sorted(residuals, key = lambda x : x[0])
        index = []
        for i in range(0,len(residuals_2)):
            plane_index = residuals_2[i][1]
            for e in range(0,len(residuals_2)):
                if(residuals[e][1] == plane_index):
                    index.append([i+e,e])
                    break

        
        index = sorted(index)
        plane_index = residuals[index[0][1]][1]

        print(planes_list[plane_index])
        #print(angle_plane[plane_index])
        #print(k_l_plane[plane_index])
        spot_plane.append([spot, planes[plane_index]])


        
    return spot_plane

def main():
    global name
    planes = find_all_planes()
    spots = load_bright_spots()

    spot_planes = match_spots_to_planes(spots, planes)

    data_file = open("Output/{}.csv".format(name + "_planes"), "w")
    data_file.write("Bright spot,Theta,Y/Z,h,k,l, plane_Theta, plane_K/L,\n")

    index = 0
    for spot in spot_planes:
        data_file.write("#{},{},{},{},{},{},{},{},\n".format(index,spot[0].theta, spot[0].y_z,spot[1].h,spot[1].k,spot[1].l,spot[1].theta,spot[1].k_l))
        index = index + 1
    data_file.close()
    
main()