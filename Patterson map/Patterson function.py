import numpy as np
import scipy
import math
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits import mplot3d
from matplotlib import cm as cmx

import statistics

h_range = []
k_range = []
l_range = []

plane_directory = {}

class plane:
    def __init__(self, h_num : int, k_num : int, l_num : int, theta : float, intensity : float) -> None:
        self.h = h_num
        self.k = k_num
        self.l = l_num
        self.I = intensity
        self.F_squared = 0
        self.bragg_angle = theta
        self.L = 1
        self.A = 1
        self.rho = 1
        self.K = 1

    def calc_L(self):
        self.L = 1 / math.sin(2*self.bragg_angle)
    def calc_rho(self):
        self.rho = (1+math.cos(2*self.bragg_angle)**2)/2
    def calf_F_squared(self):
        self.F_sqaured = self.I / (self.K * self.A * self.L * self.rho)
    def get_hkl(self):
        return self.h, self.k, self.l
    
def patterson_function(diffraction_plane : plane):
    F = diffraction_plane.F_sqaured
    h,k,l = diffraction_plane.get_hkl()
    return lambda u,v,w : F * math.cos(2*math.pi*(h*u + k*v + l*w))

def get_patterson_function(planes) -> list:
    global h_range
    global l_range
    global k_range
    global plane_directory

    electron_density_map = []

    for h in range(h_range[0], h_range[1]+1):
        for k in range(k_range[0], k_range[1]+1):
            for l in range(l_range[0], l_range[1]+1):
                plane_code = "{}{}{}".format(h,k,l)
                index = plane_directory.get(plane_code)
                if(index == None):
                    continue
                else:
                    electron_density_map.append(patterson_function(planes[index]))

    return electron_density_map


def load_pattern():
    file_path = "Patterns\\" + input("Excel file name: ") + ".xlsx"
    file = pd.read_excel(file_path)

    excel_ds = pd.DataFrame(file)
    columns = ["h","k","l","Theta","I"]
    data = []
    for col in columns:
        data_2 = []
        col_data = excel_ds[col]
        for i in range(0,len(col_data)):
            data_2.append(float(col_data[i]))
        data.append(data_2)
    return data

def sort_into_planes() -> list:
    global h_range
    global k_range
    global l_range
    global plane_directory

    h,k,l,theta,I = load_pattern()
    planes = []
    for i in range(len(h)):
        lattice_plane = plane(int(h[i]),int(k[i]),int(l[i]),theta[i], I[i])
        lattice_plane.calc_L()
        lattice_plane.calc_rho()
        lattice_plane.calf_F_squared()
        planes.append(lattice_plane)
        
        plane_code = "{}{}{}".format(int(h[i]), int(k[i]), int(l[i]))
        plane_directory[plane_code] = i
    
    h = sorted(h)
    k = sorted(k)
    l = sorted(l)

    h_range.append(int(h[0]))
    h_range.append(int(h[-1]))
    k_range.append(int(k[0]))
    k_range.append(int(k[-1]))
    l_range.append(int(l[0]))
    l_range.append(int(l[-1]))

    return planes

def evaluate_patterson_function(p_function, u, v, w):
    value = 0.0
    for i in range(0,len(p_function)):
        value = value + p_function[i](u,v, w)

    return(value)

def main():
    plane_list = sort_into_planes()
    function = get_patterson_function(plane_list)
    
    u = np.linspace(-0.7,0.7,200)
    v = np.linspace(-0.7,0.7,200)
    w = np.linspace(0,0,1)
    U,V= np.meshgrid(u,v)
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    cm = plt.get_cmap('jet')

    for x in range(0,len(w)):
        W =[]
        for i in range(0,len(u)):
            W.append([])
            for e in range(0,len(v)):
                W[i].append(w[x])
        W = np.array(W)

        data = []
        data_2 = []
        i_index = 0
        for i in range(0,len(u)):
            data.append([])
            for e in range(0,len(v)):
                data[i_index].append(evaluate_patterson_function(function,(u[i]), (v[e]), (w[x])))
                data_2.append(data[i_index][e])
            i_index = i_index + 1

        mode = max(set(data_2), key=data_2.count)
        #mode = statistics.median(data_2)
        #mode = statistics.mean(data_2)
        #removed = 0
        #removed_uv = []
        #U = []
        #V = []
        #for i in range(len(data)):
        #    removed_uv.append(0)
        #    U.append(u)
        #    V.append(v)
        #for i in range(len(data_2)):
        #    if data_2[i-removed] < mode:
        #        data_2.pop(i-removed)
        #        x = i%len(u)
        #        y = int((i-x)/len(u))
        #        U[y] = np.delete(U[y],x-removed_uv[y])
        #        V[y] = np.delete(V[y],x-removed_uv[y])
        #        removed_uv[y] = removed_uv[y] + 1
        #        removed = removed + 1
        #        #v = np.delete(v,i-removed)

        cNorm = matplotlib.colors.Normalize(vmin=min(data_2), vmax=max(data_2))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        color = scalarMap.to_rgba(data_2)
        #mode = cNorm.vmax / 2
        divide = 1/(mode - cNorm.vmin)
        min_val = cNorm.vmin
        index = 0
        for pixel in color:
            val = data_2[index]
            if val >= mode:
                pixel[3] = 1
            else:
                pixel[3] = (val-min_val)*divide
                #pixel[3] =0
            index = index + 1
        ax.scatter(U, V, W, c=color)
        #data = np.array(data)
        #ax.contour3D(U,V, data, 20, cmap='plasma')
        #ax.plot_surface(U,V, data, facecolors=cm.Oranges(data))
        #ax.scatter(u,v, w[x], c = data, cmap = cm.Oranges(data))
    ax.view_init(90,0)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('3D contour')
    plt.savefig("LiF 1.png")
    plt.show()
    


main()