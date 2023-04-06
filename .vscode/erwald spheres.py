from tkinter import *
import math

main = Tk()
main.title("Erwald spheres")
window_width = 1920
window_height = 1920
window_usable_width = window_height
w = Canvas(main,width=window_width, height=window_height,)

lattice_points = []
points = []
circles = []
intersections = []
point_intersections = []
lattice_point_size = 5
lattice_size = 101
lattice_const = 20

def draw_lattice(lattice_constant, dot_size, size):
    global lattice_points
    size = size - 1
    center_1, center_2 = window_width / 2, window_height / 2 

    center_x = center_1 - (size / 2) * lattice_constant
    center_y = center_2 - (size / 2) * lattice_constant

    for y in range(size+1):
        lattice_points.append([])
        for x in range(size+1):
            x_0 = center_x - dot_size
            y_0 = center_y - dot_size
            x_1 = center_x + dot_size
            y_1 = center_y + dot_size

            lattice_points[y].append([center_x, center_y])

            #w.create_oval(x_0,y_0,x_1,y_1, fill="#000000")
            center_x = center_x + lattice_constant
        center_x = center_1 - (size / 2) * lattice_constant
        center_y = center_y + lattice_constant

def draw_ewald_sphere(radius, angle):
    global circles
    centre_x, centre_y = window_width / 2, window_height / 2

    angle = angle * (math.pi / 180) 
    circle_cx = centre_x + math.cos(angle) * radius
    circle_cy = centre_y + math.sin(angle) * radius
    
    x_0 = circle_cx - radius
    y_0 = circle_cy - radius

    x_1 = circle_cx + radius
    y_1 = circle_cy + radius

    circles.append([circle_cx,circle_cy,radius])

    w.create_oval(x_0,y_0,x_1,y_1, width = 5)
    
def circle_intersect(circle_1, circle_2):
    global intersections

    d = math.sqrt((circle_2[0] - circle_1[0])**2 + (circle_2[1] - circle_1[1])**2)
    if(d > circle_1[2] + circle_2[2]):
        return None
    elif(d < abs(circle_1[2] - circle_2[2])):
        return None
    elif(d == 0 and circle_1[2] == circle_2[2]):
        return None
    else:
        x_0, y_0, r_0 = circle_1[0], circle_1[1], circle_1[2]
        x_1, y_1, r_1 = circle_2[0], circle_2[1], circle_2[2]
        
        a = (r_0**2 - r_1**2 + d**2)/(2 * d)
        h = math.sqrt(r_0**2 - a**2)
        x_2 = x_0 + a * (x_1 - x_0)/d
        y_2 = y_0 + a * (y_1 - y_0)/d
        x_3 = x_2 + h * (y_1 - y_0)/d
        y_3 = y_2 - h * (x_1 - x_0)/d

        x_4 = x_2 - h*(y_1 - y_0)/d
        y_4 = y_2 + h*(x_1 - x_0)/d

        x_3 = round(x_3,3)
        y_3 = round(y_3,3)
        x_4 = round(x_4,3)
        y_4 = round(y_4,3)


        intersections.append([x_3, y_3])
        intersections.append([x_4, y_4])

        #w.create_oval(x_3 - lattice_point_size, y_3-lattice_point_size, x_3 + lattice_point_size, y_3 + lattice_point_size, fill = "red")
        #w.create_oval(x_4 - lattice_point_size, y_4-lattice_point_size, x_4 + lattice_point_size, y_4 + lattice_point_size, fill = "red")



def get_index(x,y):
    num_rows = lattice_size
    num_columns = lattice_size

    center_x, center_y = window_width / 2, window_height / 2 

    if((x- center_x)% lattice_const != 0 or (y-center_y) %lattice_const != 0):
        return None
    
    top = center_y - ((num_rows-1) / 2)*lattice_const
    left = center_x - ((num_columns-1) / 2)*lattice_const

    row = (y - top)/lattice_const
    col = (x - left)/lattice_const

    return [row,col]



def circle_point_intersect(circle):
    global lattice_points
    global lattice_const
    global points


    x_min = int(circle[0]-circle[2])
    y_min = int(circle[1]-circle[2])
    x_max = int(circle[0]+circle[2])
    y_max = int(circle[1]+circle[2])

    sample = []
    
    num_col = int(math.ceil((x_max-x_min)/lattice_const))
    num_row = int(math.ceil((y_max-y_min)/lattice_const))
    index = get_index(x_min,y_min)
    if(index == None):
        return
    row,col = int(index[0]),int(index[1])
    for i in range(0,num_row+1):
        temp = []
        for e in range(0,num_col+1):
            temp.append(lattice_points[row+i][col+e])
        sample.append(temp)

    #for i in range(0, lattice_size):
    #    temp = []
    #    for e in range(0,lattice_size):
    #        temp.append(lattice_points[i][e])
    #    sample.append(temp)

    for i in range(0,len(sample)):
        for e in range(0,len(sample[i])):
            d_2 = ((sample[i][e][0]-circle[0])**2 + (sample[i][e][1]-circle[1])**2)
            if(d_2 == circle[2]**2):
                x,y = sample[i][e]
                points.append([x,y])

def rgb_to_hex(r,g,b):
    return '#%02x%02x%02x' % (r,g,b)

def highlight_intersections():
    global lattice_points
    global intersections
    global lattice_point_size
    global points

    for point in intersections:
        index = get_index(point[0], point[1])
        if(index == None):
            continue
        row,col = int(index[0]),int(index[1])
        if(point[0] == lattice_points[row][col][0]):
            if(point[1] == lattice_points[row][col][1]):
                x = point[0]
                y = point[1]
                points.append([x,y])
    
def draw_dots():
    global points
    points_2 = sorted(points, key = lambda x: x[0])
    #print(points_2)
    
    list_sorted = False
    sub_lists = []
    index = 0
    while(list_sorted == False):
        temp = [] 
        loop_ended = True
        temp.append(points_2[index])
        for i in range(index+1,len(points_2)):
            if(points_2[i][0] != points_2[i-1][0]):
                temp = sorted(temp, key = lambda x : x[1])
                index = i
                loop_ended = False
                break
            else:
                temp.append(points_2[i])
        sub_lists.append(temp)
        if(loop_ended == True):
            index = len(points_2)

        if(index == len(points_2)):
            list_sorted = True

    point_intensity = []
    for list in sub_lists:
        list_sorted = False
        sub_sub_lists = []
        index = 0
        while(list_sorted == False):
            temp = [] 
            temp.append(list[index])
            loop_ended = True
            for i in range(index+1,len(list)):
                if(list[i][1] != list[i-1][1]):
                    temp = sorted(temp, key = lambda x : x[1])
                    index = i
                    loop_ended = False
                    break
                else:
                    temp.append(list[i])
            sub_sub_lists.append(temp)

            if(loop_ended == True):
                index = len(list)

            if(index == len(list)):
                list_sorted = True
        for i in sub_sub_lists:
            point_intensity.append(i)
    #print(point_intensity)

    intensity = []
    for list in point_intensity:
        intensity.append(len(list))
    intensity = sorted(intensity)
    l_intensity = intensity[0]
    h_intensity = intensity[-1]

    steps = h_intensity - l_intensity
    step = 255 / steps

    for point in point_intensity:
        x = point[0][0]
        y = point[0][1]
        r_intensity = len(point) - l_intensity
        hex_code = rgb_to_hex(math.ceil(r_intensity * step), 255 - math.ceil((r_intensity * step)),0)
        if x == window_width/2 or y == window_height/2:
            continue
        w.create_oval(x - lattice_point_size, y - lattice_point_size, x + lattice_point_size, y + lattice_point_size, fill = "red")
    


#main.resizable(False,False)
w.pack()
draw_lattice(lattice_const, lattice_point_size, lattice_size)
for e in range(1,13):
    for i in range(0,4):
        draw_ewald_sphere(e*20, i * 90)

for i in range(0, len(circles)):
    for e in range(i,len(circles)):
        if i == e:
            continue
        else:
            circle_intersect(circles[i], circles[e])
    circle_point_intersect(circles[i])
highlight_intersections()
draw_dots()
 
main.mainloop()

