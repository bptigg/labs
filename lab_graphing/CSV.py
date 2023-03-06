import pandas as pd
import os
import sheet


def csv_handling():
    path = input("Folder to read from: ")
    path_1 = "Excel\\" + path

    try:
        possible_files = os.listdir(path_1)
    except:
        print("Directory doesn't exist")
    
    i = 1
    csv_files = []
    for x in os.listdir(path_1):
        if(os.path.splitext(x)[-1].lower() == ".csv"):
            option = int(input("{}) {} : Load yes(1) or no(0): ".format(i,x)))
            if(option == True):
               csv_files.append(x)
        i = i + 1

    return_data = {}
    for x in csv_files:
        csv_file = path_1 + "\\" + x
        csv = pd.read_csv(csv_file)

        csv_ds = pd.DataFrame(csv)
        columns = ['X', 'Y', 'X_U', 'Y_U']
        data = []
        for col in columns:
            data_2 = []
            col_data = csv_ds[col]
            for i in range(1, len(col_data)):
                data_2.append(float(col_data[i]))
            data.append(data_2)
        return_data[x] = sheet.dataframe(x, data)

    return return_data






