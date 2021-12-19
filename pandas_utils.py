import numpy as np
import pandas as pd
import os


def save_run_data(file_name, algorithm_name, quality_list, d, r, N, task):

    rootdir = os.path.dirname(__file__)
    path = rootdir + "/" + file_name + ".csv"
    try:
        df = pd.read_csv(path)
        print("File found, adding the lines to the existent dataset...")
    except:
        df = pd.DataFrame()
        print("File not found, creating a new dataset...")

    columns = ["step_" + str(i) for i in range(len(quality_list))]
    new_row = pd.DataFrame(np.array(quality_list).reshape((1, len(quality_list))), columns=columns)
    new_row['task'] = task
    new_row['algorithm'] = algorithm_name
    new_row['d'] = d
    new_row['r'] = r
    new_row['N'] = N

    i = list(new_row.columns)
    new_i = i[len(quality_list):] + i[0:len(quality_list)]
    new_row = new_row[new_i]

    df = pd.concat([df, new_row])
    df.to_csv(path, index=False)
    return df