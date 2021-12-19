import numpy as np
import pandas as pd
import os


def save_run_data(file_name, algorithm_name, quality_list, d, r, N, task):

    rootdir = os.path.dirname(__file__)
    path = rootdir + "/" + file_name + ".csv"
    new_run_id = 0
    try:
        df = pd.read_csv(path)
        print("File found, adding the lines to the existent dataset...")
        new_run_id = max(df['run_id']) + 1
    except:
        df = pd.DataFrame()
        print("File not found, creating a new dataset...")

    columns = ['step', 'quality']
    data = zip(range(len(quality_list)), quality_list)
    new_rows = pd.DataFrame(data, columns=columns)
    new_rows['run_id'] = new_run_id
    new_rows['task'] = task
    new_rows['algorithm'] = algorithm_name
    new_rows['d'] = d
    new_rows['r'] = r
    new_rows['N'] = N

    i = list(new_rows.columns)
    new_i = i[2:] + i[0:2]
    new_rows = new_rows[new_i]

    df = pd.concat([df, new_rows])
    df.to_csv(path, index=False)
    return df