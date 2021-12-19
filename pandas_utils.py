import pandas as pd
import os


def save_run_data(file_name, algorithm_name, quality_list, d, r, N, task):

    rootdir = os.path.dirname(__file__)
    path = rootdir + "/" + file_name + ".csv"
    try:
        df = pd.read_csv(path)
        print("File found, adding the lines to the existent dataset...")
        print(df.head(1))
    except:
        df = pd.DataFrame()
        print("File not found, creating a new dataset...")

    row_dict = { 'step_' + str(step) : quality for step, quality in enumerate(quality_list) }
    row_dict['task'] = task
    row_dict['algorithm'] = algorithm_name
    row_dict['d'] = d
    row_dict['r'] = r
    row_dict['N'] = N

    new_row = pd.DataFrame(row_dict, index=[0])

    df = pd.concat([df, new_row])
    df = df.reset_index(drop=True)
    df.to_csv(path, index=False)
