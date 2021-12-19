import seaborn as sns
import pandas as pd

def plot_average_over_runs(dataframe, task):
    df = dataframe[dataframe['task'] == task]
    palette = sns.color_palette("mako", 3)
    sns.lineplot(
        data=df, x="step", y="quality",
        hue="task"#, palette=palette
    )