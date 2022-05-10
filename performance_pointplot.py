from cProfile import label
import enum
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

megatron_opt_df = pd.DataFrame(data=[
                               [0, 23.64, 1.888, 'Megatron-Opt'],
                               [0, 47.28, 1.914, 'Megatron-Opt'],
                               [0, 94.56, 1.933, 'Megatron-Opt'],
                               [0, 31.52, 2.049, 'Megatron-Opt'],
                               [0, 63.04, 2.066, 'Megatron-Opt'],
                               [0, 126.08, 2.090, 'Megatron-Opt'],
                               [0, 39.41, 2.087, 'Megatron-Opt'],
                               [0, 78.82, 2.138, 'Megatron-Opt'],
                               [0, 157.64, 2.161, 'Megatron-Opt'],
                               ],
                               columns=['case', 'pflop', 'pflops', 'system'])

case_0_df = pd.DataFrame(data=[[0, 23.64, 0.971, 'Megatron-Default'],
                               [0, 47.28, 0.932, 'Megatron-Default'],
                               [0, 94.56, 0.929, 'Megatron-Default'],
                               [0, 31.52, 1.176, 'Megatron-Default'],
                               [0, 63.04, 1.179, 'Megatron-Default'],
                               [0, 126.08, 1.180, 'Megatron-Default'],
                               [0, 39.41, 1.377, 'Megatron-Default'],
                               [0, 78.82, 1.388, 'Megatron-Default'],
                               [0, 157.64, 1.413, 'Megatron-Default'],

                               [0, 23.64, 2.331, 'Ours (w/ Scheduler)'],
                               [0, 47.28, 2.683, 'Ours (w/ Scheduler)'],
                               [0, 94.56, 2.866, 'Ours (w/ Scheduler)'],
                               [0, 31.52, 2.365, 'Ours (w/ Scheduler)'],
                               [0, 63.04, 2.715, 'Ours (w/ Scheduler)'],
                               [0, 126.08, 2.925, 'Ours (w/ Scheduler)'],
                               [0, 39.41, 2.360, 'Ours (w/ Scheduler)'],
                               [0, 78.82, 2.665, 'Ours (w/ Scheduler)'],
                               [0, 157.64, 2.930, 'Ours (w/ Scheduler)'],

                               [0, 23.64, 1.166, 'Ours (w/o Scheduler)'],
                               [0, 47.28, 1.422, 'Ours (w/o Scheduler)'],
                               [0, 94.56, 1.438, 'Ours (w/o Scheduler)'],
                               [0, 31.52, 1.365, 'Ours (w/o Scheduler)'],
                               [0, 63.04, 1.599, 'Ours (w/o Scheduler)'],
                               [0, 126.08, 1.607, 'Ours (w/o Scheduler)'],
                               [0, 39.41, 1.610, 'Ours (w/o Scheduler)'],
                               [0, 78.82, 1.763, 'Ours (w/o Scheduler)'],
                               [0, 157.64, 1.776, 'Ours (w/o Scheduler)'],
                               ],
                         columns=['case', 'pflop', 'pflops', 'system'])

case_1_df = pd.DataFrame(data=[[1, 23.64, 1.109, 'Megatron-Default'],
                               [1, 47.28, 1.116, 'Megatron-Default'],
                               [1, 94.56, 1.132, 'Megatron-Default'],
                               [1, 31.52, 1.360, 'Megatron-Default'],
                               [1, 63.04, 1.443, 'Megatron-Default'],
                               [1, 126.08, 1.474, 'Megatron-Default'],
                               [1, 39.41, 1.558, 'Megatron-Default'],
                               [1, 78.82, 1.595, 'Megatron-Default'],
                               [1, 157.64, 1.639, 'Megatron-Default'],

                               [1, 23.64, 1.917, 'Ours (w/ Scheduler)'],
                               [1, 47.28, 2.063, 'Ours (w/ Scheduler)'],
                               [1, 94.56, 2.090, 'Ours (w/ Scheduler)'],
                               [1, 31.52, 2.004, 'Ours (w/ Scheduler)'],
                               [1, 63.04, 2.176, 'Ours (w/ Scheduler)'],
                               [1, 126.08, 2.252, 'Ours (w/ Scheduler)'],
                               [1, 39.41, 2.138, 'Ours (w/ Scheduler)'],
                               [1, 78.82, 2.292, 'Ours (w/ Scheduler)'],
                               [1, 157.64, 2.345, 'Ours (w/ Scheduler)'],

                               [1, 23.64, 1.545, 'Ours (w/o Scheduler)'],
                               [1, 47.28, 1.681, 'Ours (w/o Scheduler)'],
                               [1, 94.56, 1.760, 'Ours (w/o Scheduler)'],
                               [1, 31.52, 1.683, 'Ours (w/o Scheduler)'],
                               [1, 63.04, 1.883, 'Ours (w/o Scheduler)'],
                               [1, 126.08, 1.957, 'Ours (w/o Scheduler)'],
                               [1, 39.41, 1.819, 'Ours (w/o Scheduler)'],
                               [1, 78.82, 1.969, 'Ours (w/o Scheduler)'],
                               [1, 157.64, 2.042, 'Ours (w/o Scheduler)'],
                               ],
                         columns=['case', 'pflop', 'pflops', 'system'])

case_2_df = pd.DataFrame(data=[[2, 23.64, 1.111, 'Megatron-Default'],
                               [2, 47.28, 1.176, 'Megatron-Default'],
                               [2, 94.56, 1.162, 'Megatron-Default'],
                               [2, 31.52, 1.274, 'Megatron-Default'],
                               [2, 63.04, 1.396, 'Megatron-Default'],
                               [2, 126.08, 1.394, 'Megatron-Default'],
                               [2, 39.41, 1.469, 'Megatron-Default'],
                               [2, 78.82, 1.576, 'Megatron-Default'],
                               [2, 157.64, 1.610, 'Megatron-Default'],

                               [2, 23.64, 1.240, 'Ours (w/ Scheduler)'],
                               [2, 47.28, 1.447, 'Ours (w/ Scheduler)'],
                               [2, 94.56, 1.735, 'Ours (w/ Scheduler)'],
                               [2, 31.52, 1.410, 'Ours (w/ Scheduler)'],
                               [2, 63.04, 1.686, 'Ours (w/ Scheduler)'],
                               [2, 126.08, 1.956, 'Ours (w/ Scheduler)'],
                               [2, 39.41, 1.459, 'Ours (w/ Scheduler)'],
                               [2, 78.82, 1.755, 'Ours (w/ Scheduler)'],
                               [2, 157.64, 2.015, 'Ours (w/ Scheduler)'],

                               [2, 23.64, 0.850, 'Ours (w/o Scheduler)'],
                               [2, 47.28, 0.945, 'Ours (w/o Scheduler)'],
                               [2, 94.56, 1.013, 'Ours (w/o Scheduler)'],
                               [2, 31.52, 1.064, 'Ours (w/o Scheduler)'],
                               [2, 63.04, 1.231, 'Ours (w/o Scheduler)'],
                               [2, 126.08, 1.328, 'Ours (w/o Scheduler)'],
                               [2, 39.41, 1.208, 'Ours (w/o Scheduler)'],
                               [2, 78.82, 1.409, 'Ours (w/o Scheduler)'],
                               [2, 157.64, 1.540, 'Ours (w/o Scheduler)'],
                               ],
                         columns=['case', 'pflop', 'pflops', 'system'])

case_3_df = pd.DataFrame(data=[[3, 23.64, 0.610, 'Megatron-Default'],
                               [3, 47.28, 0.648, 'Megatron-Default'],
                               [3, 94.56, 0.667, 'Megatron-Default'],
                               [3, 31.52, 0.740, 'Megatron-Default'],
                               [3, 63.04, 0.788, 'Megatron-Default'],
                               [3, 126.08, 0.832, 'Megatron-Default'],
                               [3, 39.41, 0.845, 'Megatron-Default'],
                               [3, 78.82, 0.920, 'Megatron-Default'],
                               [3, 157.64, 0.957, 'Megatron-Default'],

                               [3, 23.64, 1.063, 'Ours (w/ Scheduler)'],
                               [3, 47.28, 1.222, 'Ours (w/ Scheduler)'],
                               [3, 94.56, 1.313, 'Ours (w/ Scheduler)'],
                               [3, 31.52, 1.201, 'Ours (w/ Scheduler)'],
                               [3, 63.04, 1.381, 'Ours (w/ Scheduler)'],
                               [3, 126.08, 1.465, 'Ours (w/ Scheduler)'],
                               [3, 39.41, 1.244, 'Ours (w/ Scheduler)'],
                               [3, 78.82, 1.523, 'Ours (w/ Scheduler)'],
                               [3, 157.64, 1.651, 'Ours (w/ Scheduler)'],

                               [3, 23.64, 0.739, 'Ours (w/o Scheduler)'],
                               [3, 47.28, 0.828, 'Ours (w/o Scheduler)'],
                               [3, 94.56, 0.891, 'Ours (w/o Scheduler)'],
                               [3, 31.52, 0.933, 'Ours (w/o Scheduler)'],
                               [3, 63.04, 1.079, 'Ours (w/o Scheduler)'],
                               [3, 126.08, 1.158, 'Ours (w/o Scheduler)'],
                               [3, 39.41, 1.048, 'Ours (w/o Scheduler)'],
                               [3, 78.82, 1.240, 'Ours (w/o Scheduler)'],
                               [3, 157.64, 1.360, 'Ours (w/o Scheduler)'],
                               ],
                         columns=['case', 'pflop', 'pflops', 'system'])

case_4_df = pd.DataFrame(data=[[4, 23.64, 0.285, 'Megatron-Default'],
                               [4, 47.28, 0.301, 'Megatron-Default'],
                               [4, 94.56, 0.307, 'Megatron-Default'],
                               [4, 31.52, 0.367, 'Megatron-Default'],
                               [4, 63.04, 0.382, 'Megatron-Default'],
                               [4, 126.08, 0.385, 'Megatron-Default'],
                               [4, 39.41, 0.441, 'Megatron-Default'],
                               [4, 78.82, 0.458, 'Megatron-Default'],
                               [4, 157.64, 0.467, 'Megatron-Default'],

                               [4, 23.64, 0.540, 'Ours (w/ Scheduler)'],
                               [4, 47.28, 0.759, 'Ours (w/ Scheduler)'],
                               [4, 94.56, 0.909, 'Ours (w/ Scheduler)'],
                               [4, 31.52, 0.617, 'Ours (w/ Scheduler)'],
                               [4, 63.04, 0.876, 'Ours (w/ Scheduler)'],
                               [4, 126.08, 1.106, 'Ours (w/ Scheduler)'],
                               [4, 39.41, 0.677, 'Ours (w/ Scheduler)'],
                               [4, 78.82, 0.977, 'Ours (w/ Scheduler)'],
                               [4, 157.64, 1.271, 'Ours (w/ Scheduler)'],

                               [4, 23.64, 0.256, 'Ours (w/o Scheduler)'],
                               [4, 47.28, 0.285, 'Ours (w/o Scheduler)'],
                               [4, 94.56, 0.302, 'Ours (w/o Scheduler)'],
                               [4, 31.52, 0.318, 'Ours (w/o Scheduler)'],
                               [4, 63.04, 0.365, 'Ours (w/o Scheduler)'],
                               [4, 126.08, 0.393, 'Ours (w/o Scheduler)'],
                               [4, 39.41, 0.369, 'Ours (w/o Scheduler)'],
                               [4, 78.82, 0.435, 'Ours (w/o Scheduler)'],
                               [4, 157.64, 0.476, 'Ours (w/o Scheduler)'],
                               ],
                         columns=['case', 'pflop', 'pflops', 'system'])

cases_df = [case_0_df, case_1_df, case_2_df, case_3_df, case_4_df]
fig, axes = plt.subplots(nrows=5, sharex=True, figsize=(5, 15))
sns.set_theme()
for i, df in enumerate(cases_df):
    for j in df.index:
        df.loc[j, 'pflops'] = df.loc[j, 'pflops'] / \
            megatron_opt_df.loc[j % 9, 'pflops']

    ax = sns.pointplot(ax=axes[i], data=df,
                       x='pflop', y="pflops", hue="system")
    ax.set(ylim=(0, 1.6))
    ax.set_ylabel('Relative Performance')
    ax.axhline(y=1.0, linestyle="--", color='dimgray')
    ax.text(3.5, 1.1, 'Megatron in Data Center', color='dimgray', fontsize=9)

    if i == 4:
        ax.set(xticklabels=['L24\nB1k', 'L32\nB1k', 'L48\nB1k',
                            'L24\nB2k', 'L32\nB2k', 'L48\nB2k',
                            'L24\nB4k', 'L32\nB4k', 'L48\nB4k', ])
        ax.set_xlabel('Model Architecture')
        ax.get_legend().set_title(None)
        ax.legend(loc='lower center', bbox_to_anchor=(
            0.48, -0.5), ncol=3, prop={'size': 9}, facecolor='white')
    else:
        ax.set_xlabel(None)
        ax.get_legend().remove()
plt.savefig("performance_pointplot.eps", dpi=1000)
