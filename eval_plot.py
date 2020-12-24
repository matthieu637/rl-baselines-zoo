import optuna
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

loaded_study = optuna.load_study(study_name="study", storage="sqlite:///" + sys.argv[1])

#fig = optuna.visualization.plot_parallel_coordinate(loaded_study)
#print(dir(fig))
#fig.show()

df = loaded_study.trials_dataframe()
df.dropna(inplace=True)
df.reset_index(inplace=True)

#df['time'] = df.datetime_complete - df.datetime_start
#df['time'] = df.time.astype('int') / (1_000_000_000)
#df = df[df.time>0]

print('best val:', - round(df.value.min(),4))
a = sns.lineplot(x=df.index, y=-df.value.cummin())
a.set_xlabel('trial number')
sns.scatterplot(x=df.index, y=-df.value, color='red')
a.set_ylabel('f1 score')
a.legend(['best value', "trial's value"])

print()
for i in range(5):
    print(i, np.array(df.sort_values('value'))[i,:])

print()
print()

print(df)
plt.show()

