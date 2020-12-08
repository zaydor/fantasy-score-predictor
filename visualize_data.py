import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

print('Starting...')

df = pd.read_csv('data/processed/training_data.csv')

df_all = df[['ra','sra','ry','tdr','trg','rec','recy','tdrec','fuml','snp','seas', \
    'height','weight','forty','bench','vertical','broad','shuttle','cone', \
    'ou','d_ypa','d_rtd', 'fp']]

df_perf = df[['ra','sra','ry','tdr','trg','rec','recy','tdrec','fuml','snp','seas','fp']]
df_pa = df[['height','weight','forty','bench','vertical','broad','shuttle','cone','fp']]
df_game = df[['ou','d_ypa','d_rtd','fp']]

df_all.hist(bins=10, grid=False)
plt.tight_layout()
plt.show()
plt.clf()

f, ax = plt.subplots()
corr = df_all.corr()
hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap='coolwarm', fmt='.2f', linewidths=.05)
plt.title('Fantasy Football Feature Heat Map')
plt.show()
plt.clf()

for name, data in [('Player Performance', df_perf), ('Player Attributes', df_pa), ('Defense', df_game)]:
    sns.pairplot(data)
#plt.suptitle('Pair-wise Scatter Plot of {} Features'.format(name)) # can't get this to look right
    plt.show()
    plt.clf()

