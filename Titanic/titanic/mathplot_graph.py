import sys  
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('tested.csv')
graph = df.groupby('Pclass')['Survived'].mean()
count = sns.countplot(data='graph' , x='Survival')
plt.show()

# counts = df["Survived"].value_counts().sort_index()
# ax = counts.plot(kind='bar', color=['#e74c3c', '#2ecc71'])  # red = died, green = survived
# ax.set_xticks([0,1])
# ax.set_xticklabels(["Did not survived ", "Survived"], rotation = 0  )

# plt.show()
