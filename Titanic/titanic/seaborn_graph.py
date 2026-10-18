import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load your local file using pandas
graph = pd.read_csv('tested.csv')


graph['Age']= graph['Age'].fillna(graph['Age'].median()) 
graph['Canin']= graph['Cabin'].notnull().astype(int)
graph= graph.drop(columns=['Cabin'])


fig , axes = plt.subplots(1,2 , figsize = (12,5))

sns.set_style('white')
sns.set_theme()
sns.barplot(data=graph, x='Pclass', y='Survived', hue='Sex', ax=axes[0],
            palette=['#e74c3c', '#3498db'])
axes[0].set_title('Survival Rate by Passenger Class')
axes[0].set_ylabel('Survival Rate')

sns.histplot(data=graph, x='Age', kde=True, hue='Sex', ax=axes[1],
             palette=['#e74c3c', '#3498db'], alpha=0.5)
axes[1].set_title('Age Distribution')
axes[1].set_ylabel('Number of Passengers')
plt.tight_layout()
plt.show()