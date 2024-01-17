import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./datasets/cleaned.csv")

category_counts = df['category'][80000:95000].value_counts()

category_counts.plot(kind='bar')

plt.xlabel('Category')
plt.ylabel('Stevilo')
plt.title('Novice po kategorijah')
plt.show()