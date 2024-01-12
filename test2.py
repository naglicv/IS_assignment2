import pandas as pd
from keras.preprocessing.text import one_hot


df = pd.read_csv("./datasets/cleaned.csv")
df = df[:15000].copy()
text = df['clean_text'].str.cat(sep=', ')

vocab_size = 336
result = one_hot(text, round(vocab_size*1.3))

count = 0

l1 = []
 
# traversing the array
for item in result:
    if item not in l1:
        count += 1
        l1.append(item)

print(count)