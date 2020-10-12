
#%%

import pandas as panda
from sklearn.tree import DecisionTreeClassifier

shrooms = panda.read_csv(".\mushrooms.csv", header=None, na_values='?') 

for i in range(len(shrooms)):
    if(panda.isna(shrooms.iloc[i, 11])):
        shrooms.iloc[i, 11] = 'b'

shrooms = shrooms.iloc[1:]

for col in range(23):
    print(col)
    value_set = list(set(shrooms.iloc[i, col] for i in range(len(shrooms))))

    for i in range(len(shrooms)):
        shrooms.iloc[i, col] = value_set.index(shrooms.iloc[i, col])
        
print(shrooms)
        
#%%

##shrooms = shrooms.sample(frac = 1) 

features = [shrooms.iloc[i, 1:] for i in range(len(shrooms))]
labels = [shrooms.iloc[i, 0] for i in range(len(shrooms))]

split = int(len(features) * .66)

train_features = features[:split]
train_labels = labels[:split]

test_features = features[split:]
test_labels = labels[split:]
    
DT = DecisionTreeClassifier()
DT = DT.fit(train_features, train_labels)


for i in range(len(test_features)):
    print(f'Actual label: {test_labels[i]}  '
          f'Predicted label: '
          f'{DT.predict([test_features[i]])[0]}')

accuracy = DT.score(test_features, test_labels)
print(f'Accuracy: {accuracy}')
        











