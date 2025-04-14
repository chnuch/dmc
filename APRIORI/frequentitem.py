#Frequent Itemset Mining using A-Priori Apply the Apriori algorithm on a transactional dataset (e.g., online retail). Identify frequent itemsets using a chosen minimum support threshold.

import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder

data = pd.read_csv("d:\\Downloads\\Groceries_dataset.csv\\Groceries_dataset.csv")

data['itemDescription'] = data['itemDescription'].apply(lambda x: [x])

transactions = data.groupby(['Member_number', 'Date'])['itemDescription'].sum().reset_index()

te = TransactionEncoder()
te_data = te.fit(transactions['itemDescription']).transform(transactions['itemDescription'])
df = pd.DataFrame(te_data, columns=te.columns_)

frequent_itemsets = apriori(df, min_support=0.03, use_colnames=True)

print("Frequent Itemsets:")
print(frequent_itemsets.sort_values(by='support', ascending=False))
