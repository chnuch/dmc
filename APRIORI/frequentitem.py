import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder

# Load and group data into transactions
data = pd.read_csv("d:\\Downloads\\Groceries_dataset.csv\\Groceries_dataset.csv")
data['itemDescription'] = data['itemDescription'].apply(lambda x: [x])
transactions = data.groupby(['Member_number', 'Date'])['itemDescription'].sum().tolist()

# One-hot encode
te = TransactionEncoder()
df = pd.DataFrame(te.fit(transactions).transform(transactions), columns=te.columns_)

# Find frequent itemsets
frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)

# Filter for itemsets with 2 or more items
top = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) > 1)]
top = top.sort_values(by='support', ascending=False).head(10)

# Print frequent itemsets
print("Top Frequent Itemsets:\n")
for _, row in top.iterrows():
    print(f"{set(row['itemsets'])} -> Support: {row['support']:.2f}")

# Plot
top['itemsets'] = top['itemsets'].apply(lambda x: ', '.join(list(x)))
plt.figure(figsize=(8, 5))
plt.barh(top['itemsets'], top['support'], color='skyblue')
plt.xlabel('Support')
plt.title('Top 10 Frequent Itemsets (≥2 items)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
