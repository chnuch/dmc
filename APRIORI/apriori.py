#Apply a-priori algorithm to find frequently occurring items from given data and generate strong association rules using support and confidence thresholds.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Load the dataset (make sure the file path is correct)
basket = pd.read_csv("d:\\Downloads\\Groceries_dataset.csv\\Groceries_dataset.csv")
display(basket.head())

# Transform 'itemDescription' column into a list of items
basket['itemDescription'] = basket['itemDescription'].transform(lambda x: [x])

# Group the data by 'Member_number' and 'Date', and sum the 'itemDescription' for each group
basket = basket.groupby(['Member_number', 'Date'])['itemDescription'].sum().reset_index()
display(basket.head())

# Convert the basket into a list of transactions (each transaction is a list of items)
transactions = basket['itemDescription'].tolist()

# Apply the TransactionEncoder to the data and transform it into a one-hot encoded format
encoder = TransactionEncoder()
transactions = pd.DataFrame(encoder.fit(transactions).transform(transactions), columns=encoder.columns_)

# Run the Apriori algorithm to find frequent itemsets with a minimum support of 6/len(basket)
frequent_itemsets = apriori(transactions, min_support=6/len(basket), use_colnames=True)

# Display frequent itemsets
display(frequent_itemsets)

# Generate association rules with a minimum lift of 1.5
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.5)

# Display the first few rules
display(rules.head())
print("Rules identified:", len(rules))

# Visualize the association rules in a 3D scatter plot
sns.set(style="whitegrid")
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')

# Extract support, confidence, and lift values for the plot
x = rules['support']
y = rules['confidence']
z = rules['lift']

# Set axis labels and plot the 3D scatter plot
ax.set_xlabel("Support")
ax.set_ylabel("Confidence")
ax.set_zlabel("Lift")
ax.scatter(x, y, z)
ax.set_title("3D Distribution of Association Rules")

# Show the plot
plt.show()

# Filter the rules where 'whole milk' is in the consequents and sort by lift
milk_rules = rules[rules['consequents'].astype(str).str.contains('whole milk')]

# Sort the milk-related rules by lift in descending order
milk_rules = milk_rules.sort_values(by=['lift'], ascending=False).reset_index(drop=True)

# Display the top milk-related rules
display(milk_rules.head())
