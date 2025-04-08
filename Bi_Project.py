#Loading and Libraries importation
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as s







#Data Cleaning  - Cecilia













#Data Preparation for analysis/ Classification - Jane and Nicolas
#Jane

# Set analysis date as the last invoice date in the dataset plus one day
analysis_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
print("Analysis Date:", analysis_date)

# Aggregate data for each customer to create the RFM table
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (analysis_date - x.max()).days,  # Recency
    'InvoiceNo': 'nunique',                                  # Frequency
    'TotalPrice': 'sum'                                      # Monetary
}).reset_index()

# Rename columns for clarity
rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

# Define churn: customers with recency greater than 180 days (6 months)
rfm['Churn'] = np.where(rfm['Recency'] > 180, 1, 0)

print("\nRFM Table Sample:")
print(rfm.head())

# Visualize the distribution of RFM metrics
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
sns.histplot(rfm['Recency'], kde=True)
plt.title('Recency Distribution')

plt.subplot(1, 3, 2)
sns.histplot(rfm['Frequency'], kde=True)
plt.title('Frequency Distribution')

plt.subplot(1, 3, 3)
sns.histplot(rfm['Monetary'], kde=True)
plt.title('Monetary Distribution')

plt.tight_layout()
plt.show()


#Nicholas










#Clustering - Cess and Anne

















#Association Rule Mining - Evans and Sandra
#sandra

from mlxtend.frequent_patterns import apriori, association_rules

# For basket analysis, we pivot the data such that rows represent invoices and columns represent items.
# We will use the InvoiceNo and Description fields.

# Create a basket: one-hot encoding of items per invoice
basket = df.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().reset_index().fillna(0).set_index('InvoiceNo')

# Convert quantities to binary values (1 if item was bought, else 0)
basket = basket.applymap(lambda x: 1 if x > 0 else 0)

print("\nBasket Sample:")
print(basket.head())

# Generate frequent itemsets using the apriori algorithm
frequent_itemsets = apriori(basket, min_support=0.02, use_colnames=True)
print("\nFrequent Itemsets:")
print(frequent_itemsets.head())

#Evans
# Generate association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())

# Visualize top association rules by lift
top_rules = rules.sort_values('lift', ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x='lift', y='antecedents', data=top_rules)
plt.title("Top 10 Association Rules by Lift")
plt.xlabel("Lift")
plt.ylabel("Antecedents")
plt.show()
