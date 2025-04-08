#Loading and Libraries importation
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as s







#Data Cleaning  - Cecilia













#Data Preparation for analysis/ Classification - Jane and Nicolas














#Clustering - Cess and Anne

















#Association Rule Mining - Evans and Sandra
#sandra



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
