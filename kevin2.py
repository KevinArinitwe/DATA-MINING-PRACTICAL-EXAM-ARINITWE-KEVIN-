print("PART A: DATA PREPARATION\n")

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


#  Creating transaction data


data = {
    'Transaction_ID': [1,2,3,4,5,6,7,8,9,10],
    'Items': [
        ['Bread','Milk','Eggs'],
        ['Bread','Butter'],
        ['Milk','Diapers','Beer'],
        ['Bread','Milk','Butter'],
        ['Milk','Diapers','Bread'],
        ['Beer','Diapers'],
        ['Bread','Milk','Eggs','Butter'],
        ['Eggs','Milk'],
        ['Bread','Diapers','Beer'],
        ['Milk','Butter']
    ]
}

# Load the data into a DataFrame
df = pd.DataFrame(data)
print("Original Transaction Data:")
print(df)  


# Converting transactions into one-hot format


te = TransactionEncoder()  
te_array = te.fit(df['Items']).transform(df['Items'])  
encoded_df = pd.DataFrame(te_array, columns=te.columns_)
print("\nOne-Hot Encoded Transaction Data:")
print(encoded_df)  

print("\nPART B: APRIORI ALGORITHM\n")
print("Minimum Support = 0.2")
print("Minimum Confidence = 0.5\n")


#  Finding frequent itemsets

frequent_itemsets = apriori(
    encoded_df,
    min_support=0.2,
    use_colnames=True  # Use item names instead of column indices
)
print("Frequent Itemsets:")
print(frequent_itemsets)  # Display itemsets with their support

#  Generating association rules

rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.5
)

# Showing only the key columns for interpretation
print("\nAssociation Rules (Support, Confidence, Lift):")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])