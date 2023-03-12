import pandas as pd
from collections import Counter
df = pd.read_excel("ful_data.xlsx")
print(df.columns)
import pandas as pd
from collections import Counter
import string

# Load the data into a pandas dataframe


# Count the frequency of each imaging order in the dataframe
order_counts = Counter(df['PROC_NAME'].tolist())

# Get the 8 most common imaging orders
most_common_orders = order_counts.most_common(8)

# Print the 8 most common imaging orders
print("The 8 most common imaging orders are:")
for order, count in most_common_orders:
    print(f"- {order}: {count}")

# Preprocess the strings in the column
df["Clinical signs/ symptoms"] = df["Clinical signs/ symptoms"].astype(str)
df["Clinical signs/ symptoms"] = df["Clinical signs/ symptoms"].str.lower().str.translate(str.maketrans('', '', string.punctuation))

# Concatenate all the strings in the column into one large string
all_strings = ' '.join(df["Clinical signs/ symptoms"].tolist())

# Tokenize the string into words
words = all_strings.split()

# Count the frequency of each word
word_counts = Counter(words)

# Get the top 10 words by frequency
top_10_words = word_counts.most_common(10)

df2 = df.head(7)
df2.to_csv("short.csv")
print(word_counts)

import pandas as pd
from collections import Counter

# Load the data into a pandas dataframe


# Function to get the most common imaging order for a given symptom
def get_most_common_imaging_order(symptom):
    # Filter the dataframe to only include rows where the symptom column contains the given symptom
    filtered_df = df[df['Clinical signs/ symptoms'].str.contains(symptom, na=False)]

    # Count the frequency of each imaging order in the filtered dataframe
    order_counts = Counter(filtered_df['PROC_NAME'].tolist())

    # Return the most common imaging order
    return order_counts.most_common(1)[0][0]


# Example usage
symptom = "facial"
most_common_imaging_order = get_most_common_imaging_order(symptom)
print(f"The most common imaging order for symptom '{symptom}' is '{most_common_imaging_order}'.")


#printing the 8 most common imaging orders with its freqench
import pandas as pd

