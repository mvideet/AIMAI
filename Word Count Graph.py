import pandas as pd
df = pd.read_excel("full_data.xlsx")
import numpy as np
import matplotlib.pyplot as plt


word_counts = []
df["Clinical signs/ symptoms"] = df["Clinical signs/ symptoms"].astype(str)
# Iterate over the "PROC_NAME" column
for proc_name in df["Clinical signs/ symptoms"]:
    # Split the string into a list of words
    words = proc_name.split()
    # Calculate the number of words
    word_count = len(words)
    # Append the word count to the list
    word_counts.append(word_count)

    # Calculate the average word count

average_word_count = sum(word_counts) / len(word_counts)
# Plot the histogram
plt.hist(word_counts, bins=20, edgecolor="black")

# Add labels and title to the plot
plt.xlabel("Word Count")
plt.ylabel("Frequency")
plt.title("Word Count Distribution")

# Show the plot19
#plt.show()

bins = np.arange(0, max(word_counts) + 11, 2   )

# Use the numpy histogram function to count the occurrences in each bin
hist, _ = np.histogram(word_counts, bins=bins)

# Create a table to store the results
table = pd.DataFrame({"Word Count Range": bins[:-1], "Frequency": hist})
print(table)