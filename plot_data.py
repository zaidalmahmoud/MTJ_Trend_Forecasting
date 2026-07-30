import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv("data/data.csv")

#REMOVING!!!!
# df= df.iloc[100:]
# df = df.drop(df.columns[[-1, -17]], axis=1)

# First column is Date, remaining columns are features
dates = df.iloc[:, 0]
features = df.iloc[:, 1:]

# Plot one feature at a time
for column in features.columns:
    plt.figure(figsize=(12, 5))
    plt.plot(features[column], linewidth=1.5)

    plt.title(column)
    plt.xlabel("Months")
    plt.ylabel("Trend")

    # Remove x-axis ticks and labels
    plt.xticks([])

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Wait until the window is closed before continuing
    plt.show()