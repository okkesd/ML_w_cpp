import sys
import pandas as pd
import matplotlib.pyplot as plt

# Get CSV filename
csv_file = sys.argv[1]

df = pd.read_csv(csv_file)
store_name = df["store"][0]
print(df.columns)
print(store_name)

plt.figure(figsize=(13,7))
plt.scatter(df['x'], df['y'], label="real data")
plt.plot(df['x'], df['y_pred'], label="prediction", color="red")

plt.title(f"Data for store {store_name}")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(0.1)
plt.legend()
plt.tight_layout()
plt.savefig("plot.png")
plt.show()  # Optional
