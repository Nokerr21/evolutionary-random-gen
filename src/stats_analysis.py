import pandas as pd
import scipy.stats as stats
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load data
df = pd.read_csv("res_data/results.csv")

# Test functions (Cec2017)
functions = df['function'].unique()

for func in functions:
    print(f"\nFunkcja testowa: {func}")
    sub = df[df['function'] == func]

    # Prepare data for the Kruskal-Wallis test
    grouped = [group["score"].values for name, group in sub.groupby("generator")]

    # Kruskal-Wallis test
    stat, p = stats.kruskal(*grouped)
    print(f"Kruskal-Wallis H={stat:.4f}, p={p:.4g}")

    if p < 0.05:
        print("-> Istotne różnice, wykonuję test post hoc (Dunna)...")
        posthoc = sp.posthoc_dunn(sub, val_col='score', group_col='generator', p_adjust='bonferroni')
        print(posthoc)

        os.makedirs("res_data", exist_ok=True)
        posthoc.to_csv(f"res_data/posthoc_{func}.csv")

        # Heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(posthoc, annot=True, cmap="coolwarm", fmt=".3f")
        plt.title(f"Test post hoc Dunn – {func}")
        plt.tight_layout()
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/posthoc_{func}.png")
        plt.close()
    else:
        print("-> Brak istotnych różnic, nie wykonuję testu post hoc")


chosen_func = "f2"
subset = df[df['function'] == chosen_func]

sns.set(style="whitegrid")

plt.figure(figsize=(12, 6))
sns.boxplot(x="generator", y="score", data=subset, palette="deep", showfliers=True)
sns.stripplot(x="generator", y="score", data=subset, color="black", alpha=0.5, jitter=True)

plt.title(f"Porównanie wyników dla różnych RNG\n(Funkcja: {chosen_func})")
plt.xlabel("Generator")
plt.ylabel("Najlepszy wynik")
plt.yscale("log")
plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig(f"plots/boxplot_{chosen_func}.png")
plt.show()
