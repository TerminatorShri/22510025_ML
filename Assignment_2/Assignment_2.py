import streamlit as st
import numpy as np
import scipy.stats as stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import quantile_transform

# Sidebar Inputs
st.sidebar.header("Variable Settings")
size = st.sidebar.number_input("Size of Data", value=10000)
mean_B = st.sidebar.number_input("Mean of B (Gaussian)", value=5.0)
sd_B = st.sidebar.number_input("Standard Deviation of B (Gaussian)", value=2.0)
a_I = st.sidebar.number_input("Shape Parameter of I (Power Law)", value=0.3)
p_H = st.sidebar.number_input("Probability of H (Geometric)", value=0.005)
rank_method = st.sidebar.selectbox(
    "Select Rank Method for Percentile Scaling",
    ["average", "min", "max", "dense", "ordinal"],
    index=0,
)
quantile_output = st.sidebar.radio(
    "Select Output Distribution for Quantile Transform", ["normal", "uniform"], index=0
)

# Generate Data
np.random.seed(42)
B = np.random.normal(mean_B, sd_B, size)
I = stats.powerlaw.rvs(a=a_I, size=size)
H = stats.geom.rvs(p=p_H, size=size)

data = pd.DataFrame({"B": B, "I": I, "H": H})


# Normalization Functions
def normalize(data, method):
    df = data.copy()
    if method == "Divide by Max":
        df = df / df.max()
    elif method == "Divide by Sum":
        df = df / df.sum()
    elif method == "Z-score":
        df = df.apply(stats.zscore)
    elif method == "Percentile Scaling":
        df = df.apply(lambda x: stats.rankdata(x, method=rank_method) / len(x))
    elif method == "Median Normalization":
        medians = df.median()
        target_median = medians.mean()
        multipliers = target_median / medians
        df *= multipliers
    elif method == "Quantile Normalization":
        df = pd.DataFrame(
            quantile_transform(df, output_distribution=quantile_output, copy=True),
            columns=df.columns,
        )
    return df


methods = [
    "Divide by Max",
    "Divide by Sum",
    "Z-score",
    "Percentile Scaling",
    "Median Normalization",
    "Quantile Normalization",
]

for method in methods:
    normalized_data = normalize(data, method)

    st.subheader(f"Normalization Method: {method}")

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"hspace": 0.3})
    axes[0].boxplot(
        [B, I, H],
        labels=["B (Gaussian)", "I (Power Law)", "H (Geometric)"],
        vert=False,
    )
    axes[0].set_title("Before Normalization")
    axes[0].set_xlabel("Values")
    axes[0].grid(axis="x")

    axes[1].boxplot(
        [normalized_data["B"], normalized_data["I"], normalized_data["H"]],
        labels=["B", "I", "H"],
        vert=False,
    )
    axes[1].set_title("After Normalization")
    axes[1].set_xlabel("Values")
    axes[1].grid(axis="x")

    st.pyplot(fig)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={"hspace": 0.4})
    for idx, (name, original, normalized, xlabel, ylabel) in enumerate(
        zip(
            data.columns,
            [B, I, H],
            [normalized_data["B"], normalized_data["I"], normalized_data["H"]],
            [
                "Value of B (Continuous)",
                "Value of I (Continuous)",
                "Number of Trials Until First Success",
            ],
            ["Frequency", "Frequency", "Count of Observations"],
        )
    ):
        axes[idx].hist(
            original, bins="auto", alpha=0.5, label=f"{name}_original", color="blue"
        )
        axes[idx].hist(
            normalized,
            bins="auto",
            alpha=0.5,
            label=f"{name}_normalized",
            color="orange",
        )
        axes[idx].set_title(
            f"Comparison of Original and Normalized Versions for {name}"
        )
        axes[idx].set_xlabel(xlabel)
        axes[idx].set_ylabel(ylabel)
        axes[idx].legend()

    # Side-by-Side Histograms
    # fig, axes = plt.subplots(3, 2, figsize=(14, 12), gridspec_kw={"hspace": 0.5, "wspace": 0.3})
    #
    # for idx, (name, original, normalized) in enumerate(
    #         zip(
    #             data.columns,
    #             [B, I, H],
    #             [normalized_data["B"], normalized_data["I"], normalized_data["H"]],
    #         )
    # ):
    #     # Original Data Histogram
    #     axes[idx, 0].hist(original, bins=50, alpha=0.5, label=f"{name} (Original)", color="blue")
    #     axes[idx, 0].set_title(f"Original {name}")
    #     axes[idx, 0].set_xlabel("Values")
    #     axes[idx, 0].set_ylabel("Frequency")
    #     axes[idx, 0].legend()
    #
    #     # Normalized Data Histogram
    #     axes[idx, 1].hist(normalized, bins=50, alpha=0.5, label=f"{name} (Normalized)", color="orange")
    #     axes[idx, 1].set_title(f"Normalized {name}")
    #     axes[idx, 1].set_xlabel("Values")
    #     axes[idx, 1].set_ylabel("Frequency")
    #     axes[idx, 1].legend()

    st.pyplot(fig)
