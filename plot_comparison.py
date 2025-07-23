import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import numpy as np

def plot_algorithms_comparison(csv_files, output_pdf):
    plt.figure()
    all_means = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        mainname_cols = [col for col in df.columns if "__" not in col and col != "size"]
        if not mainname_cols:
            print(f"Warning: no mainname column found in {csv_file.name}")
            continue
        main_col = mainname_cols[0]
        mainname = main_col

        means = []
        sizes = sorted(df["size"].unique(), key=int)
        for size in sizes:
            values = df[df["size"] == size][main_col].dropna().astype(float)
            if len(values) > 1:
                values = values.drop(values.idxmax())
            values = values / 1000
            mean = values.mean() if not values.empty else float("nan")
            means.append(mean)
        all_means.extend(means)
        plt.plot(
            sizes,
            means,
            marker="o",
            label=mainname
        )

    plt.xlabel("n")
    plt.ylabel("s")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    plt.xscale("log", base=2)
    plt.xticks(sizes, [str(s) for s in sizes])
    plt.gca().set_xticks(sizes, minor=False)
    plt.gca().set_xticklabels([str(s) for s in sizes])

    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.3f}"))

    if all_means:
        min_y = min(all_means)
        max_y = max(all_means)
        ticks = np.linspace(min_y, max_y, num=15)
        ax.set_yticks(ticks)

    plt.savefig(output_pdf, format="pdf")
    plt.close()
    print(f"Saved plot: {output_pdf}")

def main():
    parser = argparse.ArgumentParser(description="Plot algorithm comparison (mean-excluding-max) from CSVs")
    parser.add_argument(
        "csv_files",
        nargs="+",
        help="List of CSV files to compare"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="comp.pdf",
        help="Output PDF file name (default: algorithm_comparison_mean_excl_max.pdf)"
    )
    args = parser.parse_args()

    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)
    output_pdf = output_dir / args.output

    csv_files = [Path(f) for f in args.csv_files]
    plot_algorithms_comparison(csv_files, output_pdf)

if __name__ == "__main__":
    main()
