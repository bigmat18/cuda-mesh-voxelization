import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import numpy as np

def plot_algorithms_comparison(csv_files, output_pdf, subtract_labels=None):
    if subtract_labels is None:
        subtract_labels = []

    plt.figure()
    all_means = []
    all_sizes = set()

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        mainname_cols = [col for col in df.columns if "__" not in col and col != "size"]
        if not mainname_cols:
            print(f"Warning: no mainname column found in {csv_file.name}")
            continue
        main_col = mainname_cols[0]
        mainname = main_col

        col_to_process = main_col
        if subtract_labels:
            adjusted_values = df[main_col].copy()
            for label in subtract_labels:
                subtract_col_name = f"{main_col}__{label}"
                if subtract_col_name in df.columns:
                    adjusted_values -= df[subtract_col_name].fillna(0)
                else:
                    print(f"Warning: Column '{subtract_col_name}' not found in {csv_file.name}, cannot subtract.")
            
            df["adjusted_main_col"] = adjusted_values
            col_to_process = "adjusted_main_col"

        means = []
        sizes = sorted(df["size"].unique(), key=int)
        all_sizes.update(sizes)

        for size in sizes:
            values = df[df["size"] == size][col_to_process].dropna().astype(float)
            if len(values) > 1:
                values = values.drop(values.idxmax())
            mean = values.mean() if not values.empty else float("nan")
            means.append(mean)
        
        valid_means = [m for m in means if pd.notna(m)]
        if valid_means:
            all_means.extend(valid_means)

        plt.plot(
            sizes,
            means,
            marker="o",
            label=mainname
        )

    sorted_sizes = sorted(list(all_sizes))

    plt.xlabel("voxel grid size")
    plt.ylabel("ms")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    
    plt.xscale("log", base=2)
    plt.xticks(sorted_sizes, [str(s) for s in sorted_sizes])
    plt.gca().set_xticks(sorted_sizes, minor=False)
    plt.gca().set_xticklabels([str(s) for s in sorted_sizes])

    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.3f}".replace(".", ",")))

    if all_means:
        min_y = min(all_means)
        max_y = max(all_means)
        if min_y < max_y:
            ticks = np.linspace(min_y, max_y, num=15)
            ax.set_yticks(ticks)

    plt.tight_layout()
    plt.savefig(output_pdf, format="jpg")
    plt.close()
    print(f"Saved plot: {output_pdf}")

def main():
    parser = argparse.ArgumentParser(description="Plot algorithm comparison (mean-excluding-max) from CSVs")
    parser.add_argument("csv_files", nargs="+", help="List of CSV files to compare")
    parser.add_argument("--output", type=str, default="comp.pdf", help="Output PDF file name")
    parser.add_argument(
        "--exclude-labels",
        nargs="+",
        default=[],
        help="List of part labels to subtract from the main value."
    )
    args = parser.parse_args()

    output_dir = Path("images")
    output_dir.mkdir(exist_ok=True)
    output_pdf = output_dir / f"{args.output}.jpg"

    csv_files = [Path(f) for f in args.csv_files]
    plot_algorithms_comparison(csv_files, output_pdf, args.exclude_labels)

if __name__ == "__main__":
    main()
