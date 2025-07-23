import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import numpy as np

def plot_grouped_stacked_bars(csv_files, output_pdf):
    all_data = {}
    all_part_names = set()
    all_sizes = set()

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        mainname_col = [c for c in df.columns if "__" not in c and c != "size"][0]
        part_cols = [c for c in df.columns if "__" in c]
        
        if not part_cols:
            continue

        def mean_without_max(series):
            if len(series) > 1:
                series = series.drop(series.idxmax())
            return series.mean()

        grouped_means = df.groupby('size')[part_cols].agg(mean_without_max)
        grouped_means = (grouped_means / 1000).fillna(0)
        
        derived_label = csv_file.stem.split('_', 1)[-1]
        all_data[mainname_col] = (grouped_means, derived_label)
        all_part_names.update(part_cols)
        all_sizes.update(grouped_means.index)

    if not all_data:
        print("No data to plot.")
        return

    sorted_sizes = sorted(list(all_sizes))
    
    unique_short_parts = sorted(list({p.split('__')[-1] for p in all_part_names}))
    colors = plt.cm.get_cmap('tab20').colors
    color_map = {part: colors[i % len(colors)] for i, part in enumerate(unique_short_parts)}

    plt.figure(figsize=(12, 10))
    ax = plt.gca()

    num_algorithms = len(all_data)
    
    # --- SOLUZIONE SPESSORE FISSO ---
    # 1. Tratta le size come categorie posizionate a 0, 1, 2...
    y_indices = np.arange(len(sorted_sizes))
    # 2. Definisci un'altezza fissa per le barre
    total_group_height = 0.8  # Spazio totale per un gruppo di barre (lascia 0.2 di spazio tra i gruppi)
    single_bar_height = total_group_height / num_algorithms

    for algo_idx, (mainname, (data, derived_label)) in enumerate(all_data.items()):
        # 3. Calcola l'offset basato sull'altezza fissa
        vertical_shift = (algo_idx - (num_algorithms - 1) / 2.0) * single_bar_height
        y_positions = y_indices + vertical_shift
        
        left_offset = pd.Series(0.0, index=data.index).reindex(sorted_sizes, fill_value=0)
        
        for part_col in sorted(list(data.columns)):
            values = data[part_col].reindex(sorted_sizes, fill_value=0)
            short_label = part_col.split('__')[-1]
            
            ax.barh(
                y_positions,
                values,
                height=single_bar_height, # Altezza fissa e costante
                left=left_offset,
                label=short_label,
                color=color_map[short_label],
                align='center'
            )
            left_offset += values

        for i in range(len(sorted_sizes)):
            y_pos = y_positions[i]
            total_length = left_offset.iloc[i]
            ax.text(
                total_length * 1.01,
                y_pos,
                f" {derived_label}",
                verticalalignment='center',
                fontsize=9,
                color='black'
            )

    plt.xlabel("voxel grid size")
    plt.ylabel("seconds")
    ax.set_title("Algorithm Parts Comparison")
    ax.grid(axis='x', ls='--', alpha=0.5)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(), 
        by_label.keys(),
        title="Parts",
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=3
    )

    # 4. Imposta le etichette dell'asse y per simulare la scala logaritmica
    ax.set_yticks(y_indices)
    ax.set_yticklabels([str(s) for s in sorted_sizes])

    max_x_value = 0
    if all_data:
        all_sums = []
        for _, (df, _) in all_data.items():
            all_sums.append(df.sum(axis=1))
        if all_sums:
            max_x_value = pd.concat(all_sums).max()

    if max_x_value > 0:
        ax.set_xlim(0, max_x_value * 1.25)
        x_ticks = np.linspace(0, max_x_value, num=10)
        ax.set_xticks(x_ticks)

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.3f}".replace('.', ',')))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    
    plt.savefig(output_pdf, format="pdf")
    plt.close()
    print(f"Saved plot: {output_pdf}")

def main():
    parser = argparse.ArgumentParser(description="Plot grouped stacked horizontal bar charts from CSVs.")
    parser.add_argument(
        "csv_files",
        nargs="+",
        help="List of CSV files to plot."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="bar.pdf",
        help="Output PDF file name."
    )
    args = parser.parse_args()

    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)
    output_pdf = output_dir / args.output

    plot_grouped_stacked_bars([Path(f) for f in args.csv_files], output_pdf)

if __name__ == "__main__":
    main()
