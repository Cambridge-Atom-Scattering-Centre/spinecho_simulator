from __future__ import annotations

import csv

import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Plotting
    input_csv1 = (
        "/workspaces/spinecho_sim/data/classical_solenoid_field-vs-intensity_data1.csv"
    )
    input_csv2 = (
        "/workspaces/spinecho_sim/data/classical_solenoid_field-vs-intensity_data2.csv"
    )

    B_fields_1 = []
    final_intensities_1 = []
    with open(input_csv1, encoding="utf-8", newline="") as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        for row in reader:
            B_fields_1.append(float(row[0]))
            final_intensities_1.append(float(row[1]))
    print(f"Data loaded from: {input_csv1}")

    B_fields_2 = []
    final_intensities_2 = []
    with open(input_csv2, encoding="utf-8", newline="") as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        for row in reader:
            B_fields_2.append(float(row[0]))
            final_intensities_2.append(float(row[1]))
    print(f"Data loaded from: {input_csv2}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(B_fields_1, final_intensities_1, marker="o", label="Dataset 1")
    ax.scatter(B_fields_2, final_intensities_2, marker="s", label="Dataset 2")
    print("Plotting results...")
    ax.set_xlabel(r"Magnetic Field Strength (arbitrary units)")
    ax.set_ylabel("Final Intensity (arb. units)")
    ax.set_title("Field Dependent Depolarisation, 50 spins")
    ax.grid(visible=True)
    ax.set_ylim(bottom=0)
    ax.legend()

    # Save the plot instead of showing it
    print("Saving plot...")
    output_path = "/workspaces/spinecho_sim/figures/classical_solenoid_field-vs-intensity_plot.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {output_path}")
