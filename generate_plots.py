import subprocess
from pathlib import Path

TESTS_ASSETS_FOLDER = Path("tests")
PLOTS_FOLDER = Path("plots")
PLOTS_FOLDER.mkdir(exist_ok=True)

for filename in TESTS_ASSETS_FOLDER.iterdir():

    # ==================== PLOT COMPARISON MEMORY =======================
    plot_folder = PLOTS_FOLDER / filename.stem
    plot_folder.mkdir(exist_ok=True)
    
    command = [
        "python",
        "plot_comparison.py",
        f"benchmarks/{filename.stem}/{filename.stem}_sequential_vox.csv",  
        f"benchmarks/{filename.stem}/{filename.stem}_naive_vox.csv",  
        f"benchmarks/{filename.stem}/{filename.stem}_tiled_vox.csv",  
        "--output", 
        f"{filename.stem}/{filename.stem}_vox_comparison_memory_012.pdf"
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )

    command = [
        "python",
        "plot_comparison.py",
        f"benchmarks/{filename.stem}/{filename.stem}_naive_vox.csv",  
        f"benchmarks/{filename.stem}/{filename.stem}_tiled_vox.csv",  
        "--output", 
        f"{filename.stem}/{filename.stem}_vox_comparison_memory_12.pdf"
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )

    command = [
        "python",
        "plot_comparison.py",
        f"benchmarks/{filename.stem}/{filename.stem}_sequential_vox.csv",  
        f"benchmarks/{filename.stem}/{filename.stem}_naive_vox.csv",  
        "--output", 
        f"{filename.stem}/{filename.stem}_vox_comparison_memory_01.pdf"
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )


    command = [
        "python",
        "plot_comparison.py",
        f"benchmarks/{filename.stem}/{filename.stem}_openmp_csg.csv",  
        f"benchmarks/{filename.stem}/{filename.stem}_naive_csg.csv",  
        "--output", 
        f"{filename.stem}/{filename.stem}_csg_comparison_memory_12.pdf"
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )


    command = [
        "python",
        "plot_comparison.py",
        f"benchmarks/{filename.stem}/{filename.stem}_openmp_jfa.csv",  
        f"benchmarks/{filename.stem}/{filename.stem}_naive_jfa.csv",  
        f"benchmarks/{filename.stem}/{filename.stem}_tiled_jfa.csv",  
        "--output", 
        f"{filename.stem}/{filename.stem}_jfa_comparison_memory_012.pdf"
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )


    command = [
        "python",
        "plot_comparison.py",
        f"benchmarks/{filename.stem}/{filename.stem}_naive_jfa.csv",  
        f"benchmarks/{filename.stem}/{filename.stem}_tiled_jfa.csv",  
        "--output", 
        f"{filename.stem}/{filename.stem}_jfa_comparison_memory_12.pdf"
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )

    command = [
        "python",
        "plot_comparison.py",
        f"benchmarks/{filename.stem}/{filename.stem}_openmp_jfa.csv",  
        f"benchmarks/{filename.stem}/{filename.stem}_naive_jfa.csv",  
        "--output", 
        f"{filename.stem}/{filename.stem}_jfa_comparison_memory_01.pdf"
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )
    
    # ==================== PLOT BAR MEMORY ===========================

    command = [
        "python",
        "plot_bar_diagram.py",
        f"benchmarks/{filename.stem}/{filename.stem}_sequential_vox.csv",  
        f"benchmarks/{filename.stem}/{filename.stem}_naive_vox.csv",  
        f"benchmarks/{filename.stem}/{filename.stem}_tiled_vox.csv",  
        "--output", 
        f"{filename.stem}/{filename.stem}_vox_bar_diagram_memory_012.pdf"
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )

    command = [
        "python",
        "plot_bar_diagram.py",
        f"benchmarks/{filename.stem}/{filename.stem}_naive_vox.csv",  
        f"benchmarks/{filename.stem}/{filename.stem}_tiled_vox.csv",  
        "--output", 
        f"{filename.stem}/{filename.stem}_vox_bar_diagram_memory_12.pdf"
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )

    command = [
        "python",
        "plot_bar_diagram.py",
        f"benchmarks/{filename.stem}/{filename.stem}_sequential_vox.csv",  
        f"benchmarks/{filename.stem}/{filename.stem}_naive_vox.csv",  
        "--output", 
        f"{filename.stem}/{filename.stem}_vox_bar_diagram_memory_01.pdf"
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )


    command = [
        "python",
        "plot_bar_diagram.py",
        f"benchmarks/{filename.stem}/{filename.stem}_openmp_csg.csv",  
        f"benchmarks/{filename.stem}/{filename.stem}_naive_csg.csv",  
        "--output", 
        f"{filename.stem}/{filename.stem}_csg_bar_diagram_memory_12.pdf"
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )


    command = [
        "python",
        "plot_bar_diagram.py",
        f"benchmarks/{filename.stem}/{filename.stem}_openmp_jfa.csv",  
        f"benchmarks/{filename.stem}/{filename.stem}_naive_jfa.csv",  
        f"benchmarks/{filename.stem}/{filename.stem}_tiled_jfa.csv",  
        "--output", 
        f"{filename.stem}/{filename.stem}_jfa_bar_diagram_memory_012.pdf"
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )


    command = [
        "python",
        "plot_bar_diagram.py",
        f"benchmarks/{filename.stem}/{filename.stem}_naive_jfa.csv",  
        f"benchmarks/{filename.stem}/{filename.stem}_tiled_jfa.csv",  
        "--output", 
        f"{filename.stem}/{filename.stem}_jfa_bar_diagram_memory_12.pdf"
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )

    command = [
        "python",
        "plot_bar_diagram.py",
        f"benchmarks/{filename.stem}/{filename.stem}_openmp_jfa.csv",  
        f"benchmarks/{filename.stem}/{filename.stem}_naive_jfa.csv",  
        "--output", 
        f"{filename.stem}/{filename.stem}_jfa_bar_diagram_memory_01.pdf"
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )


    # ==================== PLOT COMPARISON NO MEMORY =======================
    plot_folder = PLOTS_FOLDER / filename.stem
    plot_folder.mkdir(exist_ok=True)
    
    command = [
        "python",
        "plot_comparison.py",
        f"benchmarks/{filename.stem}/{filename.stem}_sequential_vox.csv",  
        f"benchmarks/{filename.stem}/{filename.stem}_naive_vox.csv",  
        f"benchmarks/{filename.stem}/{filename.stem}_tiled_vox.csv",  
        "--output", 
        f"{filename.stem}/{filename.stem}_vox_comparison_no_memory_012.pdf",
        "--exclude-labels",
        "memory",
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )

    command = [
        "python",
        "plot_comparison.py",
        f"benchmarks/{filename.stem}/{filename.stem}_naive_vox.csv",  
        f"benchmarks/{filename.stem}/{filename.stem}_tiled_vox.csv",  
        "--output", 
        f"{filename.stem}/{filename.stem}_vox_comparison_no_memory_12.pdf",
        "--exclude-labels",
        "memory",
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )

    command = [
        "python",
        "plot_comparison.py",
        f"benchmarks/{filename.stem}/{filename.stem}_sequential_vox.csv",  
        f"benchmarks/{filename.stem}/{filename.stem}_naive_vox.csv",  
        "--output", 
        f"{filename.stem}/{filename.stem}_vox_comparison_no_memory_01.pdf",
        "--exclude-labels",
        "memory",
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )


    command = [
        "python",
        "plot_comparison.py",
        f"benchmarks/{filename.stem}/{filename.stem}_openmp_csg.csv",  
        f"benchmarks/{filename.stem}/{filename.stem}_naive_csg.csv",  
        "--output", 
        f"{filename.stem}/{filename.stem}_csg_comparison_no_memory_12.pdf",
        "--exclude-labels",
        "memory",
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )


    command = [
        "python",
        "plot_comparison.py",
        f"benchmarks/{filename.stem}/{filename.stem}_openmp_jfa.csv",  
        f"benchmarks/{filename.stem}/{filename.stem}_naive_jfa.csv",  
        f"benchmarks/{filename.stem}/{filename.stem}_tiled_jfa.csv",  
        "--output", 
        f"{filename.stem}/{filename.stem}_jfa_comparison_no_memory_012.pdf",
        "--exclude-labels",
        "memory",
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )


    command = [
        "python",
        "plot_comparison.py",
        f"benchmarks/{filename.stem}/{filename.stem}_naive_jfa.csv",  
        f"benchmarks/{filename.stem}/{filename.stem}_tiled_jfa.csv",  
        "--output", 
        f"{filename.stem}/{filename.stem}_jfa_comparison_no_memory_12.pdf",
        "--exclude-labels",
        "memory",
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )

    command = [
        "python",
        "plot_comparison.py",
        f"benchmarks/{filename.stem}/{filename.stem}_openmp_jfa.csv",  
        f"benchmarks/{filename.stem}/{filename.stem}_naive_jfa.csv",  
        "--output", 
        f"{filename.stem}/{filename.stem}_jfa_comparison_no_memory_01.pdf",    
        "--exclude-labels",
        "memory",
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )


    # ==================== PLOT BAR NO MEMORY ===========================

    command = [
        "python",
        "plot_bar_diagram.py",
        f"benchmarks/{filename.stem}/{filename.stem}_sequential_vox.csv",  
        f"benchmarks/{filename.stem}/{filename.stem}_naive_vox.csv",  
        f"benchmarks/{filename.stem}/{filename.stem}_tiled_vox.csv",  
        "--output", 
        f"{filename.stem}/{filename.stem}_vox_bar_diagram_no_memory_012.pdf",
        "--exclude-labels",
        "memory",
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )

    command = [
        "python",
        "plot_bar_diagram.py",
        f"benchmarks/{filename.stem}/{filename.stem}_naive_vox.csv",  
        f"benchmarks/{filename.stem}/{filename.stem}_tiled_vox.csv",  
        "--output", 
        f"{filename.stem}/{filename.stem}_vox_bar_diagram_no_memory_12.pdf",
        "--exclude-labels",
        "memory",
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )

    command = [
        "python",
        "plot_bar_diagram.py",
        f"benchmarks/{filename.stem}/{filename.stem}_sequential_vox.csv",  
        f"benchmarks/{filename.stem}/{filename.stem}_naive_vox.csv",  
        "--output", 
        f"{filename.stem}/{filename.stem}_vox_bar_diagram_no_memory_01.pdf",
        "--exclude-labels",
        "memory",
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )


    command = [
        "python",
        "plot_bar_diagram.py",
        f"benchmarks/{filename.stem}/{filename.stem}_openmp_csg.csv",  
        f"benchmarks/{filename.stem}/{filename.stem}_naive_csg.csv",  
        "--output", 
        f"{filename.stem}/{filename.stem}_csg_bar_diagram_no_memory_12.pdf",
        "--exclude-labels",
        "memory",
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )


    command = [
        "python",
        "plot_bar_diagram.py",
        f"benchmarks/{filename.stem}/{filename.stem}_openmp_jfa.csv",  
        f"benchmarks/{filename.stem}/{filename.stem}_naive_jfa.csv",  
        f"benchmarks/{filename.stem}/{filename.stem}_tiled_jfa.csv",  
        "--output", 
        f"{filename.stem}/{filename.stem}_jfa_bar_diagram_no_memory_012.pdf",
        "--exclude-labels",
        "memory",
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )


    command = [
        "python",
        "plot_bar_diagram.py",
        f"benchmarks/{filename.stem}/{filename.stem}_naive_jfa.csv",  
        f"benchmarks/{filename.stem}/{filename.stem}_tiled_jfa.csv",  
        "--output", 
        f"{filename.stem}/{filename.stem}_jfa_bar_diagram_no_memory_12.pdf",
        "--exclude-labels",
        "memory",
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )

    command = [
        "python",
        "plot_bar_diagram.py",
        f"benchmarks/{filename.stem}/{filename.stem}_openmp_jfa.csv",  
        f"benchmarks/{filename.stem}/{filename.stem}_naive_jfa.csv",  
        "--output", 
        f"{filename.stem}/{filename.stem}_jfa_bar_diagram_no_memory_01.pdf",
        "--exclude-labels",
        "memory",
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )
