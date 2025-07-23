import subprocess
import re
import csv
from pathlib import Path
import argparse

def to_snake_case(name):
    name = name.replace("::", "__")
    name = re.sub(r'(?<=[a-z0-9])([A-Z])', r'_\1', name)
    name = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', name)
    name = name.lower()
    name = re.sub(r'__+', '__', name)
    return name

parser = argparse.ArgumentParser(description="Benchmark runner")
parser.add_argument("--niter", type=int, default=10, help="Number of iterations per test")
parser.add_argument("--folder", type=str, default="./tests", help="Folder with input files")
parser.add_argument("--maxsize", type=int, default=128, help="Maximum size (power of 2, starting from 32)")
parser.add_argument("--output", type=str, default="benckmarks", help="Output folder for CSVs")
parser.add_argument("--no-sdf", action="store_true", help="Disable the conditional -s flag for the executable.")
parser.add_argument("--types", nargs="+", default=["0", "1", "2"], help="List of algorithm types to run (e.g., 0 1 2).")
args = parser.parse_args()

N_ITER = args.niter
TESTS_ASSETS_FOLDER = Path(args.folder)
OUTPUT_FOLDER = args.output
OUTPUT_FOLDER_PATH = Path(OUTPUT_FOLDER)
OUTPUT_FOLDER_PATH.mkdir(exist_ok=True)

sizes = []
size = 32
while size <= args.maxsize:
    sizes.append(str(size))
    size *= 2

EXEC_NAME = "./build/Release/apps/cli/cli"

for filename in TESTS_ASSETS_FOLDER.iterdir():
    if not filename.is_file():
        continue

    all_data = {}

    for type in args.types:
        for size in sizes:
            active_sdf = ""
            if not args.no_sdf and int(size) <= 1024:
                active_sdf = "-s"

            try:
                command = [
                    EXEC_NAME,
                    str(filename),
                    f"-n{size}",
                    f"-t{type}",
                    f"-m{N_ITER}",
                    f"-p2"
                ]
                if active_sdf:
                    command.append(active_sdf)

                print(f"Running: {' '.join(command)}")
                result = subprocess.run(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True
                )
                
                current_iteration_data = {}
                
                for line in result.stdout.splitlines():
                    match = re.search(r"\[(.*)\]: ([\d.]+) ms", line)
                    if match:
                        label = match.group(1)
                        label_clean = re.sub(r"\s*\(.*?\)", "", label)
                        parts = label_clean.split("::")
                        mainname = to_snake_case(parts[0])
                        fullname = to_snake_case(label_clean)
                        value = float(match.group(2))

                        if mainname not in all_data:
                            all_data[mainname] = {}
                        if size not in all_data[mainname]:
                            all_data[mainname][size] = []

                        is_main_algorithm_line = "__" not in fullname
                        
                        current_iteration_data[fullname] = current_iteration_data.get(fullname, 0) + value

                        if is_main_algorithm_line:
                            all_data[mainname][size].append(current_iteration_data.copy())
                            current_iteration_data.clear()

            except subprocess.CalledProcessError as e:
                print(f"Error running {EXEC_NAME} with file {filename.name}:")
                print(f"Return code: {e.returncode}")
                print(f"STDOUT:\n{e.stdout}")
                print(f"STDERR:\n{e.stderr}")
                assert False, f"Error executing the command: {e}"

    for mainname, data in all_data.items():
        all_fullnames = set()
        for size_data_list in data.values():
            for iter_dict in size_data_list:
                all_fullnames.update(iter_dict.keys())
        all_fullnames = sorted(all_fullnames)

        output_csv = OUTPUT_FOLDER_PATH / f"{filename.stem}_{mainname}.csv"
        with open(output_csv, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["size"] + all_fullnames)
            for size in sorted(data.keys(), key=int):
                for iter_dict in data[size]:
                    row = [size]
                    for fullname in all_fullnames:
                        row.append(iter_dict.get(fullname, ""))
                    writer.writerow(row)
