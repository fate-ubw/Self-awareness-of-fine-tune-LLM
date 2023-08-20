import json
import argparse
import pdb

pdb.set_trace()
parser = argparse.ArgumentParser()
parser.add_argument("--input_files")
parser.add_argument("--output_path")
args = parser.parse_args()

file_list = args.input_files.split(",")

merge_data = []
for filename in file_list:
    with open(filename,"r") as f:
        data = json.load(f)
        merge_data.extend(data)
with open(args.output_path, "w", encoding="utf-8") as f:
    json.dump(merge_data, f, indent=4, ensure_ascii=False)
    print("merge success!")
