import pandas as pd
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    args = parser.parse_args()

    ip = f"{args.dir}/output.json"
    op = f"{args.dir}/output.csv"

    df = pd.read_json(ip)
    df['conv_id'] = df['conv_id'].astype(int)
    df['turn_label'] = df['turn_label'].astype(int)
    df = df.sort_values(['conv_id','turn_label'])
    df.to_csv(op, index=False)

if __name__ == "__main__":
    main()