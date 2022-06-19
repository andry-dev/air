#!/usr/bin/python3

import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=Path)

    args = parser.parse_args()

    df = pd.read_csv(args.file, low_memory=False)

    df = df.drop(columns=['Unnamed: 0', 'Unnamed: 37'])

    df.to_csv(args.file, index=False)
