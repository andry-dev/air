#!/usr/bin/python3

import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime


def extract_date(value):
    value = value[:16] if len(value) >= 17 else value
    date = datetime.strptime(value, '%d/%m/%Y %H:%M')
    return '{}-{}-{}'.format(date.year, date.month, date.day)


def extract_hour(value):
    value = value[:16] if len(value) >= 17 else value
    date = datetime.strptime(value, '%d/%m/%Y %H:%M')
    return date.hour


def extract_month(value):
    value = value[:16] if len(value) >= 17 else value
    date = datetime.strptime(value, '%d/%m/%Y %H:%M')
    return date.month


def extract_all(value):
    value = value[:16] if len(value) >= 17 else value
    date = datetime.strptime(value, '%d/%m/%Y %H:%M')

    iso_date = '{}-{}-{}'.format(date.year, date.month, date.day)
    return (iso_date, date.month, date.hour)


def find_missing_time_rows(df):
    to_drop = []
    for row in df.iterrows():
        missing_time = len(row[1]['DataOraIncidente']) <= 10
        if missing_time:
            to_drop.append(row[0])

    return to_drop


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=Path)

    args = parser.parse_args()

    df = pd.read_csv(args.file, low_memory=False)
    df = df.drop(find_missing_time_rows(df))

    df['ISODate'] = df['DataOraIncidente'].apply(extract_date)
    df['Month'] = df['DataOraIncidente'].apply(extract_month)
    df['Hour'] = df['DataOraIncidente'].apply(extract_hour)

    print(df)

    df.to_csv(args.file, index=False)
