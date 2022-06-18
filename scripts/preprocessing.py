#!/usr/bin/python3

import pandas as pd
from pathlib import Path, PurePath
import math
import argparse

import requests
import multiprocessing

from datetime import datetime


def geocode(street, ip='[::1]', port=8080):

    payload = {
        'street': street,
        'city': 'Roma',
        'format': 'geojson'
    }

    req = requests.get(f'http://{ip}:{port}/search',
                       params=payload, headers={'Connection': 'close'})

    return req.json()['features']


def extract_coordinates(matches):
    if len(matches) == 0:
        return [8888, 8888]

    return matches[0]['geometry']['coordinates']


def coordinates_legal(coordinates):
    # The street is wrong or obscure
    if coordinates[0] == 8888 and coordinates[1] == 8888:
        return False

    # 42.82125541042326, 11.774008325698063
    # 41.220224002262036, 14.029342839702176
    # Check if the coordinates are inside Lazio's bounds
    if (11.774008325698063 < coordinates[0] and coordinates[0] < 14.029342839702176) and \
            (41.220224002262036 < coordinates[1] and coordinates[1] < 42.82125541042326):
        return True

    return False


def try_parse_number(value):
    if isinstance(value, int) or isinstance(value, float):
        return value

    try:
        return int(value)
    except ValueError:
        return None


def extract_iso_date(value):
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


def clean_data(df: pd.DataFrame):
    if 'Longitudine' in df.columns:
        df = df.rename(columns={'Longitudine': 'Longitude',
                                'Latitudine': 'Latitude'})

    invalid_rows = []

    for row in df.iterrows():
        missing_time = len(row[1]['DataOraIncidente']) <= 10
        if missing_time:
            print(f'Dropping row #{row[0]}: Missing date time.')
            invalid_rows.append(row[0])
            continue

        number_injured = try_parse_number(row[1]['NUM_FERITI'])
        if number_injured is None:
            print(f'Dropping row #{row[0]}: NUM_FERITI can\'t be parsed')
            invalid_rows.append(row[0])
            continue
        else:
            df.at[row[0], 'NUM_FERITI'] = number_injured

        row_longitude = row[1]['Longitude']
        row_latitude = row[1]['Latitude']
        if (type(row_longitude) == float and math.isnan(row_longitude)) or \
           (type(row_latitude) == float and math.isnan(row_latitude)):

            query_str = row[1]['STRADA1']

            strada02 = row[1]['Strada02']

            if type(strada02) == str and 'civico' in strada02:
                civic_number = (row[1]['Chilometrica'])
                query_str += f' {civic_number}'

            found_matches = geocode(query_str)
            coordinates = extract_coordinates(found_matches)

            if not coordinates_legal(coordinates):
                invalid_rows.append(row[0])
                print(f'Dropping row #{row[0]}: Invalid coordinates.')
                continue

            df.at[row[0], 'Longitude'] = coordinates[0]
            df.at[row[0], 'Latitude'] = coordinates[1]
        else:
            if isinstance(row_latitude, str):
                df.at[row[0], 'Latitude'] = float(
                    row_latitude.replace(',', '.'))

            if isinstance(row_longitude, str):
                df.at[row[0], 'Longitude'] = float(
                    row_longitude.replace(',', '.'))

    print(
        f'Dropping {len(invalid_rows)} rows with invalid data: {invalid_rows}')
    df = df.drop(invalid_rows)
    df = df.drop(columns=['Localizzazione1', 'STRADA1',
                          'Localizzazione2', 'STRADA2', 'Strada02',
                          'Chilometrica', 'DaSpecificare', 'Confermato'])

    df['ISODate'] = df['DataOraIncidente'].apply(extract_iso_date)
    df['Month'] = df['DataOraIncidente'].apply(extract_month)
    df['Hour'] = df['DataOraIncidente'].apply(extract_hour)
    df['Deadly'] = df['NUM_MORTI'] > 0
    df['Injured'] = df['NUM_FERITI'] > 0

    return df


def generate_clean_filepath(raw_filepath):
    parts = list(raw_filepath.parts)
    parts[1] = 'clean'
    new_path = PurePath()
    for part in parts:
        new_path = new_path / part
    return new_path


def create_clean_dataframe(filepath):
    new_df = pd.read_csv(filepath, delimiter=';',
                         on_bad_lines='error', verbose=False)

    new_path = generate_clean_filepath(filepath)
    print(new_path)

    new_df = clean_data(new_df)
    new_df.to_csv(new_path)

    return new_df


def read_csvs(directory):
    raw_datasets = list(directory.glob('**/*.csv'))
    POOL_SIZE = 32
    with multiprocessing.Pool(POOL_SIZE) as p:
        res = p.map(create_clean_dataframe, raw_datasets)
        return pd.concat(res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=Path)
    args = parser.parse_args()

    df = read_csvs(args.dir)
    # df = df.drop(columns=['Unnamed: 37'])
    df.to_csv(args.dir / '../aggregate.csv', index=False)
