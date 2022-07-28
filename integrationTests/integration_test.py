from datetime import datetime
import pandas as pd
import sys
import pickle
import os


S3_ENDPOINT_URL = 'http://localhost:4566'

def dt(hour, minute, second=0):
    return datetime(2021, 1, 1, hour, minute, second)

data = [
    (None, None, dt(1, 2), dt(1, 10)),
    (1, 1, dt(1, 2), dt(1, 10)),
    (1, 1, dt(1, 2, 0), dt(1, 2, 50)),
    (1, 1, dt(1, 2, 0), dt(2, 2, 1)),        
]

columns = ['PUlocationID', 'DOlocationID', 'pickup_datetime', 'dropOff_datetime']
df = pd.DataFrame(data, columns=columns) 

options = {
    'client_kwargs': {
        'endpoint_url': S3_ENDPOINT_URL
    }
}


df.to_parquet(
    's3://nyc-duration/in/2021-02.parquet',
    engine='pyarrow',
    compression=None,
    index=False,
    storage_options=options
)



def read_data(filename , categorical):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def save_data(df ,output_path):

    df.to_parquet(
    output_path,
    engine='pyarrow',
    compression=None,
    index=False,
    storage_options=options
    )
    

def get_input_path(year, month):
    default_input_pattern = 's3://nyc-duration/in/{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)


def get_output_path(year, month):
    default_output_pattern = 's3://nyc-duration/out/{year:04d}-{month:02d}.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)




 
def main(year , month):
    

    input_file = get_input_path(year, month)
    print(input_file)
    output_file = get_output_path(year, month)

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)


    categorical = ['PUlocationID', 'DOlocationID']

    df = pd.read_parquet(input_file, storage_options=options)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)


    print('predicted sum of duration:', sum(y_pred))


    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    save_data(df_result , output_path=output_file)

if __name__ == "__main__":
    year = int(sys.argv[1])
    month = int(sys.argv[2])

    main(year , month)