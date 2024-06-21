import pandas as pd
from datetime import datetime, timedelta

def data():
    df = pd.read_csv('DATA/upbit원본.csv')
    df['candle_date_time_kst'] = pd.to_datetime(df['candle_date_time_kst'])
    date_range = pd.date_range(start=df['candle_date_time_kst'].min(), end=df['candle_date_time_kst'].max(), freq='h')

    # 누락된 날짜 식별
    existing_dates = set(df['candle_date_time_kst'])
    complete_dates = set(date_range)
    missing_dates = complete_dates - existing_dates

    # 누락된 날짜를 데이터프레임으로 생성
    missing_dates_df = pd.DataFrame(sorted(missing_dates), columns=['candle_date_time_kst'])

    # 원본 데이터프레임과 누락된 날짜 데이터프레임 병합
    combined_df = pd.concat([df, missing_dates_df]).sort_values(by='candle_date_time_kst').reset_index(drop=True)
    combined_df.drop(columns=['market','candle_date_time_utc','unit','timestamp'], inplace=True)
    df = combined_df.ffill()
    df['target'] = df['trade_price'].diff().apply(lambda x: 1 if x > 0 else 0)
    
    return df

