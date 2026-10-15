def _clean_ktm_data(df:pd.DataFrame, year:str) -> pd.DataFrame:
    df = df.copy()

    ## 1. Remove ridership that is negative
    df = df[df['ridership'] > 0]

    ## 2. Remove station that is 'Unknown;
    df = df[df['origin'] != 'Unknown']

    ## 3. Remove duplicates
    df = df.drop_duplicates()

    ## 4. Remove rows with missing values
    df = df.dropna()
    
    ## 5. Make sure the year is on the specified year
    
    df = df[df['date'].dt.year == int(year)]

    return df

def get_clean_ktm_data(year:int)->pd.DataFrame:
    """
    Get Official KTMB data from 'https://storage.data.gov.my/transportation/ktmb/{year}.parquet'
    """
    
    URL_DATA = f'https://storage.data.gov.my/transportation/ktmb/komuter_{year}.parquet'
    
    # Read data
    df = pd.read_parquet(URL_DATA, engine='fastparquet')
    
    # Cast datetime format
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    df['datetime'] = df['date'] + pd.to_timedelta(df['time'] + ':00')
        
    return _clean_ktm_data(df, year)

def get_weather_data(weather_features: list) -> pd.DataFrame:
    response = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": 3.139, "longitude": 101.6869,          # Kuala Lumpur
            "start_date": "2024-01-01", "end_date": "2024-12-31",
            "hourly": ",".join(weather_features),
            "timezone": "Asia/Kuala_Lumpur",
        },
        timeout=60,
    )
    response.raise_for_status()
    df =  pd.DataFrame(response.json()["hourly"]).rename(columns={'time':'datetime'})
    
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    return df


def get_temporal_data(year:int)->pd.DataFrame:
    YEAR_HOURS = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
    
    ### First-level (base)
    my_holidays = holidays_lib.Malaysia(years=sorted(set(YEAR_HOURS.year)), subdiv="KUL")
    day_of_week = YEAR_HOURS.dayofweek.to_numpy()
    hour = YEAR_HOURS.hour.to_numpy()

    ### Second-levek (derrived from first-level) 
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    dow_sin = np.sin(2 * np.pi * day_of_week / 7)
    dow_cos = np.cos(2 * np.pi * day_of_week / 7)
    is_weekend = (day_of_week >= 5).astype(float)
    is_holiday = np.array([d in my_holidays for d in YEAR_HOURS.date], dtype=float)
    
    column_name = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend', 'is_holiday']
    
    np_stack = np.stack([
        hour_sin, hour_cos, dow_sin, dow_cos, is_weekend, is_holiday,
    ], axis=1)
    
    return pd.DataFrame(np_stack, columns=column_name, index=YEAR_HOURS).reset_index().rename(columns={'index':'datetime'})

def convert_df_to_numpy(df:pd.DataFrame, cols_to_drop:list[str])->np.ndarray:
    
    return df.drop(columns=cols_to_drop).to_numpy()