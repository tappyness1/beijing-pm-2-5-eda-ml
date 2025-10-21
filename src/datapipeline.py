from ucimlrepo import fetch_ucirepo
import pandas as pd

def one_hot_encode_cbwd(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode the `cbwd` column in the DataFrame.

    Args:
        df (pandas.DataFrame): Input DataFrame containing the `cbwd` column.

    Returns:
        pandas.DataFrame: DataFrame with one-hot encoded `cbwd` columns added and the original
        `cbwd` and `cbwd_NW` columns removed.
    """
    # one-hot encode cbwd
    cbwd_dummies = pd.get_dummies(df['cbwd'], prefix='cbwd')
    df = pd.concat([df, cbwd_dummies], axis=1)
    # optional: drop original categorical column
    df.drop(columns=['cbwd', 'cbwd_NW'], inplace=True)
    return df

def sliding_window_mean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the 3-point moving average for specified features in the DataFrame.

    Args:
        df (pandas.DataFrame): Input DataFrame containing the features.

    Returns:
        pandas.DataFrame: DataFrame with additional columns for the moving averages of specified features.
    """
    features_list = ['DEWP', 'TEMP', 'PRES', 'Iws']
    for feature in features_list:
        df[f'{feature}_ma3'] = df[feature].rolling(window=3).mean()

    return df

def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag features for the `pm2.5` column in the DataFrame.

    Args:
        df (pandas.DataFrame): Input DataFrame containing the `pm2.5` column.

    Returns:
        pandas.DataFrame: DataFrame with additional lag features added.
    """
    target_map = df['pm2.5'].to_dict()
    # both shift and map seem the same to me
    # df['lag1'] = df['pm2.5'].shift(24*7)
    df['lag1'] = (df.index - pd.Timedelta('7 days')).map(target_map)
    df['lag2'] = (df.index - pd.Timedelta('14 days')).map(target_map)
    df['lag3'] = (df.index - pd.Timedelta('21 days')).map(target_map)
    return df

class Datapipeline:
    def __init__(self):
        """
        Initializes the DataPipeline class by fetching the PM2.5 dataset for Beijing.
        """
        beijing_pm2_5 = fetch_ucirepo(id=381) 
        self.df = beijing_pm2_5['data']['original']

    def run_data_pipeline(self):
        """
        Clean and transform the dataset for analysis.

        Steps:
            1. Combine `year`, `month`, `day`, and `hour` into a `datetime` column.
            2. Fill missing values using backward fill.
            3. Add rolling mean features for numeric columns.
            4. One-hot encode the `cbwd` column.
            5. Drop unnecessary columns: `No`, `year`, `month`, `day`, `hour`.

        Returns:
            pandas.DataFrame: The transformed DataFrame.
        """
        self.df['datetime'] = pd.to_datetime(self.df[['year', 'month', 'day', 'hour']])
        self.df.set_index('datetime', inplace=True)
        self.df = sliding_window_mean(self.df)
        self.df = one_hot_encode_cbwd(self.df)
        self.df.drop(columns=['No', 'year', 'month', 'day', 'hour'], inplace=True)
        self.df = self.df.bfill()
        self.df = add_lags(self.df)
        
        return self.df