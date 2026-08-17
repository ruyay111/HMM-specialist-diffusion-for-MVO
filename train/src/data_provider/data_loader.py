from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from statsmodels.tsa.stattools import acf
from sklearn.isotonic import IsotonicRegression
import warnings

warnings.filterwarnings('ignore')


class Dataset_Benchmark(Dataset):
    def __init__(self,
                 data_path,
                 seq_len,
                 scale=None,
                 sample_multiplier=4,
                 start_date=None,
                 end_date=None,
                 col=None,
                 **kwargs):
        self.data_path = data_path
        self.seq_len = seq_len
        self.scale = scale
        self.sample_multiplier = sample_multiplier
        self.start_date = pd.to_datetime(start_date) if start_date != 'None' else None
        self.end_date = pd.to_datetime(end_date) if end_date != 'None' else None
        self.col = col + 1 if col is not None else None  # skip date column

        self.__read_data__()
        self.samples_len = int(self.__len__() / self.sample_multiplier)
        self.__get_samples__()

    def __read_data__(self):
        df_raw = pd.read_csv(self.data_path)

        df_raw['date'] = pd.to_datetime(df_raw['date'])
        if self.start_date is not None and self.end_date is None:
            df_raw = df_raw[df_raw['date'] >= self.start_date]
        elif self.start_date is None and self.end_date is not None:
            df_raw = df_raw[df_raw['date'] <= self.end_date]
        elif self.start_date is not None and self.end_date is not None:
            df_raw = df_raw[(df_raw['date'] >= self.start_date) & (df_raw['date'] <= self.end_date)]

        df_raw = df_raw.reset_index(drop=True)
        if self.col is not None:
            self.raw_data = df_raw.iloc[:, [0, self.col]]
            df_data = df_raw.iloc[:, [self.col]]
        else:
            self.raw_data = df_raw
            df_data = df_raw.iloc[:, 1:]
        self.__scale_data__(df_data)

    def __scale_data__(self, df):
        if self.scale == 'z':
            self.scaler = StandardScaler()
            self.data = self.scaler.fit_transform(df)
        elif self.scale == 'minmax':
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
            self.data = self.scaler.fit_transform(df)
        elif self.scale == 'quantile':
            self.scaler = QuantileTransformer(output_distribution='normal')
            self.data = self.scaler.fit_transform(df)
        elif self.scale == 'none':
            self.data = df.values
        else:
            raise ValueError('Invalid scale type')

    def __get_samples__(self):
        samples = np.zeros((self.samples_len, self.seq_len, self.data.shape[1]))
        for i in range(len(samples)):
            samples[i] = self.data[i:i + self.seq_len]
        self.samples = samples

    def __getitem__(self, index):
        idx = np.random.randint(0, len(self.samples))
        return self.samples[idx]

    def __len__(self):
        return (len(self.data) - self.seq_len + 1) * self.sample_multiplier

    def inverse_transform(self, data):
        if self.scale == 'none':
            print('No scaling applied!')
            return data
        else:
            return self.scaler.inverse_transform(data)


class Dataset_RegimeWindows(Dataset):
    """Pre-extracted regime windows of shape (N, seq_len, n_assets)."""

    def __init__(
        self,
        data_path,
        seq_len,
        scale=None,
        sample_multiplier=4,
        **kwargs,
    ):
        self.data_path = data_path
        self.seq_len = seq_len
        self.scale = scale
        self.sample_multiplier = sample_multiplier
        self.__read_data__()
        self.samples_len = int(self.__len__() / self.sample_multiplier)

    def __read_data__(self):
        windows = np.load(self.data_path)
        if windows.ndim != 3:
            raise ValueError(
                f"Expected regime windows of shape (N, seq_len, n_assets); got {windows.shape}"
            )
        if windows.shape[0] == 0:
            raise ValueError(f"No training windows in {self.data_path}")
        if windows.shape[1] != self.seq_len:
            raise ValueError(
                f"Window length {windows.shape[1]} does not match seq_len={self.seq_len}"
            )
        n_windows, seq_len, n_assets = windows.shape
        flat = windows.reshape(n_windows * seq_len, n_assets)
        self.raw_data = pd.DataFrame(flat)
        self.__scale_data__(self.raw_data)
        self.samples = self.data.reshape(n_windows, seq_len, n_assets)

    def __scale_data__(self, df):
        if self.scale == 'z':
            self.scaler = StandardScaler()
            self.data = self.scaler.fit_transform(df)
        elif self.scale == 'minmax':
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
            self.data = self.scaler.fit_transform(df)
        elif self.scale == 'quantile':
            self.scaler = QuantileTransformer(output_distribution='normal')
            self.data = self.scaler.fit_transform(df)
        elif self.scale == 'none':
            self.scaler = None
            self.data = df.values
        else:
            raise ValueError('Invalid scale type')

    def __getitem__(self, index):
        idx = np.random.randint(0, len(self.samples))
        return self.samples[idx]

    def __len__(self):
        return len(self.samples) * self.sample_multiplier

    def inverse_transform(self, data):
        if self.scale == 'none' or self.scaler is None:
            print('No scaling applied!')
            return data
        return self.scaler.inverse_transform(data)


class Dataset_Benchmark_Conditional(Dataset_Benchmark):
    def __init__(self,
                 data_path,
                 regime_path,
                 seq_len,
                 scale=None,
                 sample_multiplier=4,
                 start_date=None,
                 end_date=None,
                 cond_type=None,
                 n_regimes=None,
                 prev_len=None,
                 col=None,
                 **kwargs):
        self.data_path = data_path
        self.regime_path = regime_path
        self.seq_len = seq_len
        self.scale = scale
        self.sample_multiplier = sample_multiplier
        self.start_date = start_date
        self.end_date = end_date
        # idx: only use regime label to create embeddings
        # prob: use regime label and probability to create weighted embeddings
        self.cond_type = cond_type
        self.n_regimes = n_regimes
        self.prev_len = prev_len
        self.col = col + 1 if col is not None else None  # skip date column

        self.__read_data__()
        self.samples_len = int(self.__len__() / self.sample_multiplier)
        self.__get_samples__()

    def __read_data__(self):
        df_raw = pd.read_csv(self.data_path)
        df_regime = pd.read_csv(self.regime_path)

        df_raw['date'] = pd.to_datetime(df_raw['date'])
        df_regime['date'] = pd.to_datetime(df_regime['date'])

        # convert monthly regime to daily regime
        df_regime_daily = pd.DataFrame()
        for _, row in df_regime.iterrows():
            start_date = row['date'].replace(day=1)
            end_date = row['date']
            date_range = pd.date_range(start=start_date, end=end_date)
            daily_df = pd.DataFrame({'date': date_range, 'regime': row['regime'], 'regime_prob': row['regime_prob']})
            df_regime_daily = pd.concat([df_regime_daily, daily_df])
        df_regime_daily = df_regime_daily.reset_index(drop=True)
        df_regime_daily = pd.merge(df_raw[['date']], df_regime_daily, on='date', how='left')
        df_regime_daily = df_regime_daily.reset_index(drop=True)
        df_regime_daily['regime_prob'] = df_regime_daily['regime_prob'].apply(lambda x: eval(x))

        if self.start_date is not None and self.end_date is None:
            df_raw = df_raw[df_raw['date'] >= self.start_date]
            df_regime_daily = df_regime_daily[df_regime_daily['date'] >= self.start_date]
        elif self.start_date is None and self.end_date is not None:
            df_raw = df_raw[df_raw['date'] <= self.end_date]
            df_regime_daily = df_regime_daily[df_regime_daily['date'] <= self.end_date]
        elif self.start_date is not None and self.end_date is not None:
            df_raw = df_raw[(df_raw['date'] >= self.start_date) & (df_raw['date'] <= self.end_date)]
            df_regime_daily = df_regime_daily[
                (df_regime_daily['date'] >= self.start_date) & (df_regime_daily['date'] <= self.end_date)]
        else:
            self.start_date = df_raw['date'].min()
            self.end_date = df_raw['date'].max()
            df_regime_daily = df_regime_daily[
                (df_regime_daily['date'] >= self.start_date) & (df_regime_daily['date'] <= self.end_date)]

        df_raw = df_raw.reset_index(drop=True)
        if self.col is not None:
            self.raw_data = df_raw.iloc[:, [0, self.col]]
            df_data = df_raw.iloc[:, [self.col]]
        else:
            self.raw_data = df_raw
            df_data = df_raw.iloc[:, 1:]

        self.__scale_data__(df_data)

        sorted_regimes = sorted(df_regime_daily['regime'].unique())
        self.regimes_mapping = {regime: i + 1 for i, regime in
                                enumerate(sorted_regimes)}  # 0 is reserved for unconditional
        df_regime_daily['regime_number'] = df_regime_daily['regime'].map(self.regimes_mapping)
        self.regimes = df_regime_daily['regime_number'].values
        self.regime_prob = np.array(df_regime_daily['regime_prob'].tolist())

    def __get_samples__(self):
        samples = np.zeros((self.samples_len, self.seq_len, self.data.shape[1]))
        prev_ts = np.zeros((self.samples_len, self.prev_len, self.data.shape[1]))
        regimes = np.zeros((self.samples_len, self.seq_len))
        regimes_prob = np.zeros((self.samples_len, self.seq_len, self.n_regimes))
        for idx in range(len(samples)):
            start_idx = idx + self.prev_len
            end_idx = start_idx + self.seq_len
            samples[idx] = self.data[start_idx:end_idx]
            prev_ts[idx] = self.data[idx:idx + self.prev_len]
            regimes[idx] = self.regimes[start_idx:end_idx]
            regimes_prob[idx] = self.regime_prob[start_idx:end_idx]
        self.samples = samples
        self.prev_ts = prev_ts
        self.regimes = regimes
        self.regimes_prob = regimes_prob

    def __getitem__(self, index):
        idx = np.random.randint(0, len(self.samples))
        results = dict()
        results['samples'] = self.samples[idx]
        results['regimes'] = self.regimes[idx]
        results['regimes_prob'] = self.regimes_prob[idx]
        results['prev_ts'] = self.prev_ts[idx]
        return results

    def __len__(self):
        return (len(self.data) - self.seq_len - self.prev_len + 1) * self.sample_multiplier


class Dataset_Benchmark_FinDiff(Dataset_Benchmark):
    def __init__(self,
                 data_path,
                 scale=None,
                 sample_multiplier=4,
                 start_date=None,
                 end_date=None,
                 col=None,
                 **kwargs):
        self.data_path = data_path
        self.scale = scale
        self.sample_multiplier = sample_multiplier
        self.start_date = start_date
        self.end_date = end_date
        self.col = col + 1 if col is not None else None  # skip date column

        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(self.data_path)

        df_raw['date'] = pd.to_datetime(df_raw['date'])
        if self.start_date is not None and self.end_date is None:
            df_raw = df_raw[df_raw['date'] >= self.start_date]
        elif self.start_date is None and self.end_date is not None:
            df_raw = df_raw[df_raw['date'] <= self.end_date]
        elif self.start_date is not None and self.end_date is not None:
            df_raw = df_raw[(df_raw['date'] >= self.start_date) & (df_raw['date'] <= self.end_date)]

        df_raw = df_raw.reset_index(drop=True)
        if self.col is not None:
            self.raw_data = df_raw.iloc[:, [0, self.col]]
            df_data = df_raw.iloc[:, [self.col]]
        else:
            self.raw_data = df_raw
            df_data = df_raw.iloc[:, 1:]
        self.__scale_data__(df_data)

    def __scale_data__(self, df):
        if self.scale == 'z':
            self.scaler = StandardScaler()
            self.data = self.scaler.fit_transform(df)
        elif self.scale == 'minmax':
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
            self.data = self.scaler.fit_transform(df)
        elif self.scale == 'quantile':
            self.scaler = QuantileTransformer(output_distribution='normal')
            self.data = self.scaler.fit_transform(df)
        elif self.scale == 'none':
            self.data = df.values
        else:
            raise ValueError('Invalid scale type')

    def __getitem__(self, index):
        idx = np.random.randint(0, len(self.data))
        return self.data[idx]

    def __len__(self):
        return len(self.data) * self.sample_multiplier

    def inverse_transform(self, data):
        if self.scale == 'none':
            print('No scaling applied!')
            return data
        else:
            return self.scaler.inverse_transform(data)


def softmax(x, temperature=1.0):
    x = np.array(x)
    x_scaled = x / temperature
    exp_x = np.exp(x_scaled - np.max(x_scaled))
    return exp_x / np.sum(exp_x)


# Learning to Rank data loader. for two-stage training
class Dataset_Benchmark_LTR(Dataset):
    def __init__(self,
                 data_path,
                 seq_len,
                 scale=None,
                 sample_multiplier=4,
                 start_date=None,
                 end_date=None,
                 seq_len_multiplier=None,
                 **kwargs):
        self.data_path = data_path
        self.seq_len = seq_len
        self.scale = scale
        self.sample_multiplier = sample_multiplier
        self.start_date = start_date
        self.end_date = end_date
        self.seq_len_multiplier = seq_len_multiplier
        self.sample_window = self.seq_len * self.seq_len_multiplier

        self.__read_data__()
        self.samples_len = int(self.__len__() / self.sample_multiplier)
        self.__get_samples__()

    def __read_data__(self):
        df_raw = pd.read_csv(self.data_path)

        df_raw['date'] = pd.to_datetime(df_raw['date'])
        if self.start_date is not None and self.end_date is None:
            df_raw = df_raw[df_raw['date'] >= self.start_date]
        elif self.start_date is None and self.end_date is not None:
            df_raw = df_raw[df_raw['date'] <= self.end_date]
        elif self.start_date is not None and self.end_date is not None:
            df_raw = df_raw[(df_raw['date'] >= self.start_date) & (df_raw['date'] <= self.end_date)]

        df_raw = df_raw.reset_index(drop=True)
        self.raw_data = df_raw
        df_data = df_raw.iloc[:, 1:]
        self.__calc_relevance_score__(df_data)
        self.__scale_data__(df_data)

    def __get_samples__(self):
        samples = np.zeros((self.samples_len, self.sample_window, self.data.shape[1]))
        for i in range(len(samples)):
            samples[i] = self.data[i:i + self.sample_window]
        self.samples = samples

    def __calc_relevance_score__(self, df):
        acf_arr = np.zeros((self.sample_window, len(df.columns)))
        for i in range(len(df.columns)):
            abs_ts = np.abs(df.iloc[:, i].values)
            acf_arr[:, i] = acf(abs_ts, nlags=self.sample_window)[1:]

        raw_scores = np.mean(acf_arr, axis=1)

        x = np.arange(len(raw_scores))
        iso_reg = IsotonicRegression(increasing=False)
        monotonic_scores = iso_reg.fit_transform(x, raw_scores)

        for i in range(1, len(monotonic_scores)):
            if monotonic_scores[i] >= monotonic_scores[i - 1]:
                monotonic_scores[i] = monotonic_scores[i - 1] - 1e-4

        self.relevance_score = monotonic_scores.reshape(-1, 1)

    def __scale_data__(self, df):
        if self.scale == 'z':
            self.scaler = StandardScaler()
            self.data = self.scaler.fit_transform(df)
        elif self.scale == 'minmax':
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
            self.data = self.scaler.fit_transform(df)
        elif self.scale == 'quantile':
            self.scaler = QuantileTransformer(output_distribution='normal')
            self.data = self.scaler.fit_transform(df)
        elif self.scale == 'none':
            self.data = df.values
        else:
            raise ValueError('Invalid scale type')

    def __getitem__(self, index):
        start_idx = index // self.samples_len
        sample = self.samples[start_idx]
        selected_idx = np.random.choice(range(self.sample_window), self.seq_len, replace=False)
        x = sample[selected_idx]
        score = softmax(self.relevance_score[selected_idx])
        return x, score

    def __len__(self):
        return (len(self.data) - self.sample_window + 1) * self.sample_multiplier

    def inverse_transform(self, data):
        if self.scale == 'none':
            print('No scaling applied!')
            return data
        else:
            return self.scaler.inverse_transform(data)


def _resample(data, batch_size, replace=False):
    # Resample the given data sample.
    index = np.random.choice(
        range(data.shape[0]), size=batch_size, replace=replace)
    batch = data[index]
    return batch


class Dataset_Joint_KL_Div(Dataset):
    def __init__(self,
                 X,
                 Y,
                 num_samples=10000):
        self.X = X
        self.Y = Y
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        X_ref = _resample(self.X, 1, replace=True).squeeze()
        Y_ref = _resample(self.Y, 1, replace=True).squeeze()
        return X_ref, Y_ref

    def get_all(self):
        return self.X, self.Y


if __name__ == '__main__':
    data_path = 'F:/Projects/Ask2.ai/Diffusion/warehouse/processed/benchmark_data_log_ret_10.csv'
    regime_path = 'F:/Projects/Ask2.ai/Diffusion/warehouse/processed/macro_regime_3.csv'
    scale = 'quantile'

    dataset = Dataset_Benchmark_LTR(
        data_path=data_path,
        seq_len=128,
        scale=scale,
        sample_multiplier=8,
        seq_len_multiplier=2
    )
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    for i, (batch_data, batch_score) in enumerate(dataloader):
        print(batch_data.shape, batch_score.shape)
