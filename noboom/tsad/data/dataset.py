from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import pandas as pd
import os


class Dataset:
    """Simple dataset class that loads all sequences of a dataset into a list of numpy arrays.

    :param dataset: Name of the dataset.
    :param version: Version of the dataset of shape 'major.minor.patch'.
    :param root: Root directory for the dataset.
    :param train: Load the training set if true and the test set if false.
    :param binary_labels: If set, converts labels to traditional binary labels.
    :param download: If set, downloads the data to root.
    :param train_anomalies: If set, include anomalies in the training set.
    :param include_misc_faults: If set, include miscellaneous faults as anomalies.
    :param include_controller_faults:  If set, include controller faults as anomalies.
    :param fast_load: Save timeseries as .parquet for faster loading.
    """

    def __init__(self, dataset: str, version: str, root: str, train: bool = True,
                 binary_labels: bool = False, download: bool = False, train_anomalies: bool = False,
                 include_misc_faults: bool = False, include_controller_faults: bool = False,
                 fast_load: bool = False, cv_thresh: float = 1e-3):

        self.name = dataset
        self.version = version
        self.root = root
        self.train = train
        self.binary_labels = binary_labels
        self.train_anomalies = train_anomalies
        self.include_misc_faults = include_misc_faults
        self.include_controller_faults = include_controller_faults
        self.fast_load = fast_load
        self.cv_thresh = cv_thresh

        self._features: list[str] = []
        self._samples = []
        self._targets = []

        self.load()

    def _process_ts(self, time_series: pd.DataFrame):
        if self.include_misc_faults and self.label_misc_feature is not None:
            mask = (time_series[self.label_misc_feature] > 0) & (time_series[self.label_feature] == 0)
            time_series.loc[mask, self.label_feature] = time_series.loc[mask, self.label_misc_feature]

        if self.include_controller_faults and self.label_controller_feature is not None:
            mask = (time_series[self.label_controller_feature] > 0) & (time_series[self.label_feature] == 0)
            time_series.loc[mask, self.label_feature] = time_series.loc[mask, self.label_controller_feature]

        if self.binary_labels:
            targets = (time_series[self.label_feature].to_numpy() > 0).astype(int)
        else:
            targets = time_series[self.label_feature].to_numpy().astype(int)

        time_series.drop(self.meta_data, axis=1, inplace=True)

        time_series = time_series.astype(np.float32)

        self._samples.append(time_series.to_numpy())
        self._targets.append(targets)

    def _process_ts_npy(self, time_series: Tuple[np.ndarray, np.ndarray]):
        time_series, targets = time_series
        if self.binary_labels:
            targets = (targets > 0).astype(int)
        else:
            targets = targets.astype(int)
        time_series = time_series.astype(np.float32)
        if time_series.ndim == 3:
            self._samples.extend(time_series)
            self._targets.extend(targets)
        else:
            self._samples.append(time_series)
            self._targets.append(targets)

    def load(self):
        if self.train:
            if self.train_anomalies:
                prefix = 'train'
            else:
                prefix = 'train_normal'
        else:
            prefix = 'test'

        root_path = Path(self.root) / self.name
        dataset_files = sorted(
            path for path in root_path.rglob('*.csv') if prefix in path.name
        )
        if dataset_files:
            mode = 'PANDAS'
            def load_func_(path):
                parquet_path = path.with_suffix('.parquet')
                try:
                    time_series = pd.read_parquet(parquet_path)
                except FileNotFoundError:
                    time_series = pd.read_csv(file_path)
                    if self.fast_load:
                        time_series.to_parquet(parquet_path)
                return time_series
            load_func = load_func_
        else:
            mode = 'NPY'
            if self.train:
                prefix = 'train'
            else:
                prefix = 'test'

            dataset_files = sorted(path for path in root_path.rglob("*.npy") if prefix in path.name and 'labels' not in path.name)
            def load_func_(path):
                data = np.load(path)
                if not self.train:
                    path = Path(path)
                    targets_path = path.with_name(path.stem + '_labels.npy')
                    targets = np.load(targets_path)
                else:
                    targets = np.zeros(data.shape[:-1], dtype=np.int32)
                return data, targets
            load_func = load_func_

        for file_path in dataset_files:
            time_series = load_func(file_path)
            if mode == 'PANDAS':
                self._process_ts(time_series)
            elif mode == 'NPY':
                self._process_ts_npy(time_series)
            else:
                raise ValueError(f'Unknown mode: {mode}')
            del time_series

    def __getitem__(self, item) -> tuple[np.ndarray, np.ndarray]:
        return self._samples[item], self._targets[item]

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def seq_len(self) -> list[int]:
        return [ts.data.shape[0] for ts in self._samples]

    @property
    def num_features(self) -> int:
        return self._samples[0].shape[-1]

    @property
    def ndim(self) -> int:
        return self._samples[0].ndim

    @property
    def window_size(self) -> Optional[int]:
        if self.ndim == 3:
            return self._samples[0].shape[1]
        else:
            return None

    @property
    def label_feature(self):
        return 'Anomaly' if self.name == 'industry_process' else 'Label (advanced/hard fault)'

    @property
    def label_misc_feature(self):
        return None if self.name == 'industry_process' else 'Label (advanced/soft fault)'

    @property
    def label_controller_feature(self):
        return None if self.name == 'industry_process' else 'Label (advanced/controller fault)'

    @property
    def meta_data(self):
        return ['Anomaly'] if self.name == 'industry_process' else ['Time', 'Label (common/hard fault)',
                                                                    'Label (common/soft fault)',
                                                                    'Label (common/controller fault)',
                                                                    'Label (common/hard and soft)',
                                                                    'Label (common/all)',
                                                                    'Label (advanced/hard fault)',
                                                                    'Label (advanced/soft fault)',
                                                                    'Label (advanced/controller fault)']
