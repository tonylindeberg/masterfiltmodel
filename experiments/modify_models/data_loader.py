import pickle
import io

import numpy as np
import pandas as pd
import torch

from data_tools.normalization import normalize


class _CpuUnpickler(pickle.Unpickler):
    """
    Custom unpickler for loading PyTorch objects specifically into CPU memory.
    """

    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def _compute_fft(matrices):
    """
    Computes the Fast Fourier transform of the given matrices.
    :param matrices: Numpy ndarray of shape (num_matrices, x1, x2)
    :return:
    """
    fft_result = np.fft.fftn(matrices, axes=(1, 2))
    fft_shifted = np.fft.fftshift(fft_result, axes=(1, 2))
    return fft_shifted


def load_data(file: str, normalization: str = "min_max", window_size: int = None) -> pd.DataFrame:
    """
    Loads the filters from the given file and creates a pandas dataframe.
    :param file: path to the file containing the filters.
    :param normalization: Normalization method for filters, for choices look at the `_normalize` function.
    :param window_size: Filter a specific size of filters, helps when model had different filter sizes.
    :return: Pandas dataframe containing the filters.
    """
    with open(file, "rb") as input_file:
        conv_weights = _CpuUnpickler(input_file).load()
        conv_weights = [np.squeeze(t.detach().numpy()) for t in conv_weights]
        conv_weights = [(idx, filter_) for idx, layer in enumerate(conv_weights) for filter_ in layer]

        # Creating columns "Layer" and "OriginalFilter" corresponding to the layer index and the filter itself.
        df = pd.DataFrame(conv_weights, columns=['Layer', 'OriginalFilter'])
        if window_size:
            df = df[df['OriginalFilter'].apply(lambda x: len(x) == window_size)]
        del conv_weights

        # Removing rows that contain NaN values
        df = df.dropna()
        if df.empty:
            return None
        df = df[df['OriginalFilter'].apply(lambda x: np.any((np.abs(x) > 0.0001)))]  # removing rows with all zeros

        # Creating column "Filter" corresponding to the normalized original filter.
        filters = normalize(np.stack(df["OriginalFilter"].values), normalization)

        # Creating column "Fourier" corresponding to Fourier transform of the normalized filter.
        ffts = _compute_fft(filters)
        magnitudes = np.abs(ffts)  # Compute magnitudes
        phases = np.cos(np.angle(ffts)) + 1 # Compute phases

        df['Filter'] = list(filters)
        df['Fourier'] = list(ffts)
        df['FFT_Magnitudes'] = list(magnitudes)
        df['FFT_Phases'] = list(phases)

        return df
