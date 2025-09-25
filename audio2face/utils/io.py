import io

import numpy as np


def load_npz(file_to_load: io.BytesIO | str,
             float_dtype: np.floating | None = None) -> dict:
    """Load npz file with allow_pickle=True.

    Loads data from a compressed numpy archive file (.npz) with pickle support
    enabled. Handles special cases for scalar arrays and string arrays.

    Args:
        file_to_load (io.BytesIO | str):
            Path to a npz file, or a io.BytesIO object containing npz data.
        float_dtype (np.floating | None, optional):
            Target floating point data type for conversion. If provided,
            all floating point arrays will be converted to this type.
            Defaults to None.

    Returns:
        dict:
            Dictionary containing the loaded data with keys from the npz file.
    """
    with np.load(file_to_load, allow_pickle=True) as npz_file:
        tmp_data_dict = dict(npz_file)
        npz_dict = dict()
        for key, value in tmp_data_dict.items():
            if isinstance(value, np.ndarray) and\
                    len(value.shape) == 0:
                # value is not an ndarray before dump
                value = value.item()
            elif isinstance(value, np.ndarray) and\
                    len(value.shape) == 1 and\
                    isinstance(value[0], np.str_):
                value = value.tolist()
            elif float_dtype is not None and \
                    isinstance(value, np.ndarray) and \
                    value.dtype.kind == 'f':
                value = value.astype(float_dtype)
            npz_dict.__setitem__(key, value)
    return npz_dict




def export_npz(data_dict: dict,
               file_to_save: io.BytesIO | str) -> io.BytesIO | None:
    """Export dictionary to compressed npz file.

    Saves dictionary data as compressed numpy archive format (.npz),
    supporting both file path and memory buffer output.

    Args:
        data_dict (dict):
            Dictionary containing data to be exported.
        file_to_save (io.BytesIO | str):
            File path to save to, or io.BytesIO object for in-memory output.

    Returns:
        io.BytesIO | None:
            Returns the io.BytesIO object (positioned at beginning) if
            file_to_save is io.BytesIO, otherwise returns None.
    """
    np.savez_compressed(file_to_save, **data_dict)
    if isinstance(file_to_save, io.BytesIO):
        file_to_save.seek(0)
        return file_to_save
    return None
