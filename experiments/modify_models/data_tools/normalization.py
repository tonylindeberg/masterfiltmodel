import numpy as np


def normalize(x: np.ndarray, form: str = "center_norm"):
    """
    Normalizes the input data.
    :param x: Numpy ndarray input
    :param form: Type of normalization, it can be "min_max", "center", "center_norm", "sum", or None.
    In the case of None the function will not normalize the data and will return the x.
    :return:
    """
    x = x.copy()
    match form:
        case None:
            pass
        case "min_max":
            x -= x.min(axis=(1, 2), keepdims=True)
            x /= x.max(axis=(1, 2), keepdims=True)
        case "sum":
            x -= x.min(axis=(1, 2), keepdims=True)
            x /= x.sum(axis=(1, 2), keepdims=True)
        case "center":
            x -= x.mean(axis=(1, 2), keepdims=True)
        case "center_norm":
            x -= x.mean(axis=(1, 2), keepdims=True)
            x /= np.linalg.norm(x, axis=(1, 2), keepdims=True)

        case _:
            raise ValueError("""The parameter 'form' is not supported.""")

    return x
