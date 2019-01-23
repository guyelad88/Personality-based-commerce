import numpy as np
import pandas as pd
from gensim.models import KeyedVectors


def load_gensim_kv(path_embd, vector_size, binary=False, limit=None):
    kv = KeyedVectors(vector_size=vector_size).load_word2vec_format(path_embd, binary=binary, limit=limit)
    return kv

def list_of_strings_to_list_of_lists(list_of_strings):
    list_of_lists = [[e for e in s.split()] for s in list_of_strings]

    return list_of_lists


def list_of_lists_to_list_of_strings(list_of_lists):
    return [" ".join(lst) for lst in list_of_lists]


def get_len(element):
    if isinstance(element, (tuple, list, dict)):
        return len(element)
    elif isinstance(element, (np.ndarray, pd.DataFrame, pd.Series)):
        return element.shape[0]
    else:
        raise TypeError("unsupported argument type")


def smartdict_init(dct, default):

    sd = SmartDict(default)
    for k, v in dct.items():
        sd[k] = v

    return sd


class SmartDict(dict):
    def __init__(self, default):
        self.default = default

        self.fn = True if callable(default) else False

    def __missing__(self, key):
        return self.default(key) if self.fn else self.default


def _pprint(params, offset=0, printer=repr):
    """Pretty print the dictionary 'params'

    Parameters
    ----------
    params : dict
        The dictionary to pretty print

    offset : int
        The offset in characters to add at the begin of each line.

    printer : callable
        The function to convert entries to strings, typically
        the builtin str or repr

    """
    # Do a multi-line justified repr:
    options = np.get_printoptions()
    np.set_printoptions(precision=5, threshold=64, edgeitems=2)
    params_list = list()
    this_line_length = offset
    line_sep = ',\n' + (1 + offset // 2) * ' '
    for i, (k, v) in enumerate(sorted(params.items())):
        if type(v) is float:
            # use str for representing floating point numbers
            # this way we get consistent representation across
            # architectures and versions.
            this_repr = '%s=%s' % (k, str(v))
        else:
            # use repr of the rest
            this_repr = '%s=%s' % (k, printer(v))
        if len(this_repr) > 500:
            this_repr = this_repr[:300] + '...' + this_repr[-100:]
        if i > 0:
            if this_line_length + len(this_repr) >= 75 or '\n' in this_repr:
                params_list.append(line_sep)
                this_line_length = len(line_sep)
            else:
                params_list.append(', ')
                this_line_length += 2
        params_list.append(this_repr)
        this_line_length += len(this_repr)

    np.set_printoptions(**options)
    lines = ''.join(params_list)
    # Strip trailing space to avoid nightmare in doctests
    lines = '\n'.join(l.rstrip(' ') for l in lines.split('\n'))
    return lines
