from collections import namedtuple
from typing import NamedTuple


def convert_dict_to_tuple(dictionary: dict) -> NamedTuple:
    for key, value in dictionary.items():
        if isinstance(value, dict):
            dictionary[key] = convert_dict_to_tuple(value)
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)