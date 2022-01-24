# importing the module
import json
# declaringa a class
class Obj:

    # constructor
    def __init__(self, dict1):
        self.__dict__.update(dict1)



def dict2obj(nested_dict,max_level=float("inf")):
    """
    transform a nested dictionnary to a python object
    with key accessed as attributes, traversing the
    nested dictionnary max_level times
    :param nested_dict:
    :param max_level: int greater than 1
    :return:
    """
    nested_dict_copy = nested_dict.copy()
    for (key,value) in nested_dict_copy.items():
        if isinstance(value,dict):
            nested_dict_copy[key] = dict2obj(value,max_level-1)
    obj = Obj(nested_dict_copy)
    return obj

# def dict2obj_for_nested_class(nested_dict,max_level=2):
#     for level in max_level:
#         nested_dict = dict2obj(nested_dict)
