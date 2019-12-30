import enum
import pickle as pckl


def enum_to_string(item: enum.Enum) -> str:
    return str(item).split(".")[-1]

def pickle(file_path, data):
    output_file = open(file_path, 'wb')
    pckl.dump(data, output_file)
    output_file.close()

def unpickle(file_path):
    input_file = open(file_path, 'rb')
    data = pckl.load(input_file)
    input_file.close()
    return data