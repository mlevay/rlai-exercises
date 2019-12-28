import enum


def enum_to_string(item: enum.Enum) -> str:
    return str(item).split(".")[-1]