import re


def natural_sort_key(key):
    """
    Generate a sort key that handles numeric portions naturally.
    For example, this ensures '2' comes before '10' rather than after.

    Splits the string representation into chunks of digits and non-digits,
    converting digit chunks to integers for proper numeric comparison.
    """
    key_str = str(key)
    # Split into numeric and non-numeric parts
    parts = re.split(r"(\d+)", key_str)
    # Convert numeric parts to integers for proper sorting
    result = []
    for part in parts:
        if part.isdigit():
            result.append((0, int(part)))  # 0 prefix to sort numbers before strings
        elif part:
            result.append((1, part.lower()))  # 1 prefix, lowercase for case-insensitive
    return result
