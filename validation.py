def chronological_split(data, validation_fraction=0.2, gap=0):
    """Split ordered rows without shuffling, optionally leaving an embargo gap."""
    if not 0 < validation_fraction < 1:
        raise ValueError('validation_fraction must be between 0 and 1')
    if gap < 0:
        raise ValueError('gap cannot be negative')

    split_index = int(len(data) * (1 - validation_fraction))
    validation_start = split_index + gap
    if split_index == 0 or validation_start >= len(data):
        raise ValueError('not enough rows for the requested split and gap')
    return data.iloc[:split_index], data.iloc[validation_start:]
