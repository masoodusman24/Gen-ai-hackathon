def categorize_value(value, normal_range):
    """
    Categorize the test value relative to the normal range.
    Returns 'Low', 'Normal', or 'High'.
    """
    if value < normal_range[0]:
        return "Low"
    elif value > normal_range[1]:
        return "High"
    else:
        return "Normal"
