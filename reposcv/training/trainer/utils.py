def get_dict(names, values) -> dict:
    result = zip(names, values)
    d = {k: v for k, v in result}
    # print(d)
    return d