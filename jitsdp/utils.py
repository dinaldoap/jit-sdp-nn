def mkdir(dir):
    dir.mkdir(parents=True, exist_ok=True)


def split_arg(argv, name):
    value_index = argv.index(name) + 1
    value = argv[value_index]
    return argv[:value_index] + value.split() + argv[value_index + 1:]
