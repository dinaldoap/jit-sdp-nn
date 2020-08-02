# coding=utf-8
def mkdir(dir):
    dir.mkdir(parents=True, exist_ok=True)


def split_args(argv, names):
    cli_tokens = to_cli_tokens(names)
    new_argv = argv
    for cli_token in cli_tokens:
        new_argv = split_arg(new_argv, cli_token)
    return new_argv


def to_cli_tokens(names):
    return ['--{}s'.format(name) for name in names]


def split_arg(argv, name):
    try:
        value_index = argv.index(name) + 1
    except ValueError:  # argument not in list
        return argv
    value = argv[value_index]
    return argv[:value_index] + value.split() + argv[value_index + 1:]


def create_config_template(args, names):
    new_args = dict(args)
    plurals = to_plural(names)
    for plural_name in plurals:
        del new_args[plural_name]
    return new_args


def to_plural(names):
    return ['{}s'.format(name) for name in names]
