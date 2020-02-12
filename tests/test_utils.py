from jitsdp.utils import split_arg


def test_split_arg():
    args = ['exec', '--arg_to_split', 'value1 value2',
            '--arg_ok', 'value1', 'value2']
    splitted = split_arg(args, '--arg_to_split')
    expected = ['exec', '--arg_to_split', 'value1',
                'value2', '--arg_ok', 'value1', 'value2']
    assert expected == splitted
