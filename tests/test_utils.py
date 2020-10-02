# coding=utf-8
from jitsdp import utils

def test_standard_run_command():
    assert 'jitsdp borb' == utils.standard_run_command(['/home/jitsdp', 'borb'])