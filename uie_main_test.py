import argparse

from yaml import parse
from runner import UIE_Runner

parser = argparse.ArgumentParser()
parser.add_argument('--opt_path', type=str)
args = parser.parse_args()

test_runner = UIE_Runner(args.opt_path, type='test')
test_runner.main_test_loop()
    