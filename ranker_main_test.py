import argparse

from yaml import parse
from runner import Ranker_Runner
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--opt_path', type=str)
parser.add_argument('--test_ckpt_path', type=str)
args = parser.parse_args()

options = utils.get_option(args.opt_path)
if args.test_ckpt_path:
    options['test']['test_ckpt_path'] = args.test_ckpt_path
    
test_runner = Ranker_Runner(options, type='test')
test_runner.main_test_loop()
    