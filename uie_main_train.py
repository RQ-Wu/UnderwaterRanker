from requests import options
from runner import UIE_Runner
import argparse
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--opt_path', type=str)
args = parser.parse_args()
options = utils.get_option(args.opt_path)

runner_train = UIE_Runner(options)
runner_train.main_loop()