from runner import UIE_Runner
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--opt_path', type=str)
args = parser.parse_args()

runner_train = UIE_Runner(args.opt_path)
runner_train.main_loop()