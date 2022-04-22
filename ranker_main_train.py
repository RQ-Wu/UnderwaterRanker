from runner import Ranker_Runner, UIE_Runner
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--opt_path', type=str)
args = parser.parse_args()

runner_train = Ranker_Runner(args.opt_path)
runner_train.main_loop()