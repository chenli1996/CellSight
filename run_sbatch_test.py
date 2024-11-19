import argparse

parser = argparse.ArgumentParser(description='GraphGRU Training Script')
parser.add_argument('--data', type=str, default='fsvvd_raw', help='Name of the dataset to use, fsvvd_raw, 8i etc')
parser.add_argument('--pred', type=int, default=4, help='Index of the feature to predict, 2,3,4 etc')
args = parser.parse_args()
print('data:',args.data)
print('pred:',args.pred)