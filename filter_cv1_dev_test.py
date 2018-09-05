import pandas
import sys
import os


mixed = pandas.read_csv(sys.argv[1], encoding='utf-8')
dev_paths = pandas.read_csv(sys.argv[2], encoding='utf-8')
test_paths = pandas.read_csv(sys.argv[3], encoding='utf-8')

dev_paths['path'] = '/snakepit/shared/data/mozilla/CommonVoice/v2.0-alpha1.0/en/valid/' + dev_paths['path'].astype(str)
test_paths['path'] = '/snakepit/shared/data/mozilla/CommonVoice/v2.0-alpha1.0/en/valid/' + test_paths['path'].astype(str)

dev_indices = mixed['wav_filename'].isin(dev_paths['path'])
test_indices = mixed['wav_filename'].isin(test_paths['path'])
train_indices = ~(dev_indices | test_indices)

output_folder = sys.argv[4]
mixed[dev_indices].to_csv(os.path.join(output_folder, 'cv_en_valid_dev.csv'), index=False)
mixed[test_indices].to_csv(os.path.join(output_folder, 'cv_en_valid_test.csv'), index=False)
mixed[train_indices].to_csv(os.path.join(output_folder, 'cv_en_valid_train.csv'), index=False)
