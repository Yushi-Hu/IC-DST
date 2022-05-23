# sample a 1% training set (same for 2.1 and 2.4)
python sample.py --input_fn mwz2.1/train_dials.json --target_fn mw21_1p_train_v1.json --ratio 0.01 --seed 88

# sample a 1% training set (same for 2.1 and 2.4)
python sample.py --input_fn mwz2.1/train_dials.json --target_fn mw21_1p_train_v2.json --ratio 0.01 --seed 42

# sample a 1% training set (same for 2.1 and 2.4)
python sample.py --input_fn mwz2.1/train_dials.json --target_fn mw21_1p_train_v3.json --ratio 0.01 --seed 888

# sample a 5% training set (same for 2.1 and 2.4)
python sample.py --input_fn mwz2.1/train_dials.json --target_fn mw21_5p_train_v1.json --ratio 0.05 --seed 88

# sample a 5% training set (same for 2.1 and 2.4)
python sample.py --input_fn mwz2.1/train_dials.json --target_fn mw21_5p_train_v2.json --ratio 0.05 --seed 42

# sample a 5% training set (same for 2.1 and 2.4)
python sample.py --input_fn mwz2.1/train_dials.json --target_fn mw21_5p_train_v3.json --ratio 0.05 --seed 888

# sample a 10% training set (same for 2.1 and 2.4)
python sample.py --input_fn mwz2.1/train_dials.json --target_fn mw21_10p_train_v1.json --ratio 0.1 --seed 88

# sample a 10% training set (same for 2.1 and 2.4)
python sample.py --input_fn mwz2.1/train_dials.json --target_fn mw21_10p_train_v2.json --ratio 0.1 --seed 42

# sample a 10% training set (same for 2.1 and 2.4)
python sample.py --input_fn mwz2.1/train_dials.json --target_fn mw21_10p_train_v3.json --ratio 0.1 --seed 888

# process MultiWOZ 2.1 and 2.4 train set (they are the same)
python sample.py --input_fn mwz2.1/train_dials.json --target_fn mw21_100p_train.json --ratio 1.0

# MultiWOZ 2.4 10% dev set for evaluation
python sample.py --input_fn mwz2.4/dev_dials.json --target_fn mw24_10p_dev.json --ratio 0.1 --seed 88

# process MultiWOZ 2.1 dev set
python sample.py --input_fn mwz2.1/dev_dials.json --target_fn mw21_100p_dev.json --ratio 1.0

# process MultiWOZ 2.1 test set
python sample.py --input_fn mwz2.1/test_dials.json --target_fn mw21_100p_test.json --ratio 1.0

# process MultiWOZ 2.4 dev set
python sample.py --input_fn mwz2.4/dev_dials.json --target_fn mw24_100p_dev.json --ratio 1.0

# process MultiWOZ 2.4 test set
python sample.py --input_fn mwz2.4/test_dials.json --target_fn mw24_100p_test.json --ratio 1.0