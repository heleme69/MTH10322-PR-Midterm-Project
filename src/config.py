"""Các cấu hình thống nhất cho notebook và demo (tập đầy đủ)."""
SEED = 42
ES_PATIENCE = 5
ES_MONITOR = "val_acc"
ES_MAX_EPOCHS = {"Softmax Regression": 60, "MLP": 40}
RF_MAX_DEPTH = 15
RF_MAX_SAMPLES = 15000
RF_N_BINS = 16
RF_MIN_SAMPLES_LEAF = 5
