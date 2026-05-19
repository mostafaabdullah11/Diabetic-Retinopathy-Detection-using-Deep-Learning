"""
logger.py — Clean per-epoch training logger
"""


class TrainingLogger:
    def __init__(self, fold: int):
        self.fold = fold
        print(f"\n{'='*65}")
        print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>9} | {'QWK':>7} | {'Head LR':>10}")
        print(f"{'─'*65}")

    def log(self, epoch: int, train_loss: float, val_loss: float, qwk: float, lr: float):
        print(f"{epoch:>6} | {train_loss:>10.4f} | {val_loss:>9.4f} | {qwk:>7.4f} | {lr:>10.2e}")
