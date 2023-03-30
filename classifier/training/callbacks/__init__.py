from .clr import LrFinder, SuperConvergence, WarmRestart
from .standard_callbacks import (CSVLogger, EarlyStopping, ModelCheckpoint,
                                 ReduceLROnPlateau, Tensorboard)

__mapping__ = {
    "csv_logger": CSVLogger,
    "tensorboard": Tensorboard,
    "plateau": ReduceLROnPlateau,
    "early_stopping": EarlyStopping,
    "checkpoint": ModelCheckpoint,
    "super_convergence": SuperConvergence,
    "lr_finder": LrFinder,
    "warm_restart": WarmRestart,
}
