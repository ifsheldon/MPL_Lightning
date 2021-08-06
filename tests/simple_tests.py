import mpl_lightning as mpl
import pytorch_lightning as pl


def test_data():
    model = mpl.LightningMPL(10, 28, 2)
    data_module = mpl.CIFAR10SSL_DM("../data", 32, 32 * 4, 32, 4000, True, 1000, (2, 10), 1)
    trainer = pl.Trainer(fast_dev_run=1, gpus=1)
    trainer.fit(model, data_module)
    trainer.validate(model, data_module)


if __name__ == "__main__":
    test_data()
