from multiply_model import MultiplyModel
from lib.datasets import create_dataset
import hydra
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import os
import glob
from omegaconf import OmegaConf

@hydra.main(config_path="confs", config_name="taichi01_base")
def main(opt):
    pl.seed_everything(42)
    print("Working dir:", os.getcwd())
    print(OmegaConf.to_yaml(opt))

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints/",
        filename="{epoch:04d}-{loss}",
        save_on_train_epoch_end=True,
        every_n_epochs=100,
        save_top_k=-1,
        save_last=True)
    logger = WandbLogger(project=opt.project_name, name=f"{opt.exp}/{opt.run}")

    trainer = pl.Trainer(
        # gpus=1,
        devices=1,
        accelerator="gpu",
        callbacks=[checkpoint_callback],
        max_epochs=10000,
        check_val_every_n_epoch=50,
        logger=logger,
        log_every_n_steps=1,
        num_sanity_val_steps=0
    )

    betas_path = os.path.join(hydra.utils.to_absolute_path('..'), 'data', opt.dataset.train.data_dir, 'mean_shape.npy')
    model = MultiplyModel(opt, betas_path)
    trainset = create_dataset(opt.dataset.train)
    validset = create_dataset(opt.dataset.valid)

    if opt.model.is_continue == True:
        # checkpoint = sorted(glob.glob("checkpoints/*.ckpt"))[-1]
        checkpoint = sorted(glob.glob("checkpoints/epoch=*.ckpt"))[-1]
        trainer.fit(model, trainset, validset, ckpt_path=checkpoint)
    else: 
        trainer.fit(model, trainset, validset)


if __name__ == '__main__':
    main()