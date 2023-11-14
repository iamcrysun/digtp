"""
Reference: https://github.com/ramsrigouthamg/Paraphrase-any-question-with-T5-Text-To-Text-Transfer-Transformer-
Next ref: https://github.com/ceshine/finetuning-t5/tree/master/paraphrase
"""
import os
from dataclasses import dataclass, asdict
import pytorch_lightning as pl
import pytorch_lightning_spells as pls
import typer
from transformers import T5ForConditionalGeneration, T5Tokenizer

from Cord19Dataset import Cord19Dataset, DATASET_DIR, Parts
from t2t import T5BaseModel, masked_cross_entropy_loss


@dataclass
class T5Model(pl.LightningModule):
    base_t5_model: str = "t5-base"
    learning_rate: float = 1e-4
    epochs: int = 5
    fp16: bool = False
    batch_size: int = 16
    max_len: int = 64
    grad_accu: int = 1
    num_gpus: int = 1
    dataset_dir: str = "dataset_cache/"

    def __init__(self, **kwargs):
        super().__init__()

        model = T5ForConditionalGeneration.from_pretrained(self.base_t5_model)
        tokenizer = T5Tokenizer.from_pretrained(self.base_t5_model)
        self.t5_model = T5BaseModel(self, model, tokenizer)

        self.save_hyperparameters()
        self.train_dataset = Cord19Dataset(Parts.TRAIN)
        print("Train dataset: ", len(self.train_dataset))
        self.valid_dataset = Cord19Dataset(Parts.VALID)
        print("Valid dataset: ", len(self.valid_dataset))

    def __hash__(self):
        return hash((
            self.base_t5_model, self.learning_rate, self.epochs,
            self.fp16, self.batch_size, self.max_len,
            self.grad_accu, self.num_gpus, self.dataset_dir
        ))



def main(
        t5_model: str = "t5-base", lr: float = 1e-4,
        epochs: int = 5, fp16: bool = False,
        batch_size: int = 16, max_len: int = 64,
        grad_accu: int = 1, num_gpus: int = 1
):
    pl.seed_everything(int(os.environ.get("SEED", 738)))

    pl_module = T5Model(
        base_t5_model=t5_model,
        learning_rate=lr,
        epochs=epochs,
        fp16=fp16,
        batch_size=batch_size,
        max_len=max_len,
        grad_accu=grad_accu,
        num_gpus=num_gpus
    )

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath="model_checkpoints",
            monitor='val_loss',
            mode="min",
            filename='{step:06d}-{val_loss:.4f}',
            save_top_k=1,
            save_last=False
        ),
        pl.callbacks.LearningRateMonitor(logging_interval='step'),
    ]

    trainer = pl.Trainer(
        accelerator='dp' if num_gpus > 1 else None,
        precision=16 if pl_module.fp16 else 32,
        gpus=pl_module.num_gpus,
        val_check_interval=0.25,
        gradient_clip_val=10,
        max_epochs=pl_module.epochs,
        callbacks=callbacks,
        accumulate_grad_batches=pl_module.grad_accu,
        logger=[
            pl.loggers.TensorBoardLogger("tb_logs", name=""),
            pls.loggers.ScreenLogger(),
        ],
        log_every_n_steps=100
    )

    trainer.fit(pl_module)

    assert isinstance(callbacks[0], pl.callbacks.ModelCheckpoint)
    print(callbacks[0].best_model_path)
    pl_module = T5Model.load_from_checkpoint(
        callbacks[0].best_model_path
    )
    pl_module.t5_model.model.save_pretrained(f"{t5_model}_best")
    pl_module.t5_model.tokenizer.save_pretrained(f"{t5_model}_best")
    print("Best model saved")


if __name__ == "__main__":
    typer.run(main)
