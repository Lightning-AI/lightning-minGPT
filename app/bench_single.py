from urllib.request import urlopen

import torch
import torch._dynamo
from torch.utils.data import DataLoader

import lightning as L
from lightning_mingpt import data, models, bench


class GPTBench(bench.Bench):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_workers = 4
        self.batch_size = 64
        self.max_epochs = 3
        self.precision = 32
        self.model_type = "gpt-micro"
        self.num_runs = 3

    def create(self):
        torch.set_float32_matmul_precision("high")
        torch._dynamo.config.suppress_errors = True

        with urlopen("https://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt") as f:
            text = f.read()

        dataset = data.CharDataset(text, block_size=128)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)

        model = models.GPT(
            vocab_size=dataset.vocab_size,
            block_size=dataset.block_size,
            model_type=self.model_type,
        )

        return model, dataloader

    def train(self, model, dataloader):
        trainer = L.Trainer(
            max_epochs=self.max_epochs,
            gradient_clip_val=1.0,
            accelerator="cuda",
            devices=1,
            precision=self.precision,
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            logger=False,
            replace_sampler_ddp=False,
            num_sanity_val_steps=0,
            reload_dataloaders_every_n_epochs=1000,
        )

        trainer.fit(model, dataloader)
        final_loss = trainer.fit_loop.running_loss.last()
        return final_loss.item() if final_loss is not None else None

    def run(self):
        model, dataloader = self.create()

        self.run_benchmark(name="nocompile", fn=self.train, args=(model, dataloader), num_runs=self.num_runs)

        model, dataloader = self.create()
        model = torch.compile(model)

        self.run_benchmark(name="compile", fn=self.train, args=(model, dataloader), num_runs=self.num_runs)


# app = L.LightningApp(
#         GPTBench(cloud_compute=L.CloudCompute("gpu-fast"))
#         )

app = L.LightningApp(
    bench.BenchRun(
        GPTBench,
        num_nodes=1,
        cloud_compute=L.CloudCompute("gpu-fast"),
    )
)