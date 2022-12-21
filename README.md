# lightning-minGPT

lightning-minGPT is an implementation of Andrej Karpathy's minGPT in Lightning.

It is aimed at providing a minimal Lightning layer on top of minGPT, while leveraging the full breadth of Lightning.

There are currently a few options:

- GPT: the GPT model from minGPT vanilla
- DeepSpeedGPT: the GPT model from minGPT made DeepSpeed-ready (set `--strategy=deepspeed`)
- FSDPGPT: the GPT model from minGPT made FSDP (native)-ready (set `--strategy=fsdp_native`)

minGPT is vendored with the repo in the `mingpt` directory. Find the LICENSE for minGPT there.

Thanks to:

- @karpathy for the original minGPT implementation
- @williamFalcon for the first Lightning port
- @SeanNaren for the DeepSpeed pieces

## MinGPT

First install the dependencies

```shell
pip install -r requirements.txt
```

then

```shell
python train.py
```

See

```shell
python train.py --help
```

for the available flags.

## MinGPT DeepSpeed

Install the extra-dependencies:

```shell
pip install -r requirements/deepspeed.txt
```

and pass the `strategy` flag to the script

```shell
python train.py --strategy deepspeed
```

## MinGPT FSDP native

Pass the `strategy` flag to the script

```shell
python train.py --strategy fsdp_native
```

## PyTorch 2.0

To run on dynamo/inductor from the PyTorch 2.0 compiler stack, run

```shell
python train.py --compile dynamo
```

Note that you will need a recent `torch` nightly (1.14.x) for `torch.compile`
to be available.

## Credits

- https://github.com/karpathy/minGPT
- https://github.com/SeanNaren/minGPT
- https://pytorch-lightning.readthedocs.io/en/stable/advanced/model_parallel.html

## License

Apache 2.0 license https://opensource.org/licenses/Apache-2.0
