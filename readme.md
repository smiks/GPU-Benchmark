## Requirements

Anaconda \
https://www.anaconda.com/download/success

If you have NVIDIA, then you have to install cudatoolkit: 

```conda install cudatoolkit```

You also have to install torch for anaconda and dependencies required to run on NVIDIA

```conda install pytorch torchv ision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia```

---
Code also runs CPU only ( in this case you only need <i>pytorch</i> )