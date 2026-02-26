# TD-and-PPO-in-RKHS

This project implements the algorithms from the paper:

Lu Zou, Wendi Ren, Weizhong Zhang, Liang Ding, Shuang Li. Sampling Complexity of TD and PPO in RKHS. ICLR 2026.


## 📋 Requirements

- Python 3.7+
- PyTorch
- NumPy
- Matplotlib
- Gym (OpenAI Gym)

## 🚀 Installation & Usage

### 1. Install Dependencies

```bash
pip install torch numpy matplotlib gym
```

### 2. Run Experiments

```bash
python main.py
```

## ⚙️ Configuration

You can modify the following hyperparameters in the `main()` function of `main.py`:

- `env_name`: Environment name (default: `'CartPole-v1'`)
- `learning_rate`: Learning rate (default: 1e-3)
- `T`: Number of training epochs per update (default: 4)
- `batchsize`: Batch size (default: 32)
- `schedule_pows`: List of power exponents for β scheduling strategies (default: `[-0.2, -0.5, -1.5]`)
- `seeds`: List of random seeds (default: `[1,2,4,5,6,7,8,9,777,999]`)

## 📊 Output

Experimental results are automatically saved to the `results/` directory, including:

- **Parameter file**: `parameters.json` - Records all experimental hyperparameters
- **Raw returns plot**: Shows raw return curves for each episode
- **Smoothed returns plots**: Moving average smoothed return curves with different window sizes (20, 40, 60, 80, 100)
- All plots are saved in both PNG and PDF formats

Directory structure:
```
results/
└── {env_name}/
    └── {timestamp}/
        ├── parameters.json
        ├── {env_name}_beta_schedule_raw_returns_*.png/pdf
        └── {env_name}_beta_schedule_smoothed_returns_w{window}_*.png/pdf
```

## 📁 Project Structure

```
TD-and-PPO-in-RKHS/
├── main_discrete.py      # Main program file
├── README.md             # Project documentation
└── results/              # Experimental results directory
    └── {env_name}/
        └── {timestamp}/
            └── ...
```

## Reference
- https://github.com/wisnunugroho21/reinforcement_learning_truly_ppo/tree/master

## Citation

If you find this code useful in your research, please cite:

```bibtex
@inproceedings{zou2026sampling,
  title     = {Sampling Complexity of TD and PPO in RKHS},
  author    = {Zou, Lu and Ren, Wendi and Zhang, Weizhong and Ding, Liang and Li, Shuang},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```
