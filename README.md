# Replay based Continual Learning (SplitMnist)

This project implements and compares continual learning strategies using a single Multi-Layer Perceptron (MLP) on the classic SplitMNIST benchmark.

It starts with a baseline "no replay" approach, demonstrationg Catastrophic Forgetting, and will progressively add memory replay techniques to mitigate it.


-----

## Project Goals
- Build an end-to-end continual learning pipeline from scratch
- Visually demonstrate catastrophic forgetting
- Implement and evaluate replay-based mitigation strategies


------

## What is SplitMNIST?
The MNIST dataset is split into 5 sequential binary classification tasks:

| Task | Digits |
|------|--------|
| 1    | 0 vs 1 |
| 2    | 2 vs 3 |
| 3    | 4 vs 5 |
| 4    | 6 vs 7 |
| 5    | 8 vs 9 |

In task-incremental learning, the model learns these tasks sequentially, with no access to prior taks data once it's trained.

-----

## Baseline: No Replay

In this setup, we train the MLP task-by-task without any memory of previous tasks. This results in catastrophic forgetting, clearly visible in the accuracy trends.

### Results:

| Task ID | Accuracy after Final Task |
|---------|---------------------------|
| Task 1  | 24%                       |
| Task 2  | 67%                       |
| Task 3  | 17%                       |
| Task 4  | 84%                       |
| Task 5  | 99%                       |


### Accuracy Plot:

![Task-wise Accuracy Plot](results/accuracy_plot.png)


## How to Run

1. Clone the repo
2. Run setup
    ```bash
    ./setup.sh

3. Run baseline
    cd baseline_no_replay
    python main.py

-----

## Author

**Aashish Pd Pandey

[![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/aashish-prashad-pandey-02388a1a7/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white)](https://github.com/aashish-pandey)

