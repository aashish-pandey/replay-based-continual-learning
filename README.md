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
| Task 1  | 24.3%                     |
| Task 2  | 67.3%                     |
| Task 3  | 16.9%                     |
| Task 4  | 84.3%                     |
| Task 5  | 99.2%                     |


### Accuracy Plot:

![Task-wise Accuracy Plot](/baseline_no_replay/results/accuracy_plot.png)


## Random Replay: Continual Learning on SplitMNIST
This experiment uses a Replay Buffer to mitigate catastrophic forgetting in a continual leaning setup with 5 SplitMNIST tasks.

-----

## What's Random Replay?
As the model learns each new task, it saves a few examples into a replay buffer. during training on new tasks, it replays previous samples to retain prior knowledge.

-----

## Results: Baseline VS Replay
| Task | No Replay | Random Replay |
|------|-----------|---------------|
| 1    | 24.3%     | 26.0%         |
| 2    | 67.3%     | 66.7%         |
| 3    | 16.9%     | 19.1%         |
| 4    | 84.3%     | 94.6%         |
| 5    | 99.2%     | 99.5%         |

## Accuracy Over Time
![Replay Accuracy](/random_replay/results/accuracy_plot.png)

## Episodic Replay - Continual Learning on SplitMNIST

This experiment implements class-balanced episodic memory as a continual learning strategy using a Multi-Layer Perceptron (MLP) on the SplitMNIST benchmark. 
This method improves ove random replay by allocating a fixed number of memory slot per class, ensuring more balanced and informative memory usage.

## Results: BAseline vs Random Vs Episodic Replay
| Task | No Replay | Random Replay | Episodic Replay |
|------|-----------|---------------|-----------------|
| 1    | 24.3%     | 26.0%         | 31.5%           |
| 2    | 67.3%     | 66.7%         | 67.8%           |
| 3    | 16.9%     | 19.1%         | 19.1%           |
| 4    | 84.3%     | 94.6%         | 94.6%           |
| 5    | 99.2%     | 99.5%         | 99.5            |

## Accuracy Over Time
![Episodic Accuracy](/episodic_replay/results/accuracy_plot.png)

## How to Run

1. Clone the repo
2. Run setup
    ```bash
    ./setup.sh

3. Run baseline
    ```bash
    cd baseline_no_replay
    python main.py

4. Run Random replay
    ```bash
    cd random_replay
    python main.py
-----

## Author

**Aashish Pd Pandey

[![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/aashish-prashad-pandey-02388a1a7/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white)](https://github.com/aashish-pandey)

