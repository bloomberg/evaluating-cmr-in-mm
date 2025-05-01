# Benchmark Granularity and Model Robustness for Image-Text Retrieval: A Reproducibility Study

This repository contains the official code for the SIGIR 2025 paper:

**Benchmark Granularity and Model Robustness for Image-Text Retrieval: A Reproducibility Study**

🎓 *Accepted at the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2025)*

We provide code, configuration files, and resources to reproduce all experiments, results, and analyses from the paper.

---

## 📁 Project Structure

------------
    ├── README.md          <- The top-level README for developers using this project.
    |
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes directory a Python module
    │   │
    │   ├── evaluation.py  <- Entry-point for the CLI
    │   │
    │   ├── models         <- Package to load models to make
    │   │                     predictions
    │   └── ...            <- ...
    │
    ├── config             <- Config files to associate models and benchmarks to evaluate.
    |
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`

## Setting Dev Environment

1. Create a clean virtual environment and install the requirements for this project

    ```
    make create-venv
    ```

2. Activate virtual environment

    ```
    source .venv/bin/activate
    ```

3. Install required Python packages:

    ```
    pip install -r requirements.txt
    ```

## Running the experiments

### Model evaluation example

```
python3 src/evaluation.py --dataset f30k --model clip --task t2i --perturbation none
```

### Example for results printer

```
python3 src/evaluation.py --dataset f30k --model clip --task t2i --perturbation none
```

Example for jaccacd similarity computation

### Results and Robustness Analysis

All experimental outputs, including accuracy metrics, perturbation robustness comparisons, and plots, can be generated using the scripts in:

- src/
- notebooks/

You can also use:

```
python3 src/evaluation.py --dataset coco --model blip --task i2t --perturbation jaccard
```

To run model evaluation under specific robustness settings.

## Citation

If you use this codebase or find our study useful in your research, please cite our paper:

```
@inproceedings{hendriksen2025granularity,
  title     = {Benchmark Granularity and Model Robustness for Image-Text Retrieval: A Reproducibility Study},
  author    = {Mariya Hendriksen, Shuo Zhang, Ridho Reinanda, Mohamed Yahya, Edgar Meij and Maarten de Rijke},
  booktitle = {Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year      = {2025}
}
```

## Contact

If you have any questions or feedback, please reach out via [mariya.hendriksen@gmail.com] or [szhang611@gmail.com].
