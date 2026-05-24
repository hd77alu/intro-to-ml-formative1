# Intro to ML — Formative 1: Classical ML and NN

This Project implemented and compared two perspectives on classification:

- Two Classical machine learning models (Logistic regression and Random forest).
- A basic neural network implemented from scratch in NumPy.

## Dataset

- Title: Climate Model Simulation Crashes
- Source: [UCI Machine Learning Repository](https://archive-beta.ics.uci.edu/dataset/252/climate+model+simulation+crashes)

- Characteristics: ~540 simulations, 18 continuous input parameters (Latin hypercube sampling), binary outcome (success vs crash) with a strong class imbalance (majority successes).

## Project Structure

- `Assignment1_NN.ipynb` — main Jupyter Notebook containing data preprocessing, classical models (Logistic Regression, Random Forest), a scratch NumPy neural network, experiments, and evaluation figures.
- `pop_failures.dat` — raw dataset files used by the notebook (downloaded from UCI). 
- `README.md` — this file with dataset and setup information.

## Setup

1. **Clone or Download the Repository**
```bash
git clone https://github.com/hd77alu/intro-to-ml-formative1
cd intro-to-ml-formative1
```

2. **Create and activate a virtual environment (recommended):**

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

3. **Install required packages:**

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

4. **Launch the notebook:**

```bash
jupyter notebook Assignment1_NN.ipynb
```

If you prefer, open the notebook in Google Colab using the badge at the top of `Assignment1_NN.ipynb`.

## Implementation Notes

- Classical models: Logistic Regression and Random Forest are implemented using `scikit-learn`. Multiple regularization and structural experiments are run (L1, L2, ElasticNet for LR; constrained depth, min leaf size, restricted features for RF) and evaluated on accuracy, precision, recall, and F1-score.
- Neural network: A three-layer feed-forward network is implemented from scratch in NumPy (vectorized forward pass, numerically stable binary cross-entropy loss, hand-coded backpropagation, and batch gradient descent). Learning-curve, ROC/AUC, and precision–recall diagnostics are included.
- Experiments: The notebook documents deliberate hyperparameter choices and presents a comparative table and confusion matrices to interpret trade-offs between precision and recall for this imbalanced problem.
