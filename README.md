# Software Defect Prediction on NASA KC1 Dataset

## Project Overview

This project studies software defect prediction using the NASA KC1 dataset.
The aim is to understand how well machine learning and deep learning models can predict defective software modules, and how techniques like Explainable AI (XAI), feature selection, and uncertainty handling can improve model reliability and understanding.

Software defect prediction models help identify modules that are more likely to contain defects before testing. This allows developers to focus their testing and maintenance efforts on the most risky parts of the code, saving time and improving software quality.

---

## Research Questions

1. **RQ1:** How well do software defect prediction models on the KC1 dataset perform when some data is missing, and can models that handle uncertainty make the predictions more reliable?
2. **RQ2:** How can explainable AI (XAI) techniques improve the interpretability and understanding of defect prediction models on the KC1 dataset?
3. **RQ3:** How do different feature selection and representation methods impact machine learning and deep learning defect prediction models on KC1?

---

## Dataset Information

**Dataset:** NASA KC1

- **Source:** PROMISE Repository / NASA Metrics Data Program (MDP)
- **Number of instances:** 2,109 modules
- **Number of attributes:** 21 features (software metrics such as lines of code, complexity, coupling, etc.)
- **Target variable:** `defects` (1 = defective, 0 = non-defective)

**Download link:**
[NASA PROMISE KC1 Dataset (GitHub)](https://github.com/ApoorvaKrisna/NASA-promise-dataset-repository)

---

## Project Setup

### Requirements

You need Python 3.8 or higher. Install the following libraries:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn shap lime tensorflow
```
