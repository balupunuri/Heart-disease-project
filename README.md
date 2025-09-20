# Heart Disease Prediction - Machine Learning Project

**What this is**
A simple machine-learning project (Python) to predict heart disease using a tabular dataset.
This bundle contains:
- `data/heart.csv` — a synthetic sample dataset (for quick testing).
- `heart_disease.ipynb` — Jupyter notebook with example EDA and training steps.
- `heart_disease.py` — standalone script that trains multiple models and saves the best one.
- `requirements.txt` — Python libraries used.
- `README.md` — this file.

**How to use**
1. (Optional) Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate   # macOS / Linux
   venv\Scripts\activate    # Windows
   ```
2. Install requirements:
   ```
   pip install -r requirements.txt
   ```
3. Run the Jupyter notebook:
   ```
   jupyter notebook heart_disease.ipynb
   ```
   or run the script to train and save best model:
   ```
   python heart_disease.py --data ./data/heart.csv --out best_model.joblib
   ```

**Notes**
- The included `data/heart.csv` is synthetic (generated for demonstration). For real experiments, replace it with the UCI / Kaggle Heart Disease dataset and keep the same column names (particularly the `target` column where 1 indicates disease).
- The script uses GridSearchCV with simple parameter grids. Feel free to expand or try other models.
- If you want me to include the real UCI dataset inside the ZIP (if you want me to fetch it), tell me and I can fetch and add it — but I did not include external downloads in this package by default.

**License / Attribution**
Use this project for learning and non-commercial experiments.
