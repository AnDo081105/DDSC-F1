# DDSC-F1

**Project:** Race time prediction for Grand Prix drivers.

---

**Purpose:** Predict per-driver race time using historical event, practice and qualifying features to identify likely race winners.

---

**Requirements**

- Python 3.8–3.11
- Packages (see `requirements.txt`): pandas, numpy, scikit-learn, xgboost, joblib, streamlit, plotly, matplotlib, fastf1

---

**Install**

```bash 
pip install -r requirements.txt
```

---

**Prepare Data**

- Collect data through `fastf1` using the data script

python [data.py](http://_vscodecontentref_/2)

This writes `dataset/train_data.csv` and `dataset/test_data_2025.csv`. Ensure Driver and Race_Time are present where required.

---

**Run the app**

`streamlit run [app.py](http://_vscodecontentref_/3)`

---

**Project structure**

Files

`data.py` — data collection (uses fastf1 and a cache/ folder)
`model.ipynb` — training notebook
`app.py` — Streamlit UI
`dataset/` — input CSVs (`train_data.csv`, `test_data_2025.csv`)
`model_data/` — saved model & preprocessor artifacts
