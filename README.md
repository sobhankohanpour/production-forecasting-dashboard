# 🛢️ Well Production Forecasting Dashboard
![License](https://img.shields.io/github/license/sobhankohanpour/production-forecasting-dashboard)
![Last Commit](https://img.shields.io/github/last-commit/sobhankohanpour/production-forecasting-dashboard)
![Issues](https://img.shields.io/github/issues/sobhankohanpour/production-forecasting-dashboard)
![Pull Requests](https://img.shields.io/github/issues-pr/sobhankohanpour/production-forecasting-dashboard)

![Repo Size](https://img.shields.io/github/repo-size/sobhankohanpour/production-forecasting-dashboard)
![Code Size](https://img.shields.io/github/languages/code-size/sobhankohanpour/production-forecasting-dashboard)
![Contributors](https://img.shields.io/github/contributors/sobhankohanpour/production-forecasting-dashboard)
![Forks](https://img.shields.io/github/forks/sobhankohanpour/production-forecasting-dashboard)
![GitHub Stars](https://img.shields.io/github/stars/sobhankohanpour/production-forecasting-dashboard)

An interactive Streamlit application for petroleum engineers to visualize, explore, and forecast well production using advanced machine learning and customizable datasets.


## 🚀 Overview

The **Well Production Forecasting Dashboard** is a user-friendly platform designed to streamline data exploration, visualization, and production forecasting for oil & gas wells.
It allows users to:

* Upload their own datasets or choose from real-world North Dakota oil & gas datasets
* Explore rich visualizations (distribution, relational, categorical plots)
* Perform basic data engineering before modeling
* Train machine learning models *(coming soon)*
* Generate production predictions *(coming soon)*

This app is ideal for petroleum engineers, data scientists, and researchers working on reservoir analysis, production insights, and forecasting workflows.


## 📁 Project Structure

```
production-forecasting-dashboard/
│
├── app/
│   └── main.py                     # Main Streamlit app
│
├── data/                           # Included real-world datasets
│   ├── ND_cumulative_formation_2020.xlsx
│   ├── ND_gas_1990_to_present.xlsx
│   ├── ND_historical_barrels_of_oil_produced_by_county.xlsx
│   └── ND_historical_MCF_gas_produced_by_county.xlsx
│
├── src/
│   └── plots.py                    # All plot functions (sns + matplotlib)
│   └── model.py                    # Machine learning: CART training, evaluation, saving
│
├── strings/
│   └── strings.py                  # String constants for UI texts, descriptions, and messages
│
├── .gitignore
├── README.md
└── requirements.txt                # Python dependencies
```

---

## 🧠 Features

### 🔹 1. Dataset Handling

* Upload **your own `.xlsx` files**
* Or select from **four real-world North Dakota datasets**
* Automatic description + preview
* Cleans and prepares the dataset for plotting or modeling

### 🔹 2. Exploratory Data Analysis

The dashboard includes **15+ interactive plot types**, grouped into:

#### 📊 Distribution Plots

* Distribution plot
* Histogram with adjustable bins
* KDE
* ECDF
* Rug plot

#### 🧩 Categorical Plots

* Catplot
* Strip plot
* Swarm plot
* Box plot
* Violin plot
* Point plot
* Bar plot

#### 🔗 Relational Plots

* Scatter plot
* Line plot

Each plot uses clean, readable Seaborn + Matplotlib visuals optimized for Streamlit.


## 🤖 Machine Learning (CART Decision Tree)

This app supports CART decision tree models for classification or regression:

- Train on your dataset using any numeric or properly formatted date/time columns
- Handles non-numeric features automatically (label encoding or timestamp conversion)
- Evaluate using:
   - **Accuracy** (for classification)
   - **MSE** and **R²** (for regression)
- Save models locally for future predictions


## 🔮 Prediction (Coming Soon)

* Predict well production using trained models
* Custom input forms
* Download predictions


## ▶️ How to Run the App

### **1️⃣ Clone the repository**

```bash
git clone https://github.com/sobhankohanpour/production-forecasting-dashboard
cd production-forecasting-dashboard
```

### **2️⃣ Install dependencies**

Create a virtual environment and install required packages:

```bash
pip install -r requirements.txt
```

### **3️⃣ Launch the Streamlit app**

```bash
streamlit run app/main.py
```


## 📦 Dependencies

Core libraries used in this project:

* `streamlit`
* `pandas`
* `matplotlib`
* `seaborn`
* `scikit-learn`
* `joblib`



## 📘 Included Datasets

The `data/` folder contains curated production datasets from North Dakota:

* **Cumulative Oil Production by Formation (2020)**
* **Gas Production (1990–Present)**
* **Historical Monthly Oil Production by County**
* **Historical Monthly Gas Production by County**

All are ready for direct loading in the dashboard.


## 🧩 Plot Functions (from `src/plots.py`)

The app provides reusable plot functions such as:

* `scatterplot()`
* `lineplot()`
* `distplot()`, `histplot()`, `kdeplot()`
* `ecdfplot()`, `rugplot()`
* `catplot()`, `stripplot()`, `swarmplot()`
* `boxplot()`, `violinplot()`
* `pointplot()`, `barplot()`

You can easily extend or modify these functions for more visualizations.


## 📄 License

MIT License


## 🤝 Contributing

Contributions, improvements, and feature requests are welcome!
Feel free to open an issue or submit a pull request.


## ⭐ Support

If you find this project useful, consider giving the repository a **star** ⭐ on GitHub — it helps others discover it!
