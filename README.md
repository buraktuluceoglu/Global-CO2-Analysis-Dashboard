# 🌍 Global CO2 Emissions & Climate Analysis Dashboard

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

> **CEN445 Introduction to Data Visualization - Fall 2025** > A comprehensive, interactive dashboard exploring the history, geography, and drivers of global CO2 emissions.

---

## 📖 Project Overview

Climate change is the defining crisis of our time. This project utilizes **Advanced Data Visualization** techniques and **Machine Learning** algorithms to analyze global CO2 emissions from **1900 to 2022**. 

Built with **Streamlit**, the dashboard allows users to:
* Visualize emissions on a **3D Interactive Globe**.
* Track the flow of fossil fuels using **Sankey Diagrams**.
* Analyze historical trends with **Linear Regression Forecasts**.
* Explore multidimensional relationships using **Parallel Coordinates** and **AI Clustering**.


## 📊 Visualization Techniques

We implemented **9 distinct visualization methods**, focusing on advanced interactivity and storytelling.

### 1. Global & Structural Analysis
* **3D Orthographic Map:** Displays global emission distribution on a rotating globe. *Includes K-Means Clustering to group similar countries.*
* **Sankey Diagram:** Visualizes the flow of CO2 from specific fuel sources (Coal, Oil, Gas, Cement) to different continents.
* **Sunburst / Treemap:** Shows the hierarchical breakdown of emissions (Continent > Country).

### 2. Time Series & Trend Analysis
* **Animated Bubble Chart:** A motion chart showing the evolution of GDP vs. CO2 over the last century (Gapminder style).
* **Predictive Line Chart:** Visualizes historical trends and provides a **Linear Regression forecast** for emissions up to 2032.
* **Interactive Heatmap:** A color-coded matrix showing emission intensity (per capita) changes across countries and years.

### 3. Relational & Multidimensional Analysis
* **Parallel Coordinates Plot:** Allows comparison of countries across multiple dimensions (GDP, Population, Energy, CO2).
* **Feature Importance (AI):** Uses a **Random Forest Regressor** to calculate and visualize which factors drive CO2 emissions the most.
* **3D Scatter Plot:** Plots countries in a 3D space (GDP-Pop-CO2) colored by AI-determined clusters.
* **Box Plot:** Demonstrates the statistical distribution and inequality of emissions across continents.

---

## 🤖 Machine Learning Integration

To go beyond simple descriptive analytics, we integrated `scikit-learn` to provide predictive and clustering insights:

1.  **K-Means Clustering:** Used in the *3D Map* and *3D Scatter Plot* to automatically group countries into clusters based on their economic and environmental similarities, filling missing data with median imputation.
2.  **Linear Regression:** Used in the *Trend Analysis* to forecast future emission trajectories for selected countries based on historical data (1900-2022).
3.  **Random Forest Regressor:** Used in the *Multidimensional Tab* to calculate "Feature Importance," identifying which structural variables (e.g., GDP, Energy Use) are the strongest predictors of high emissions.

---

## 🚀 Installation & How to Run

Follow these steps to run the dashboard on your local machine.

### Prerequisites
* Python 3.8 or higher
* Git

### Steps

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/buraktuluceoglu/Global-CO2-Analysis-Dashboard](https://github.com/buraktuluceoglu/Global-CO2-Analysis-Dashboard)
    cd global-co2-analysis-dashboard
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application**
    ```bash
    python3 -m streamlit run app.py
    ```

4.  **View in Browser**
    The app will open automatically at `http://localhost:8501`.

---

## 📂 Data Source & Processing

* **Dataset:** [Our World in Data (OWID) - CO2 and Greenhouse Gas Emissions](https://github.com/owid/co2-data).
* **File:** `owid-co2-data.csv`
* **Preprocessing Steps:**
    * **Missing Values:** Missing GDP and Population data were handled using forward-fill or median imputation to ensure chart continuity.
    * **Continent Mapping:** Manual correction dictionary applied to fix missing or "Other" continent labels for 50+ countries (e.g., merging North/South America into "Americas" for consistency).
    * **Log Transformation:** Applied to GDP and Population data for Clustering and 3D plotting to handle extreme outliers.
