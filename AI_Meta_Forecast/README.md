# 🌞 AI_Meta_Forecast: Meta-Forecasting for Solar Power Generation

This repository contains the implementation of **Dynamic ElasticNet (DELNET)** and **Dynamic Particle Swarm Optimization (DPSO)**-based meta-forecasting approaches for solar power generation.  

## 🧠 Project Overview

Meta-forecasting is a higher-level forecasting approach where outputs from multiple models are combined using a meta-model (or optimizer).  
In this project, two methods are compared:

- **DELNET (Dynamic ElasticNet):** A regression-based approach that dynamically learns the best combination of forecasts using ElasticNetCV.
- **DPSO (Dynamic Particle Swarm Optimization):** A swarm intelligence approach that optimizes forecast weights through Particle Swarm Optimization (PSO).

The code supports **rolling window forecasting**, computes **RMSE** and **MSE**, and saves the results to Excel files for evaluation.

## 📁 Repository Structure
```bash
AI_Meta_Forecast/
│
├── Code/
│ └── meta_forecast.py # Main production-ready forecasting script
│
├── Data/
│ └── (empty) # User should place their data file here
│
├── Results/
│ └── (auto-created) # Forecast results are saved here
│
├── README.md
│
└── Requirments.txt  

```


##  ⚙️ How to Run

1. **Prepare your environment**

   ```python
   pip install -r requirements.txt
   ```
    (You can create a virtual environment before installing dependencies.)

2. **Place your dataset**

    - Save your Excel file in the `Data/` folder.
    - Update the file name in the script:

    ```python
    DATA_FILE = "Data/YOUR_DATA_FILE.xlsx"
    ```

3. **Run the forecasting script**

    - Execute the script using:

    ```python
    python Code/meta_forecast.py
    ```

4. **View results**
    - The results will be saved in the Results/ folder as Excel files.    


## ⚠️ Data Handling

> The dataset used for this study is protected under a Non-Disclosure Agreement (NDA).  
> Users are expected to provide their own dataset and ensure appropriate preprocessing.

Please **check and clean your data** before running the script:

- Handle missing values  
- Remove duplicates  
- Validate datetime formatting  
- Ensure numeric columns are correctly parsed


## 📄 Reference

*"Meta-Forecasting for Solar Power Generation: Algorithm-Based Swarm Intelligence."*  
In *Proceedings of the 2024 20th International Conference on the European Energy Market (EEM)*.  
IEEE, 2024.  
[DOI: 10.1109/EEM60825.2024.10608959](https://doi.org/10.1109/EEM60825.2024.10608959)
