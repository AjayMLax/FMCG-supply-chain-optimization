import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cvxpy as cp
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


## Import data
df = pd.read_csv(".spyder-py3/OR projects/Supply Chain optimization/Data/FMCG_data.csv")
print(df.head())
print(df.info())

# Histogram of the current product weight to see trends
sns.set_theme(style="whitegrid", palette="pastel")
sns.histplot(df['product_wg_ton'], kde=True)
plt.title("Distribution of product_wg_ton")
plt.xlabel("Product weight (tons)")
plt.ylabel("Frequency")
plt.show()

# Correlation heatmap to spot relationships for prediction model
plt.figure(figsize=(14,10))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".1f")
plt.title("Correlation Heatmap")
plt.show()


## Build a Linear Regression model for prodcut_wg_ton

# Check how many siginificant rows between correlated columns of prouct_wg_ton
df_model = df[['product_wg_ton', 'storage_issue_reported_l3m', 'wh_est_year', 'wh_breakdown_l3m', 'transport_issue_l1y']]
df_model = df_model.dropna()
#print("Remaining rows:", len(df_model))


# Split the data into training and features
x = df_model.drop(columns=['product_wg_ton'])
y = df_model['product_wg_ton']

# Split data into training and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 42)
#print(f"Training set size: {len(x_train)} , {len(y_train)}")
#print(f"Test set size: {len(x_test)} , {len(y_test)}")
#print(x_train.head())
#print(x_train.iloc[0:5,0:3])

# Create Linear Regression model
model = LinearRegression()
model.fit(x_train, y_train)
print("\nModel:")
print(f"Intercept : {model.intercept_}")
print(f"Coefficients : {model.coef_}")

# Predict using model and look at errors
y_pred = model.predict(x_test)
print("\nFirst 5 values of Prediction vs Actual product weight:")
print(f"Predictions : {y_pred[:5]}")
print(f"Actual : {y_test[:5].values}")

# Scatter plot of regression fit
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.4, color='tab:blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect prediction')
plt.xlabel("Actual Product weight")
plt.ylabel("Predicted Product weight")
plt.title("Actual vs. Predicted Product Weight")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Scatter plot of residual from regression
residuals = y_test - y_pred

plt.figure(figsize=(6,4))
plt.scatter(y_pred, residuals, alpha=0.4, color='tab:orange')
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel("Predicted product_wg_ton")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.grid(True)
plt.tight_layout()
plt.show()

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nErrors of Prediction vs Test values:")
print(f"Mean Absolute Error : {mae:.2f}")
print(f"Mean Squared Error : {mse:.2f}")
print(f"R² Score : {r2:.4f}")


## Begin Optimization

# Create data frame with specific variables
df_vars = df[['product_wg_ton', 'storage_issue_reported_l3m', 'wh_est_year', 'wh_breakdown_l3m', 'transport_issue_l1y', 'dist_from_hub', 'workers_num']]
df_vars = df_vars.dropna()
#print(f"\n{df_vars.shape}\n")

# Define variables
n = (df_vars.shape)[0]
x = cp.Variable(n)
x_opt = df_vars[['storage_issue_reported_l3m', 'wh_est_year', 'wh_breakdown_l3m', 'transport_issue_l1y']]
demand = model.predict(x_opt)
dist = (df_vars['dist_from_hub']).values
workers = (df_vars['workers_num']).values

# Define the problem and plot the solution for different parameters
alphas = [0, 0.01, 0.1]         # Penalization parameter for large distance
worker_ratios = [100, 500]      # Max product weight that can be handled by a worker

colors = ['tab:blue', 'tab:orange', 'tab:green']

for w_index, w_ratio in enumerate(worker_ratios):
    plt.figure(figsize=(8,6))
    
    for i, alpha in enumerate(alphas):
        x = cp.Variable(n)
        obj = cp.Minimize(cp.sum_squares(x - demand) + (alpha * cp.sum(cp.multiply(dist,x))))
        constraints = [x >= 0, x <= (w_ratio*workers)]
        problem = cp.Problem(obj,constraints)
        problem.solve()
        
        # Plot to compare between optimized vs predicted
        plt.scatter(demand, x.value, color=colors[i], alpha=0.3, s=8, label=f"α = {alpha}")
    
    plt.plot([demand.min(), demand.max()], [demand.min(), demand.max()], 'k--', label='Perfect match')
    plt.xlabel('Predicted Supply (tons)')
    plt.ylabel('Optimized Supply (tons)')
    plt.title(f'Predicted vs. Optimized Supply (worker_ratio = {w_ratio})')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Create csv file with final output
df_output = df.loc[df_vars.index].copy()

df_output['predicted_demand'] = demand
df_output['optimized_supply'] = x.value

df_output.to_csv(".spyder-py3/OR projects/Supply Chain optimization/Results/optimized_supply_output.csv", index=False)
