import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def fill_na_with_random(df):
    df_filled = df.copy()
    for col in df_filled.select_dtypes(include=[np.number]).columns:
        if df_filled[col].isna().any():
            col_min = df_filled[col].min()
            col_max = df_filled[col].max()
            random_values = np.random.uniform(col_min, col_max, df_filled[col].isna().sum())
            df_filled.loc[df_filled[col].isna(), col] = random_values
    return df_filled

df = pd.read_excel("D:/LPU/Sem 4/INT 375 - DS Toolbox/Project/Uncleaned filled NA.xlsx", sheet_name="Growth of Indian Shipping")
df.info()
#df = fill_na_with_random(df)
X = df[['Year']]
y = df['Total-No. of vessels']

# Train linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict on training data
y_pred = model.predict(X)

# Forecast for next 10 years
future_years = pd.DataFrame({'Year': np.arange(X['Year'].max() + 1, X['Year'].max() + 11)})
future_pred = model.predict(future_years)

# Metrics
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(X, y, label='Actual', marker='o')
plt.plot(X, y_pred, label='Model Prediction', linestyle='--')
plt.plot(future_years, future_pred, label='Forecast (Next 10 Years)', marker='x', color='green')
plt.xlabel('Year')
plt.ylabel('Total Number of Vessels')
plt.title('Growth of Indian Fleet: Forecasting with Linear Regression')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.4f}")
print("/nForecast for Next 10 Years:")
forecast_df = future_years.copy()
forecast_df['Predicted Vessels'] = future_pred.astype(int)
print(forecast_df)

#--------------------------------------------------------------------------------
#Line plot
df = pd.read_excel("D:/LPU/Sem 4/INT 375 - DS Toolbox/Project/Uncleaned filled NA.xlsx", sheet_name="Major and Non-Major Ports")
#df = fill_na_with_random(df)

selected_ports = ['Chennai Port', 'Mumbai Port', 'Vishakhapatnam Port', 'Cochin Ports']
df_selected = df[df['port_name'].isin(selected_ports)]

# Plotting
plt.figure(figsize=(12, 6))
for port in selected_ports:
    port_data = df_selected[df_selected['port_name'] == port]
    plt.plot(port_data['year/ports'], port_data['total_traffic'], marker='o', label=port)

plt.title('Total Traffic Over the Years – Selected Ports')
plt.xlabel('Year')
plt.ylabel('Total Traffic (MMT)')
plt.xticks(rotation=90)
plt.legend(title='Port')
plt.grid(True)
plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------
df_output = pd.read_excel("D:/LPU/Sem 4/INT 375 - DS Toolbox/Project/Uncleaned filled NA.xlsx",
    sheet_name="Average Output per Ship Berth "
)
latest_year = '2020-21'
df_latest = df_output[df_output['Year'] == latest_year]

ports = [
    'SMP Kolkata D.S', 'SMP Haldia D.C', 'Paradip', 'Visakhapatnam', 
    'Kamarajar', 'Chennai', 'V.O. Chidambaranar', 'Cochin', 
    'New Mangalore', 'Mormugao', 'J.L.Nehru', 'Mumbai', 'Deendayal', 'All Ports'
]
y_values = df_latest[ports].iloc[0]

plt.figure(figsize=(10,6))
plt.bar(ports, y_values,color='#003366')
plt.title(f'Avg Output per Ship Berth Day - {latest_year}')
plt.xticks(rotation=45)
plt.ylabel('Output (Tonnes)')
plt.tight_layout()
plt.show()


#-----------------------------------------------------------------------------
df_util = pd.read_excel(
    "D:/LPU/Sem 4/INT 375 - DS Toolbox/Project/Uncleaned filled NA.xlsx", 
    sheet_name="Capacity Utilization"
)

df_util = df_util[df_util['Port'] != 'All Ports']
utilization = df_util['2020-21 - Utilization (%)']

# Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(utilization, bins=10, color='#003366', edgecolor='black')
plt.title('Capacity Utilization Histogram (2020–21)')
plt.xlabel('Utilization (%)')
plt.ylabel('Number of Ports')
plt.grid(True)
#plt.tight_layout()
plt.show()

#---------------------------------------------------------------------------------
df_scatter = pd.read_excel("D:/LPU/Sem 4/INT 375 - DS Toolbox/Project/Uncleaned filled NA.xlsx", sheet_name="Capacity Utilization")
df_2021 = df_scatter[['Port', '2020-21- Traffic', '2020-21 - Capacity']]
df_2021 = df_2021.rename(columns={
    '2020-21- Traffic': 'Traffic',
    '2020-21 - Capacity': 'Capacity'
})
# Plot
plt.figure(figsize=(8, 6))
plt.scatter(df_2021['Capacity'], df_2021['Traffic'], alpha=0.7)
plt.title('Traffic vs Capacity (2020-21)')
plt.xlabel('Capacity (Million Tonnes)')
plt.ylabel('Traffic (Million Tonnes)')
plt.grid(True)
plt.show()

#-----------------------------------------------------------------------

df_corr = pd.read_excel("D:/LPU/Sem 4/INT 375 - DS Toolbox/Project/Uncleaned filled NA.xlsx", sheet_name="Growth of Indian Shipping")
corr_matrix = df_corr[['Coastal-GRT', 'Overseas-GRT', 'Total-GRT']].corr()

plt.figure(figsize=(6,4))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
#-------------------------------------------------------------
df = pd.read_excel("D:/LPU/Sem 4/INT 217 - Data Management/Project/Ports/Uncleaned filled NA.xlsx", sheet_name="Growth of Indian Shipping")

# Fill NaNs with random values
#df = fill_na_with_random(df)

# Filter only up to 2021
df_filtered = df[df['Year'] <= 2021]

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(df_filtered['Year'], df_filtered['Coastal-GRT'], marker='o', label='Coastal GRT')
plt.plot(df_filtered['Year'], df_filtered['Overseas-GRT'], marker='s', label='Overseas GRT')
plt.plot(df_filtered['Year'], df_filtered['Total-GRT'], marker='^', label='Total GRT')

plt.title('GRT Trends: Coastal vs Overseas vs Total (Till 2021)')
plt.xlabel('Year')
plt.ylabel('GRT')
plt.xticks(rotation=90)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
#----------------------------------------------------------------------------
df = pd.read_excel("D:/LPU/Sem 4/INT 217 - Data Management/Project/Ports/Uncleaned filled NA.xlsx", sheet_name="Growth of Indian Shipping")
df_filtered = df[df['Year'] <= 2021]
heatmap_data = df_filtered[['Year', 'Coastal-GRT', 'Overseas-GRT', 'Total-GRT']]
heatmap_data.set_index('Year', inplace=True)
plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data.T, annot=True, cmap='coolwarm', cbar=True, fmt='.1f', linewidths=0.5)

plt.title('Heatmap: GRT Trends - Coastal vs Overseas vs Total (Till 2021)')
plt.ylabel('GRT Type')
plt.xlabel('Year')
plt.show()

#---------------------------------------------------------------------------------------

df_pair = pd.read_excel("D:/LPU/Sem 4/INT 375 - DS Toolbox/Project/Uncleaned filled NA.xlsx", sheet_name="Growth of Indian Shipping")
df_pair_filtered = df_pair[['Year', 'Coastal-No. of vessels', 'Coastal-GRT', 'Coastal-Average GRT', 
                            'Overseas-No. of vessels', 'Overseas-GRT', 'Overseas-Average GRT', 
                            'Total-No. of vessels', 'Total-GRT', 'Total-Average GRT']]

# Plotting the pair plot
sns.pairplot(df_pair_filtered)
plt.suptitle('Pair Plot – Indian Shipping Growth', y=1.02)
plt.show()
#-------------------------------------------------------------------------------

df_ports = pd.read_excel("D:/LPU/Sem 4/INT 375 - DS Toolbox/Project/Uncleaned filled NA.xlsx", sheet_name="Major and Non-Major Ports")
sns.countplot(data=df_ports, x='type_of_port',color='#003366')
plt.title('Count of Major vs Non-Major Ports')
plt.ylabel('Number of Ports')
plt.show()

#--------------------------------------------------------------------------------
df_paradip = pd.read_excel("D:/LPU/Sem 4/INT 375 - DS Toolbox/Project/Uncleaned filled NA.xlsx", sheet_name="Turnaround Time (TRT)")
df_paradip = df_paradip[['Year', 'Paradip']]  # Only keep relevant columns

# Plotting the line plot
plt.figure(figsize=(10,5))
sns.lineplot(x='Year', y='Paradip', data=df_paradip, marker='o', label='Paradip Port')
plt.title('Traffic Trend – Paradip Port')
plt.xlabel('Year')
plt.ylabel('Traffic (MMT)')
plt.xticks(rotation=90)
plt.grid(True)
plt.show()

#-------------------------------------------------------------------------------
df_age = pd.read_excel("D:/LPU/Sem 4/INT 375 - DS Toolbox/Project/Uncleaned filled NA.xlsx", sheet_name="Indian Tonnage Age-wise")
latest_year = df_age['Year'].max()
df_latest_age = df_age[df_age['Year'] == latest_year]

# Prepare data for the pie chart
labels = ['Up to 5 Yrs.', '6-15 Yrs.', '16-20 Yrs.', 'Over 20 Yrs.']
sizes = [
    df_latest_age['Proportion of Indian Tonnage-Upto 5 Yrs.'].values[0],
    df_latest_age['Proportion of Indian Tonnage-6-15 Yrs.'].values[0],
    df_latest_age['Proportion of Indian Tonnage-16-20 Yrs.'].values[0],
    df_latest_age['Proportion of Indian Tonnage-Over 20 Yrs.'].values[0]
]
custom_colors = ['#FF6F61','#008080','#66B2FF','#FF9933']
plt.figure(figsize=(7, 7))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=custom_colors, startangle=140)
plt.title(f'Tonnage Distribution by Age Group ({latest_year})')
plt.axis('equal')
plt.show()