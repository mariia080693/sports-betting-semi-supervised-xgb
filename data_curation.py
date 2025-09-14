import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({
    'figure.figsize': (10, 6),
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.edgecolor': 'black',
    'axes.linewidth': 1,
    'lines.markersize': 6,
    'lines.linewidth': 2,
    'legend.fontsize': 12,
    'font.family': 'sans-serif',
    'font.sans-serif': 'Arial',
    'axes.prop_cycle': plt.cycler(color=['green']),
    'figure.autolayout': True
})

# Data loading and exploration
df = pd.read_excel("Modeling Task_v3b.xlsx")

# Initial data checks
print(df.head(), "\n")
print("-" * 50)
print(df.info(), "\n")
print("-" * 50)
print(f"Duplicate rows: {df.duplicated().sum()}", "\n")
print("-" * 50)

# Check if any columns have mixed types
for col in df.columns:
    types_in_col = df[col].apply(type).nunique()
    if types_in_col > 1:
        print(f"Column '{col}' has mixed types: {df[col].apply(type).unique()}", "\n")
print("-" * 50)

# Initial feature engineering  
df['SurveyAnswer'] = df['SurveyAnswer'].replace('No Answer', pd.NA)
df['DaysReg'] = (df['RegistrationDate'].max() - df['RegistrationDate']).dt.days
df['DaysToFirstBet'] = (df['FirstBetDate'] - df['RegistrationDate']).dt.days

# Check if RegistrationDate is after FirstBetDate
invalid_dates = df[df['DaysToFirstBet'] < 0]
print(invalid_dates[['RegistrationDate', 'FirstBetDate', 'DaysToFirstBet']].head())
print(f"\n{len(invalid_dates)/len(df)*100:.2f}% of total data with negative DaysToFirstBet\n")
print("-" * 50)

# Numeric columns analysis + multicollinearity investigation
print(df.describe(), "\n")
print("-" * 50)
print(df[['Age', 'FirstWeekTurnover', 'DaysReg', 'DaysToFirstBet']].corr(), "\n")
print("-" * 50)

# Bar plot for DaysReg
df['DaysReg'].hist(bins=30, edgecolor='black')
plt.xlabel('Days Since Registration')
plt.ylabel('Number of Users')
plt.title('User Registrations per Day')
plt.xticks(range(df['DaysReg'].min(), df['DaysReg'].max()+1, 100), rotation=90)
plt.show()

# Bar plot for Age
df['Age'].hist(bins=30, edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Number of Users')
plt.title('User Age Distribution')
plt.xticks(range(df['Age'].min(), df['Age'].max()+1, 5))
plt.show()

# Scatter plot for Age vs FirstWeekTurnover
plt.scatter(df['Age'], df['FirstWeekTurnover'], edgecolor='black')
plt.xticks(range(18, df['Age'].max()+1, 5))
plt.xlabel("Age")
plt.ylabel("First Week Turnover")
plt.title("Age vs First Week Turnover")
plt.tight_layout()
plt.show()

# Scatter plot for DaysReg vs FirstWeekTurnover
plt.scatter(df['DaysReg'], df['FirstWeekTurnover'], label='DaysReg', edgecolor='black')
plt.xlabel('Days')
plt.ylabel('First Week Turnover')
plt.title('Days Since Registration vs First Week Turnover')
plt.legend()
plt.show()

# Categorical columns analysis
# Standardize RegistrationDevice
df['RegistrationDevice'] = np.where(
    df['RegistrationDevice'].str.contains('mobile', case=False, na=False),
    'Mobile',
    df['RegistrationDevice']
)

df['RegistrationDevice'] = np.where(
    df['RegistrationDevice'].str.contains('tablet', case=False, na=False),
    'Tablet',
    df['RegistrationDevice']
)

# Standardize FirstBetDevice
df['FirstBetDevice'] = np.where(
    df['FirstBetDevice'].str.contains('phone', case=False, na=False),
    'Phone',
    df['FirstBetDevice']
)

df['FirstBetDevice'] = np.where(
    df['FirstBetDevice'].str.contains('internet', case=False, na=False),
    'Internet',
    df['FirstBetDevice']
)

df['FirstBetDevice'] = np.where(
    df['FirstBetDevice'].str.contains('mobile', case=False, na=False),
    'Mobile',
    df['FirstBetDevice']
)

df['FirstBetDevice'] = np.where(
    df['FirstBetDevice'].str.contains('tablet', case=False, na=False),
    'Tablet',
    df['FirstBetDevice']
)

categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if col in df.columns:
        print(f"{col} distribution:")
        print(df[col].value_counts(dropna=False))
        print(f"Unique values: {df[col].nunique()}")
        print("-" * 50)

# Bar plot for MainBetSport
df['MainBetSport'].hist(bins=30, edgecolor='black')
plt.xlabel('Main Bet Sport')
plt.ylabel('Number of Users')
plt.title('MainBetSport distribution')
plt.xticks(rotation=90)
plt.show()

# Bar plot for SurveyAnswer
df['SurveyAnswer'].hist(bins=5, edgecolor='black')
plt.xlabel('Survey Answer')
plt.ylabel('Number of Users')
plt.title('SurveyAnswer distribution')
plt.xticks(rotation=90)
plt.show()

# Final data preparation
df = df[['Age', 'State', 'RegistrationDevice', 'FirstBetDevice', 'AcquisitionSource', 'MainBetSport', 'FirstWeekTurnover', 'SurveyAnswer', 'DaysReg', 'DaysToFirstBet']]
df = df[df['DaysToFirstBet'] >= 0].copy()
print(df.head(), '\n')
print("-" * 50)
print(df.info(), '\n')
print("-" * 50)

# Compare labeled vs unlabeled data distributions
has_label = df["SurveyAnswer"].notna()

plt.hist(df['DaysReg'], bins=50, alpha=0.7, label='All Data', color='green', edgecolor='black')
plt.hist(df[has_label]['DaysReg'], bins=50, alpha=0.7, label='Labeled Data', color='yellow', edgecolor='black')
plt.xlabel('Days')
plt.ylabel('Number of Users')
plt.title('Days between Registration and Registration_max: All Data vs Labeled Data')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Save to CSV
df.to_csv("data.csv", index=False)



