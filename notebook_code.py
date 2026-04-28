# Cell 1
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# File path to the original raw dataset
DATA_PATH = r"data/AI purchase cost project (STUDENT DATA) V3.csv"

# Let's load the data
df = pd.read_csv(DATA_PATH, sep=";", decimal=",")
print(f"Dataset Loaded Successfully! Shape: {df.shape}")
df.head()





# Cell 3
def preprocess_data(df):
    print("Starting preprocessing...")
    df_clean = df.copy()

    # 1. Drop Shipments
    if 'Shipment' in df_clean.columns:
        df_clean = df_clean.drop('Shipment', axis=1)

    # 2. Date parsing (LAADDATUM)
    df_clean['LAADDATUM'] = pd.to_datetime(df_clean['LAADDATUM'], format='%d/%m/%Y', errors='coerce')
    df_clean['Load_Year'] = df_clean['LAADDATUM'].dt.year
    df_clean['Load_Month'] = df_clean['LAADDATUM'].dt.month
    df_clean['Load_DayOfWeek'] = df_clean['LAADDATUM'].dt.dayofweek
    # We keep broader calendar patterns, but drop the day-of-month because
    # it showed almost no signal for the target and adds unnecessary noise.
    df_clean = df_clean.drop('LAADDATUM', axis=1)

    # 3. Y/N -> 1/0
    bool_cols = ['Crossdock', 'ADR', 'Express', 'Thermo']
    for col in bool_cols:
        df_clean[col] = df_clean[col].map({'Y': 1, 'N': 0}).fillna(0).astype('int8')



    # 5. Distance: stored as a European decimal string (comma as decimal separator).
    #    Convert to float and fill missing rows with the median so the model
    #    always receives a numeric value.
    if 'Distance' in df_clean.columns:
        df_clean['Distance'] = (
            df_clean['Distance']
            .astype(str)
            .str.replace(',', '.', regex=False)
            .replace('nan', float('nan'))
        )
        df_clean['Distance'] = pd.to_numeric(df_clean['Distance'], errors='coerce')
        distance_median = df_clean['Distance'].median()
        df_clean['Distance'] = df_clean['Distance'].fillna(distance_median)

    # 6. Handle High-Cardinality Variables (Frequency Encoding)
    high_card_cols = ['Load code', 'Unload code', 'Distribution driven by code']
    for col in high_card_cols:
        freq_encoding = df_clean[col].value_counts() / len(df_clean)
        df_clean[col + '_freq'] = df_clean[col].map(freq_encoding)
        df_clean = df_clean.drop(col, axis=1)

    return df_clean

df_processed = preprocess_data(df)
print(f"Data preprocessed! New Shape: {df_processed.shape}")
df_processed.head()


# Cell 5
# Encode target
le = LabelEncoder()
df_processed['Price category encoded'] = le.fit_transform(df_processed['Price category'].astype(str))

# Create mapping dictionary for future reference
price_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(f"Category Mapping (first 5 shown): {list(price_mapping.items())[:5]}")

# Define Features (X) and Target (y)
y = df_processed['Price category encoded']

# Drop target columns and features we do not want to use for modelling.
features_to_drop_before_modelling = ['Price category', 'Price category encoded', 'Load_Day']
X = df_processed.drop(columns=features_to_drop_before_modelling, errors='ignore')
print(f"Features removed before modelling: {features_to_drop_before_modelling}")

# Split data (80% Train, 20% validation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training features shape: {X_train.shape}")
print(f"Testing features shape: {X_test.shape}")


# Cell 7
from IPython.display import display
import matplotlib.pyplot as plt

# High-level dataset summary for documentation before modelling.
load_dates = pd.to_datetime(df['LAADDATUM'], format='%d-%m-%Y', errors='coerce')

dataset_overview = pd.DataFrame({
    'Metric': [
        'Rows',
        'Columns',
        'Date range start',
        'Date range end',
        'Missing cells',
        'Duplicate rows',
        'Unique price categories',
    ],
    'Value': [
        len(df),
        df.shape[1],
        load_dates.min().date(),
        load_dates.max().date(),
        int(df.isna().sum().sum()),
        int(df.duplicated().sum()),
        df['Price category'].nunique(),
    ]
})

feature_profile = pd.DataFrame({
    'Column': df.columns,
    'Data Type': df.dtypes.astype(str).values,
    'Missing Values': df.isna().sum().values,
    'Missing %': (df.isna().mean() * 100).round(2).values,
    'Unique Values': df.nunique().values,
})

print('Dataset overview:')
display(dataset_overview)

print('Feature profile:')
display(feature_profile.sort_values(['Missing Values', 'Unique Values'], ascending=[False, False]).reset_index(drop=True))

# Cell 9
price_counts = df['Price category'].value_counts().sort_values(ascending=False)
price_percent = price_counts / price_counts.sum()

price_distribution = pd.DataFrame({
    'Count': price_counts,
    'Share %': (price_percent * 100).round(2),
})

display(price_distribution)

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(price_counts.index.astype(str), price_counts.values, color='steelblue', edgecolor='black', linewidth=0.4)
ax.set_title('Price Category Distribution Across the Full Dataset')
ax.set_xlabel('Price Category')
ax.set_ylabel('Number of Shipments')
ax.grid(axis='y', linestyle='--', alpha=0.35)

for bar, pct in zip(bars, price_percent.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f'{pct:.1%}',
        ha='center',
        va='bottom',
        fontsize=8,
        rotation=90
    )

plt.tight_layout()
plt.show()

# Cell 11
corr_matrix = X.corr(numeric_only=True)

fig, ax = plt.subplots(figsize=(10, 8))
heatmap = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
ax.set_yticks(range(len(corr_matrix.index)))
ax.set_yticklabels(corr_matrix.index)
ax.set_title('Correlation Heatmap of Processed Features')

cbar = plt.colorbar(heatmap, ax=ax)
cbar.set_label('Correlation')

plt.tight_layout()
plt.show()

high_corr_pairs = (
    corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    .stack()
    .reset_index()
)
high_corr_pairs.columns = ['Feature 1', 'Feature 2', 'Correlation']
high_corr_pairs['Absolute Correlation'] = high_corr_pairs['Correlation'].abs()
high_corr_pairs = high_corr_pairs.sort_values('Absolute Correlation', ascending=False)

print('Strongest feature relationships:')
display(high_corr_pairs.head(10).round(3))

# Cell 14
from IPython.display import display
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt

# Estimate how much information each feature provides about the target.
mi_scores = mutual_info_classif(X_train, y_train, random_state=42)

feature_influence = (
    pd.DataFrame({
        'Feature': X_train.columns,
        'Mutual Information': mi_scores
    })
    .sort_values('Mutual Information', ascending=False)
    .reset_index(drop=True)
)
feature_influence['Influence Rank'] = feature_influence.index + 1
feature_influence = feature_influence[['Influence Rank', 'Feature', 'Mutual Information']]

print('Feature influence ranking (all features):')
display(feature_influence.style.format({'Mutual Information': '{:.4f}'}))

top_n = min(10, len(feature_influence))
top_features = feature_influence.head(top_n)

plt.figure(figsize=(10, 6))
plt.barh(top_features['Feature'][::-1], top_features['Mutual Information'][::-1], color='steelblue')
plt.xlabel('Mutual Information Score')
plt.ylabel('Feature')
plt.title(f'Top {top_n} Most Influential Features')
plt.tight_layout()
plt.show()

# Show how the strongest features differ across target classes.
summary_features = feature_influence.head(min(5, len(feature_influence)))['Feature'].tolist()
feature_summary_by_category = (
    df_processed.groupby('Price category')[summary_features]
    .agg(['mean', 'median'])
    .round(3)
)

print('How the top features vary across price categories:')
display(feature_summary_by_category)


# Cell 17
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Initialize the first model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# TODO: Fit the model using rf_model.fit(X_train, y_train) 
# Note: This dataframe is extremely large (~600k rows) so training might take a minute!


# Cell 19
from time import perf_counter
from sklearn.metrics import accuracy_score, classification_report, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

model_results = []
trained_models = {}

# Keep the original labels handy so reports are readable instead of looking like secret numeric codes.
target_names = [str(label) for label in le.classes_]


def evaluate_model(model_name, model, X_train, X_test, y_train, y_test):
    """Train a model, evaluate it, and store the results in a tidy table."""
    start_train = perf_counter()
    model.fit(X_train, y_train)
    train_seconds = perf_counter() - start_train

    start_predict = perf_counter()
    y_pred = model.predict(X_test)
    predict_seconds = perf_counter() - start_predict

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    model_results.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Macro F1': macro_f1,
        'Weighted F1': weighted_f1,
        'Training time (sec)': train_seconds,
        'Prediction time (sec)': predict_seconds
    })
    trained_models[model_name] = model

    print(f'{model_name} finished training. The machine did not complain, which is always a good start.')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Macro F1: {macro_f1:.4f}')
    print(f'Weighted F1: {weighted_f1:.4f}')
    print(f'Training time: {train_seconds:.1f} seconds')
    print(f'Prediction time: {predict_seconds:.1f} seconds')
    print('\nClassification report:')
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

    return y_pred

print('Evaluation helpers are ready.')
print(f'We are training on {X_train.shape[0]:,} rows and testing on {X_test.shape[0]:,} rows.')

# Cell 22
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=250,
    max_depth=None,
    min_samples_leaf=2,
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1
)

rf_predictions = evaluate_model(
    'Random Forest',
    rf_model,
    X_train,
    X_test,
    y_train,
    y_test
)

# Cell 25
fig, ax = plt.subplots(figsize=(10, 8))
ConfusionMatrixDisplay.from_predictions(
    y_test,
    rf_predictions,
    display_labels=target_names,
    xticks_rotation=45,
    cmap='Blues',
    ax=ax,
    colorbar=False
)
ax.set_title('Random Forest Confusion Matrix')
plt.tight_layout()
plt.show()

# Cell 28
rf_importance = (
    pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': rf_model.feature_importances_
    })
    .sort_values('Importance', ascending=False)
    .reset_index(drop=True)
)

print('Top Random Forest feature importances:')
display(rf_importance.head(15).style.format({'Importance': '{:.4f}'}))

plt.figure(figsize=(10, 6))
plt.barh(
    rf_importance.head(15)['Feature'][::-1],
    rf_importance.head(15)['Importance'][::-1],
    color='forestgreen'
)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Random Forest: Top 15 Feature Importances')
plt.tight_layout()
plt.show()

# Cell 31
try:
    from xgboost import XGBClassifier

    boosted_model_name = 'XGBoost'
    boosted_model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        objective='multi:softprob',
        eval_metric='mlogloss',
        tree_method='hist',
        random_state=42,
        n_jobs=-1
    )
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier

    boosted_model_name = 'HistGradientBoosting'
    boosted_model = HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        early_stopping=True,
        random_state=42
    )

print(f'Using {boosted_model_name} for the gradient-boosted tree model.')

boosted_predictions = evaluate_model(
    boosted_model_name,
    boosted_model,
    X_train,
    X_test,
    y_train,
    y_test
)

# Cell 34
fig, ax = plt.subplots(figsize=(10, 8))
ConfusionMatrixDisplay.from_predictions(
    y_test,
    boosted_predictions,
    display_labels=target_names,
    xticks_rotation=45,
    cmap='Purples',
    ax=ax,
    colorbar=False
)
ax.set_title(f'{boosted_model_name} Confusion Matrix')
plt.tight_layout()
plt.show()

# Cell 37
from sklearn.inspection import permutation_importance

boosted_importance_values = getattr(boosted_model, 'feature_importances_', None)
importance_source = 'built-in tree importance'

if boosted_importance_values is None:
    importance_source = 'permutation importance on a test-set sample'
    sample_size = min(20000, len(X_test))
    X_importance = X_test.sample(n=sample_size, random_state=42)
    y_importance = y_test.loc[X_importance.index]

    permutation_scores = permutation_importance(
        boosted_model,
        X_importance,
        y_importance,
        n_repeats=3,
        random_state=42,
        scoring='f1_weighted',
        n_jobs=-1
    )
    boosted_importance_values = permutation_scores.importances_mean

boosted_importance = (
    pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': boosted_importance_values
    })
    .sort_values('Importance', ascending=False)
    .reset_index(drop=True)
)

print(f'Top {boosted_model_name} feature importances using {importance_source}:')
display(boosted_importance.head(15).style.format({'Importance': '{:.4f}'}))

plt.figure(figsize=(10, 6))
plt.barh(
    boosted_importance.head(15)['Feature'][::-1],
    boosted_importance.head(15)['Importance'][::-1],
    color='mediumpurple'
)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title(f'{boosted_model_name}: Top 15 Feature Importances')
plt.tight_layout()
plt.show()

# Cell 40
results_df = (
    pd.DataFrame(model_results)
    .sort_values(['Weighted F1', 'Macro F1', 'Accuracy'], ascending=False)
    .reset_index(drop=True)
)

print('Model comparison:')
display(results_df.style.format({
    'Accuracy': '{:.4f}',
    'Macro F1': '{:.4f}',
    'Weighted F1': '{:.4f}',
    'Training time (sec)': '{:.1f}',
    'Prediction time (sec)': '{:.1f}'
}))

best_model_name = results_df.loc[0, 'Model']
best_model = trained_models[best_model_name]

print(f'Best model based on weighted F1, with macro F1 as the tie-breaker: {best_model_name}')