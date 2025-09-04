# get the needed packages

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as ss
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from imblearn.over_sampling import RandomOverSampler
import warnings
warnings.filterwarnings("ignore")

# Check how two categories are related
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

# Show pictures of category counts
def visualize_categorical_columns(df):
    cat_cols = [col for col in df.columns 
                if df[col].dtype == 'object' or str(df[col].dtype) == 'category']
    
    if not cat_cols:
        print("No categorical columns found.")
        return
    
    for col in cat_cols:
        plt.figure(figsize=(8, 4))
        sns.countplot(data=df, x=col)
        plt.title(f'Count Plot of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# Print how many of each category
def print_categorical_value_counts(df):
    cat_cols = [col for col in df.columns if df[col].dtype == 'object' or str(df[col].dtype) == 'category']
    
    for col in cat_cols:
        print(f"Value counts for '{col}':")
        print(df[col].value_counts())
        print("-" * 50)

# Show age spread picture
def plot_age_distribution(df):
    sns.histplot(df['Age'], bins=30, stat='density', color='skyblue', label='Age Histogram')
    mu, std = df['Age'].mean(), df['Age'].std()
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'r', linewidth=2, label='Normal Curve')
    plt.title('Age Distribution with Normal Curve')
    plt.xlabel('Age')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Show age box picture
def plot_age_boxplot(df):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df['Age'])
    plt.title('Age Boxplot')
    plt.xlabel('Age')
    plt.tight_layout()
    plt.show()

# Main work function
def main():
    df = pd.read_excel(r"covid19.xlsx")
    
    # Show category pictures
    print("Plotting categorical column distributions...")
    visualize_categorical_columns(df)
    
    # Print category numbers
    print("Categorical value counts:")
    print_categorical_value_counts(df)
    
    # Remove empty columns
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    cols_to_drop = missing_percentage[missing_percentage > 80].index.tolist()
    df = df.drop(columns=cols_to_drop)
    
    # Calculate age from birth year
    current_year = pd.Timestamp.now().year
    df['Age'] = current_year - df['Birth Year']
    
    # Show age pictures
    print("Plotting age distribution...")
    plot_age_distribution(df)
    plot_age_boxplot(df)
    
    # Fill empty categories
    cat_cols = [col for col in df.columns if df[col].dtype == 'object' or str(df[col].dtype) == 'category']
    df[cat_cols] = df[cat_cols].fillna("UNKNOWN")
    
    # Fill empty ages
    median_age = df['Age'].median()
    df['Age'] = df['Age'].fillna(median_age)
    df.drop(columns=['Birth Year'], inplace=True)
    
    # Clean up sex column
    df['Sex'] = df['Sex'].replace("OTHER", "UNKNOWN")
    
    # Clean up result column
    df['Result'] = df['Result'].apply(lambda x: 'POSITIVE' if x == 'POSITIVE' else 'NOT_POSITIVE')
    df['Result'] = df['Result'].map({'NOT_POSITIVE': 0, 'POSITIVE': 1})
    
    # Split data
    X = df.drop(columns=['Result'])
    y = df['Result']
    
    # Convert categories to numbers
    cat_cols = [col for col in X.columns if X[col].dtype == 'object' or str(X[col].dtype) == 'category']
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=False).astype(int)
    
    # Split for testing
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    
    # Try different models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    # Test first models
    print("Initial Model Performance:")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"\n{name} - Accuracy: {accuracy_score(y_test, y_pred):.3f}")
        print(classification_report(y_test, y_pred))
    
    # Fix unbalanced data
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    
    # Convert again
    cat_cols = [col for col in X_resampled.columns 
                if X_resampled[col].dtype == 'object' or str(X_resampled[col].dtype) == 'category']
    X_resampled_encoded = pd.get_dummies(X_resampled, columns=cat_cols, drop_first=False).astype(int)
    
    # Split again
    X_train, X_test, y_train, y_test = train_test_split(X_resampled_encoded, y_resampled, test_size=0.2, random_state=42)
    
    # Test models again
    print("\nOversampled Model Performance:")
    best_accuracy = 0
    best_model_name = None
    best_model = None
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"\n{name} - Accuracy: {acc:.3f}")
        print(classification_report(y_test, y_pred))
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name
            best_model = model
    
    print(f"\nBest model: {best_model_name} ({best_accuracy:.3f})")
    
    # Tune best model
    if best_model_name == "Random Forest":
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
    elif best_model_name == "Decision Tree":
        param_grid = {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
    elif best_model_name == "Logistic Regression":
        param_grid = {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }
    elif best_model_name == "Gradient Boosting":
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    else:
        param_grid = {}
    
    if param_grid:
        grid_search = GridSearchCV(estimator=best_model,
                                   param_grid=param_grid,
                                   cv=5,
                                   scoring='accuracy',
                                   n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        print("\nBest hyperparameters:", grid_search.best_params_)
        print("Best cross-validation accuracy:", grid_search.best_score_)
        
        tuned_model = grid_search.best_estimator_
        y_pred_tuned = tuned_model.predict(X_test)
        print("\nTuned Model Test Performance:")
        print("Test Set Accuracy:", accuracy_score(y_test, y_pred_tuned))
        print(classification_report(y_test, y_pred_tuned))
        
        # Show ROC curve
        y_probs = tuned_model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_probs, pos_label=1)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8,6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.show()
        
        # Test predictions
        sample_data = X_resampled_encoded.head(5)
        sample_predictions = tuned_model.predict(sample_data)
        print("\nSample predictions (first 5):", sample_predictions)
        
        sample_data = X_resampled_encoded.tail(5)
        sample_predictions = tuned_model.predict(sample_data)
        print("Sample predictions (last 5):", sample_predictions)

if __name__ == "__main__":
    main()