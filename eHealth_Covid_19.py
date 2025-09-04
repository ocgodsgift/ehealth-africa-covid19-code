# Import necessary libraries for data processing, visualization, and machine learning
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

# Function to calculate Cramér's V statistic for measuring association between categorical variables
def cramers_v(x, y):
    """
    Calculates the strength of association between two categorical variables.
    Cramér's V ranges from 0 (no association) to 1 (perfect association).
    
    Parameters:
    x, y: Two categorical variables (pandas Series)
    
    Returns:
    Cramér's V statistic value
    """
    # Create a contingency table (cross-tabulation) of the two variables
    confusion_matrix = pd.crosstab(x, y)
    # Calculate chi-squared statistic from the contingency table
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum() 
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    
    # Apply bias correction to phi-squared
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    # Apply bias correction to row and column counts
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    
    # Calculate and return Cramér's V
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

# Function to visualize the distribution of categorical variables using count plots
def visualize_categorical_columns(df):
    """
    Creates count plots for all categorical columns in the DataFrame.
    Helps understand the frequency distribution of each categorical variable.
    
    Parameters:
    df: pandas DataFrame containing the data
    """
    # Identify categorical columns (object type or explicitly categorical)
    cat_cols = [col for col in df.columns 
                if df[col].dtype == 'object' or str(df[col].dtype) == 'category']
    
    if not cat_cols:
        print("No categorical columns found.")
        return
    
    # Create a count plot for each categorical column
    for col in cat_cols:
        plt.figure(figsize=(8, 4))
        sns.countplot(data=df, x=col)
        plt.title(f'Count Plot of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45) 
        plt.tight_layout()
        plt.show()

# Function to display value counts for all categorical variables
def print_categorical_value_counts(df):
    """
    Prints the frequency distribution of values for each categorical column.
    Useful for understanding the composition of categorical data.
    
    Parameters:
    df: pandas DataFrame containing the data
    """
    # Identify categorical columns
    cat_cols = [col for col in df.columns if df[col].dtype == 'object' or str(df[col].dtype) == 'category']
    
    # Print value counts for each categorical column
    for col in cat_cols:
        print(f"Value counts for '{col}':")
        print(df[col].value_counts())
        print("-" * 50)

# Function to visualize the age distribution with a normal curve overlay
def plot_age_distribution(df):
    """
    Creates a histogram of age distribution with a fitted normal curve.
    Helps assess whether age follows a normal distribution.
    
    Parameters:
    df: pandas DataFrame containing an 'Age' column
    """
    # Create histogram of age values
    sns.histplot(df['Age'], bins=30, stat='density', color='skyblue', label='Age Histogram')
    
    # Calculate mean and standard deviation of age
    mu, std = df['Age'].mean(), df['Age'].std()
    
    # Generate points for the normal curve
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    
    # Plot the normal curve
    plt.plot(x, p, 'r', linewidth=2, label='Normal Curve')
    plt.title('Age Distribution with Normal Curve')
    plt.xlabel('Age')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Function to create a box plot of age distribution
def plot_age_boxplot(df):
    """
    Creates a box plot to visualize the distribution of age values.
    Shows median, quartiles, and potential outliers.
    
    Parameters:
    df: pandas DataFrame containing an 'Age' column
    """
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df['Age'])
    plt.title('Age Boxplot')
    plt.xlabel('Age')
    plt.tight_layout()
    plt.show()

# Main function that orchestrates the entire data analysis and modeling process
def main():
    """
    Main function that loads data, performs preprocessing, exploratory analysis,
    and builds machine learning models for COVID-19 test result prediction.
    """
    # Load the COVID-19 dataset from Excel file
    df = pd.read_excel(r"covid19.xlsx")
    
    # Visualize distributions of categorical variables
    print("Plotting categorical column distributions...")
    visualize_categorical_columns(df)
    
    # Display value counts for categorical variables
    print("Categorical value counts:")
    print_categorical_value_counts(df)
    
    # Handle missing data: remove columns with more than 80% missing values
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    cols_to_drop = missing_percentage[missing_percentage > 80].index.tolist()
    df = df.drop(columns=cols_to_drop)
    
    # Create Age feature from Birth Year (assuming current year for calculation)
    current_year = pd.Timestamp.now().year
    df['Age'] = current_year - df['Birth Year']
    
    # Visualize age distribution
    print("Plotting age distribution...")
    plot_age_distribution(df)
    plot_age_boxplot(df)
    
    # Handle missing values in categorical columns by filling with "UNKNOWN"
    cat_cols = [col for col in df.columns if df[col].dtype == 'object' or str(df[col].dtype) == 'category']
    df[cat_cols] = df[cat_cols].fillna("UNKNOWN")
    
    # Handle missing age values by filling with median age
    median_age = df['Age'].median()
    df['Age'] = df['Age'].fillna(median_age)
    df.drop(columns=['Birth Year'], inplace=True) 
    
    # Clean up the Sex column by standardizing "OTHER" values to "UNKNOWN"
    df['Sex'] = df['Sex'].replace("OTHER", "UNKNOWN")
    
    # Preprocess the target variable: convert to binary classification
    # "POSITIVE" becomes 1, all other results become 0
    df['Result'] = df['Result'].apply(lambda x: 'POSITIVE' if x == 'POSITIVE' else 'NOT_POSITIVE')
    df['Result'] = df['Result'].map({'NOT_POSITIVE': 0, 'POSITIVE': 1})
    
    # Prepare features (X) and target variable (y) for modeling
    X = df.drop(columns=['Result'])
    y = df['Result']
    
    # Convert categorical variables to numerical format using one-hot encoding
    cat_cols = [col for col in X.columns if X[col].dtype == 'object' or str(X[col].dtype) == 'category']
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=False).astype(int)
    
    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    
    # Define multiple classification models to compare
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    # Evaluate initial performance of all models
    print("Initial Model Performance:")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"\n{name} - Accuracy: {accuracy_score(y_test, y_pred):.3f}")
        print(classification_report(y_test, y_pred))
    
    # Handle class imbalance using Random OverSampling
    # This creates synthetic samples of the minority class to balance the dataset
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    
    # Re-encode categorical variables after resampling
    cat_cols = [col for col in X_resampled.columns 
                if X_resampled[col].dtype == 'object' or str(X_resampled[col].dtype) == 'category']
    X_resampled_encoded = pd.get_dummies(X_resampled, columns=cat_cols, drop_first=False).astype(int)
    
    # Split the resampled data into new training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled_encoded, y_resampled, test_size=0.2, random_state=42)
    
    # Evaluate models on the balanced dataset
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
        
        # Track the best performing model
        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name
            best_model = model
    
    print(f"\nBest model: {best_model_name} ({best_accuracy:.3f})")
    
    # Perform hyperparameter tuning for the best model using GridSearchCV
    # Different parameter grids are defined for different model types
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
    
    # Perform grid search if parameters are defined
    if param_grid:
        grid_search = GridSearchCV(estimator=best_model,
                                   param_grid=param_grid,
                                   cv=5,
                                   scoring='accuracy',
                                   n_jobs=-1)
        
        # Fit grid search to training data
        grid_search.fit(X_train, y_train)
        
        print("\nBest hyperparameters:", grid_search.best_params_)
        print("Best cross-validation accuracy:", grid_search.best_score_)
        
        # Evaluate the tuned model on test data
        tuned_model = grid_search.best_estimator_
        y_pred_tuned = tuned_model.predict(X_test)
        print("\nTuned Model Test Performance:")
        print("Test Set Accuracy:", accuracy_score(y_test, y_pred_tuned))
        print(classification_report(y_test, y_pred_tuned))
        
        # Create ROC curve to visualize model performance across thresholds
        y_probs = tuned_model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_probs, pos_label=1)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
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
        
        # Make sample predictions on first and last 5 instances
        sample_data = X_resampled_encoded.head(5)
        sample_predictions = tuned_model.predict(sample_data)
        print("\nSample predictions (first 5):", sample_predictions)
        
        sample_data = X_resampled_encoded.tail(5)
        sample_predictions = tuned_model.predict(sample_data)
        print("Sample predictions (last 5):", sample_predictions)

# Entry point of the script
if __name__ == "__main__":
    main()