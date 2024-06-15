import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
file_path = '/path/to/your/Titanic-Dataset.csv'  # update this path as needed
titanic_df = pd.read_csv(file_path)

# Data preprocessing
# Fill missing Age values with the median
titanic_df['Age'].fillna(titanic_df['Age'].median(), inplace=True)

# Drop the Cabin column
titanic_df.drop(columns=['Cabin'], inplace=True)

# Fill missing Embarked values with the most common port
titanic_df['Embarked'].fillna(titanic_df['Embarked'].mode()[0], inplace=True)

# Create FamilySize feature
titanic_df['FamilySize'] = titanic_df['SibSp'] + titanic_df['Parch'] + 1

# Create IsAlone feature
titanic_df['IsAlone'] = 1  # initialize to 1 (true)
titanic_df['IsAlone'].loc[titanic_df['FamilySize'] > 1] = 0  # set to 0 (false) if FamilySize > 1

# Encode categorical variables
titanic_df = pd.get_dummies(titanic_df, columns=['Sex', 'Embarked'], drop_first=True)

# Drop unnecessary columns
titanic_df.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)

# Split data into features and target
X = titanic_df.drop(columns=['Survived'])
y = titanic_df['Survived']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Initial Model Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# Hyperparameter Tuning with Grid Search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize Grid Search with cross-validation
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, 
                           cv=5, n_jobs=-1, verbose=2, scoring='accuracy')

# Perform Grid Search to find the best parameters
grid_search.fit(X_train, y_train)

# Get the best parameters and the best estimator
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Make predictions with the best model
best_y_pred = best_model.predict(X_test)

# Evaluate the best model
best_accuracy = accuracy_score(y_test, best_y_pred)
best_conf_matrix = confusion_matrix(y_test, best_y_pred)
best_class_report = classification_report(y_test, best_y_pred)

print("Best Parameters:", best_params)
print("Best Model Accuracy:", best_accuracy)
print("Confusion Matrix:\n", best_conf_matrix)
print("Classification Report:\n", best_class_report)
