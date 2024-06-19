# Import packages needed
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, log_loss
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Set directory
in_directory = "/home/david/bioinformatica/Pruebas/Biomarcadores_Transposed/Todos_filtered_imputed.csv"
out_directory = "/home/david/bioinformatica/Pruebas/Figures/"

results = []


def evaluation_model(model_name, model, param_grid, input_directory):
    """Performs the 5 models with the implementation of the PCA and evaluates the confusion matrices, ROC Curves,
     and Learning curves for validation and test sets"""
    data_frame = pd.read_csv(input_directory)

    # Divide the dataset into independent (x) and dependent variables (y)
    x = data_frame.drop(['BH', 'Symptomatic'], axis=1)
    y = data_frame['Symptomatic']

    # Standardize the features
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=62)
    x_pca = pca.fit_transform(x_scaled)

    # explained variance for the selected components
    explained_variance = pca.explained_variance_ratio_
    print(f"Total explained variance: {np.sum(explained_variance)}")

    # Split the data into training (80%) and test (20%) sets after PCA
    x_train_val, x_test, y_train_val, y_test = train_test_split(x_pca, y, test_size=0.2, stratify=y, random_state=40)

    # Split the training data into training (75%) and validation (25%) sets
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.25, stratify=y_train_val,
                                                      random_state=40)

    # Perform GridSearchCV
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc')
    grid_search.fit(x_train, y_train)
    best_model = grid_search.best_estimator_
    best_params = best_model.get_params()
    print(f"Best hyperparameters for {model_name}: {best_params}")

    # Model evaluation on validation set
    y_val_pred = best_model.predict(x_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_proba = best_model.predict_proba(x_val)[:, 1]
    print(f"Validation Accuracy for {model_name}: {val_accuracy}")

    # Model evaluation on test set
    y_test_pred = best_model.predict(x_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_proba = best_model.predict_proba(x_test)[:, 1]
    test_log_loss = log_loss(y_test, test_proba)
    print(f"Test Accuracy for {model_name}: {test_accuracy}")
    print(f"Test Log-Loss for {model_name}: {test_log_loss}")

    # Cross validation
    cv_scores = cross_val_score(best_model, x_train, y_train, cv=5, scoring='accuracy')
    cv_accuracy = np.mean(cv_scores)
    print(f"Cross-validation Accuracy for {model_name}: {cv_accuracy}")

    # ROC AUC for validation
    val_fpr, val_tpr, _ = roc_curve(y_val, val_proba)
    roc_auc_val = auc(val_fpr, val_tpr)
    print(f"Validation ROC AUC for {model_name}: {roc_auc_val}")

    # ROC AUC for test
    test_fpr, test_tpr, _ = roc_curve(y_test, test_proba)
    roc_auc_test = auc(test_fpr, test_tpr)
    print(f"Test ROC AUC for {model_name}: {roc_auc_test}")

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(test_fpr, test_tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc_test:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend(loc='lower right')
    roc_curve_file = os.path.join(out_directory, f'ROC_curve_{model_name}.png')
    plt.savefig(roc_curve_file)
    plt.close()
    plt.show()

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    sns.heatmap(
        conf_matrix,
        annot=True,
        cbar=False,
        annot_kws={"size": 8},
        vmin=-1,
        vmax=1,
        center=0,
        cmap='RdBu',
        square=True,
        ax=ax
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right',
    )
    ax.tick_params(labelsize=10)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix for {model_name}')
    conf_matrix_file = os.path.join(out_directory, f'Confusion_matrix_{model_name}.png')
    plt.savefig(conf_matrix_file)
    plt.close()
    plt.show()

    # Learning curves
    title = f"Learning Curves ({model_name})"
    plot_learning_curve(best_model, title, x_train, y_train, cv=5)
    learning_curve_file = os.path.join(out_directory, f'Learning_curve_{model_name}.png')
    plt.savefig(learning_curve_file)
    plt.show()

    return {
        'Model': model_name,
        'Best Hyperparameters': best_params,
        'Validation Accuracy': val_accuracy,
        'Test Accuracy': test_accuracy,
        'Validation ROC AUC': roc_auc_val,
        'Test ROC AUC': roc_auc_test,
        'Test Log-Loss': test_log_loss
    }


def plot_learning_curve(estimator, title, x, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """Generate plots of the test and training learning curve."""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, x, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="b")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="b",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# Define models and their respective parameter grids
models = [
    ('KNN', KNeighborsClassifier(), {'n_neighbors': np.arange(1, 58), 'metric': ['euclidean', 'manhattan', 'chebyshev',
                                                                                 'minkowski']}),
    ('SVM', SVC(class_weight='balanced', probability=True), {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                                                             'C': [0.01, 0.1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001],
                                                             'coef0': [1, 0.1, 0.01, 0.001]}),
    ('Random Forest', RandomForestClassifier(class_weight='balanced'), {'n_estimators': [100, 200, 300, 500, 1000],
                                                                        'max_depth': [10, 20, 50, 100],
                                                                        'min_samples_split': [25, 100, 300],
                                                                        'min_samples_leaf': [25, 100, 300],
                                                                        'max_features': ['sqrt', 'log2']}),
    ('XGBoost', XGBClassifier(), {'n_estimators': [100, 200, 300, 500], 'max_depth': [10, 20, 50, 100],
                                  'subsample': [0.7, 0.9], 'colsample_bytree': [0.7, 0.9], 'learning_rate': [0.01, 0.1]}),
    ('Logistic Regression', LogisticRegression(), {'C': [0.001, 0.01, 0.1], 'penalty': ['l1', 'l2'],
                                                   'solver': ['saga'], 'max_iter': [10000]})
]

# Train and evaluate each model
for model_name, model, param_grid in models:
    model_results = evaluation_model(model_name, model, param_grid, in_directory)
    results.append(model_results)

results_df = pd.DataFrame(results)
results_df.to_csv('/home/david/bioinformatica/Pruebas/Figures/Results.csv', index=False)

print(results_df)
