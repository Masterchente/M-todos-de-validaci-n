from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from collections import Counter
import numpy as np

def load_datasets():
    iris = load_iris()
    wine = load_wine()
    return iris, wine

def print_class_distribution(y, dataset_name):
    """
    :param y: Clases
    :param dataset_name: Nombre del dataset
    """
    class_distribution = Counter(y)
    print(f"\nDistribución de clases en {dataset_name}:")
    for clase, count in class_distribution.items():
        print(f"Clase {clase}: {count} muestras")

def hold_out(X, y, r):
    """
    :param X: Características
    :param y: Clases
    :param r: Proporción para el conjunto de prueba
    :return: Conjuntos de entrenamiento y prueba
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=r, random_state=42)
    return X_train, X_test, y_train, y_test

def k_fold_cross_validation(X, y, k):
    """
    :param X: Características
    :param y: Clases
    :param k: Número de folds
    :return: Lista de conjuntos de entrenamiento y prueba para cada fold
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    folds = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        folds.append((X_train, X_test, y_train, y_test))
    return folds

def leave_one_out(X, y):
    """
    :param X: Features
    :param y: Labels
    :return: Lista de conjuntos de entrenamiento y prueba para cada iteración
    """
    loo = LeaveOneOut()
    folds = []
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        folds.append((X_train, X_test, y_train, y_test))
    return folds

def evaluate_model(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def main():
    iris, wine = load_datasets()
    print_class_distribution(iris.target, "Iris")
    print_class_distribution(wine.target, "Wine")

    # <-- Hold-Out -->
    r = 0.2  # 20% para prueba
    X_train, X_test, y_train, y_test = hold_out(wine.data, wine.target, r)
    print(f"\nHold-Out (Wine): Tamaño de entrenamiento: {len(X_train)}, Tamaño de prueba: {len(X_test)}")
    accuracy = evaluate_model(X_train, X_test, y_train, y_test)
    print(f"Accuracy (Hold-Out): {accuracy:.4f}")


    X_train, X_test, y_train, y_test = hold_out(iris.data, iris.target, r)
    print(f"\nHold-Out (Iris): Tamaño de entrenamiento: {len(X_train)}, Tamaño de prueba: {len(X_test)}")
    accuracy = evaluate_model(X_train, X_test, y_train, y_test)
    print(f"Accuracy (Hold-Out): {accuracy:.4f}")
 
    # <-- K-Fold para Wine -->
    k = 5
    folds = k_fold_cross_validation(wine.data, wine.target, k)
    print(f"\nK-Fold (Wine): Numero de folds: {len(folds)}")
    accuracies = []
    for i, (X_train, X_test, y_train, y_test) in enumerate(folds):
        accuracy = evaluate_model(X_train, X_test, y_train, y_test)
        accuracies.append(accuracy)
        print(f"Fold {i+1} Accuracy: {accuracy:.4f}")
    print(f"Promedio de Accuracy (K-Fold): {np.mean(accuracies):.4f}")

    folds = k_fold_cross_validation(iris.data, iris.target, k)
    print(f"\nK-Fold (Iris): Numero de folds: {len(folds)}")
    accuracies = []
    for i, (X_train, X_test, y_train, y_test) in enumerate(folds):
        accuracy = evaluate_model(X_train, X_test, y_train, y_test)
        accuracies.append(accuracy)
        print(f"Fold {i+1} Accuracy: {accuracy:.4f}")
    print(f"Promedio de Accuracy (K-Fold): {np.mean(accuracies):.4f}")


    # Leave-One-Out para Iris
    loo_folds = leave_one_out(iris.data, iris.target)
    print(f"\nLeave-One-Out (Iris): Number de iterations: {len(loo_folds)}")
    accuracies = []
    for X_train, X_test, y_train, y_test in loo_folds:
        accuracy = evaluate_model(X_train, X_test, y_train, y_test)
        accuracies.append(accuracy)
    print(f"Promedio de Accuracy (Leave-One-Out): {np.mean(accuracies):.4f}")

    loo_folds = leave_one_out(wine.data, wine.target)
    print(f"\nLeave-One-Out (Wine): Number de iterations: {len(loo_folds)}")
    accuracies = []
    for X_train, X_test, y_train, y_test in loo_folds:
        accuracy = evaluate_model(X_train, X_test, y_train, y_test)
        accuracies.append(accuracy)
    print(f"Promedio de Accuracy (Leave-One-Out): {np.mean(accuracies):.4f}")

if __name__ == "__main__":
    main()