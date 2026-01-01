from sklearn.model_selection import train_test_split
from metrics import classification_metrics, regression_metrics
from visualize import plot_confusion, plot_regression

def train_and_evaluate(
    model,
    X,
    y,
    task="classification",
    test_size=0.2,
    random_state=42,
    labels=None,
    class_names=None,
    plot=True
):

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if task=="classification" else None
    )

    # Fit the model
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Classification task
    if task == "classification":
        metrics = classification_metrics(y_test, y_pred)

        # Plot confusion matrix safely
        if plot:
            plot_confusion(y_test, y_pred, labels=labels,class_names=class_names
                           )

    # Regression task
    else:
        metrics = regression_metrics(y_test, y_pred)

        if plot:
            plot_regression(y_test, y_pred)

    return metrics
