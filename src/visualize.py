import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix


def plot_confusion(
    y_true,
    y_pred,
    labels=None,
    class_names=None,
    title="Confusion Matrix"
):
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Use class names ONLY for display
    display_labels = class_names if class_names is not None else labels

    fig = ff.create_annotated_heatmap(
        z=cm,
        x=display_labels,
        y=display_labels,
        colorscale='Blues',
        showscale=True
    )

    fig.update_yaxes(autorange='reversed')
    fig.update_layout(title=title)
    fig.show()




def plot_regression(y_true, y_pred, title="Regression: True vs Predicted"):
    fig = px.scatter(
        x=y_true, 
        y=y_pred, 
        labels={'x': 'True', 'y': 'Predicted'},
        title=title
    )
    fig.add_shape(
        type='line',
        x0=min(y_true), x1=max(y_true),
        y0=min(y_true), y1=max(y_true),
        line=dict(color='red', dash='dash')
    )
    fig.show()


def compare_models(metrics_list, metric_name="Accuracy", title="Model Comparison"):
    
    import pandas as pd
    df = pd.DataFrame(metrics_list)
    fig = px.bar(
        df,
        x="Model",
        y=metric_name,
        text=metric_name,
        title=title
    )
    fig.show()