import pandas as pd
import skops.io as sio
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

config: dict[str, any] = {
    "num_estimators": 20,
    "random_state": 12345,
    "train_size": 0.75,
    "test_size": 0.25,
    "impute_strategy": "median"
}

## Loading the Data
drug_df = pd.read_csv(r"Data/drug.csv")
drug_df = drug_df.sample(frac=1)

## Train Test Split
from sklearn.model_selection import train_test_split

X = drug_df.drop("Drug", axis=1).values
y = drug_df.Drug.values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=config['test_size'], random_state=config['random_state']
)


## Pipeline
cat_col = [1,2,3]
num_col = [0,4]

transform = ColumnTransformer(
    [
        ("encoder", OrdinalEncoder(), cat_col),
        ("num_imputer", SimpleImputer(strategy=config['impute_strategy']), num_col),
        ("num_scaler", StandardScaler(), num_col),
    ]
)
pipe = Pipeline(
    steps=[
        ("preprocessing", transform),
        (
            "model",
            RandomForestClassifier(
                n_estimators=config["num_estimators"],
                random_state=config["random_state"],
                verbose = 2
            ),
        ),
    ]
)

## Training
pipe.fit(X_train, y_train)


## Model Evaluation
predictions = pipe.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="macro")

print("Accuracy:", str(round(accuracy, 2) * 100) + "%", "F1:", round(f1, 2))


## Confusion Matrix Plot
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

predictions = pipe.predict(X_test)
cm = confusion_matrix(y_test, predictions, labels=pipe.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_)
disp.plot()
plt.savefig("Results/model_results.png", dpi=200)

## Write metrics to file
with open("Results/metrics.txt", "w") as outfile:
    outfile.write(f"\nAccuracy = {round(accuracy, 2)}, F1 Score = {round(f1, 2)}")

## Saving the model file
sio.dump(pipe, "Model/drug_pipeline.skops")
