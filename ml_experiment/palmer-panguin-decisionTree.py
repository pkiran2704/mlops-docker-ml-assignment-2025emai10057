import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Load Data
print("Fetching Palmer Penguin dataset...")
df = sns.load_dataset("penguins").dropna()

# Preprocessing
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])
X = pd.get_dummies(df.drop(columns=['species', 'island', 'sex']), drop_first=True)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train, y_train)

# Metrics
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)
print(f"Model Precision: {precision_score(y_test, y_pred, average='macro'):.4f}")
print(f"Model AUC Score: {roc_auc_score(y_test, y_proba, multi_class='ovr'):.4f}")