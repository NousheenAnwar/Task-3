Output

Shape of dataset: (10000, 14)

First 5 rows:
   RowNumber  CustomerId   Surname  CreditScore Geography  Gender  ...    Balance  NumOfProducts  HasCrCard  IsActiveMember  EstimatedSalary  Exited
0          1    15634602  Hargrave          619    France  Female  ...       0.00              1          1               1        101348.88       1
1          2    15647311      Hill          608     Spain  Female  ...   83807.86              1          0               1        112542.58       0
2          3    15619304      Onio          502    France  Female  ...  159660.80              3          1               0        113931.57       1
3          4    15701354      Boni          699    France  Female  ...       0.00              2          0               0         93826.63       0
4          5    15737888  Mitchell          850     Spain  Female  ...  125510.82              1          1               1         79084.10       0

[5 rows x 14 columns]

Column names: ['RowNumber', 'CustomerId', 'Surname', 'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited']

Missing values:
 RowNumber          0
CustomerId         0
Surname            0
CreditScore        0
Geography          0
Gender             0
Age                0
Tenure             0
Balance            0
NumOfProducts      0
HasCrCard          0
IsActiveMember     0
EstimatedSalary    0
Exited             0
dtype: int64

After dropping useless columns:
   CreditScore Geography  Gender  Age  Tenure    Balance  NumOfProducts  HasCrCard  IsActiveMember  EstimatedSalary  Exited
0          619    France  Female   42       2       0.00              1          1               1        101348.88       1
1          608     Spain  Female   41       1   83807.86              1          0               1        112542.58       0
2          502    France  Female   42       8  159660.80              3          1               0        113931.57       1
3          699    France  Female   39       1       0.00              2          0               0         93826.63       0
4          850     Spain  Female   43       2  125510.82              1          1               1         79084.10       0

Gender after Label Encoding (0=Female, 1=Male):
Gender
1    5457
0    4543
Name: count, dtype: int64

Columns after One-Hot Encoding Geography:
['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited', 'Geography_Germany', 'Geography_Spain']

Features (X) shape: (10000, 11)
Target (y) shape: (10000,)

Churn distribution:
 Exited
0    7963
1    2037
Name: count, dtype: int64

Training samples: 8000
Testing samples:  2000

✅ Model training complete!

🎯 Accuracy: 86.60%

📊 Classification Report:
              precision    recall  f1-score   support

      Stayed       0.88      0.97      0.92      1607
     Churned       0.77      0.46      0.57       393

    accuracy                           0.87      2000
   macro avg       0.82      0.71      0.75      2000
weighted avg       0.86      0.87      0.85      2000


📈 Plot saved as 'churn_results.png'

=======================================================
🔍 TOP 5 FEATURES THAT INFLUENCE CHURN:
=======================================================
  1. Age                       → 0.2398
  2. EstimatedSalary           → 0.1466
  3. CreditScore               → 0.1442
  4. Balance                   → 0.1386
  5. NumOfProducts             → 0.1303

📌 WHAT DO THESE RESULTS MEAN?
-----------------------------------------------
• Accuracy ~86%: The model correctly predicts whether
  a customer will churn 86% of the time.

• Feature Importance tells us WHICH factors matter most
  for predicting churn. Higher score = more influence.

• Common top features in this dataset:
  - Age          → Older customers churn more
  - NumOfProducts→ Having 1 or 3+ products increases risk
  - IsActiveMember→ Inactive members are more likely to leave
  - Balance      → High balance but no activity = churn risk
  - Geography    → Germany has higher churn rate

• Confusion Matrix:
  - True Positives (bottom-right): Correctly predicted churners
  - False Negatives (bottom-left): Missed churners (most costly!)
