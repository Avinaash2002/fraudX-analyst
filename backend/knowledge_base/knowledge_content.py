"""
FraudX Analyst - Knowledge Base Content
=========================================
All fraud education content that will be uploaded to Pinecone.
Each entry has a title, category, and content.
"""

KNOWLEDGE_BASE = [
    {
        "title": "What is Credit Card Fraud?",
        "category": "basics",
        "content": """
Credit card fraud is the unauthorized use of someone's credit card or credit card information 
to make purchases or withdraw funds. It is one of the most common forms of financial fraud 
and affects millions of people worldwide every year.

Credit card fraud can happen in several ways:
- Physical theft of the card
- Data breaches where card numbers are stolen from businesses
- Phishing attacks where criminals trick you into revealing your card details
- Skimming devices placed on ATMs or point-of-sale terminals
- Account takeover where criminals gain access to your online banking

The consequences of credit card fraud include financial loss, damaged credit score, 
time spent disputing charges, and emotional stress. Banks and financial institutions 
lose billions of dollars annually to credit card fraud.

Early detection is critical. The sooner fraud is detected, the less financial damage occurs.
Most banks offer zero liability protection, meaning you won't be responsible for unauthorized 
charges if you report them promptly.
"""
    },
    {
        "title": "Types of Credit Card Fraud",
        "category": "types",
        "content": """
There are several major types of credit card fraud that consumers and businesses should be aware of:

1. Card-Not-Present (CNP) Fraud
This is the most common type of fraud in the digital age. It occurs when a criminal uses 
stolen card details to make online or phone purchases without needing the physical card.
CNP fraud increased dramatically with the rise of e-commerce.

2. Card-Present Fraud
This occurs when a physical card is used fraudulently. It includes counterfeit cards made 
using skimmed data, stolen physical cards, and cards obtained through mail theft.

3. Account Takeover Fraud
Criminals gain access to an existing account by stealing login credentials through phishing, 
data breaches, or social engineering. They then change account details and make unauthorized transactions.

4. Identity Theft
Using stolen personal information to open new credit card accounts in someone else's name.
The victim may not discover this for months until they check their credit report.

5. Friendly Fraud (Chargeback Fraud)
A cardholder makes a legitimate purchase but then falsely claims the transaction was 
unauthorized to get a refund while keeping the goods or services.

6. Card Skimming
Physical devices are attached to ATMs, gas pumps, or point-of-sale terminals to steal 
card data when the card is swiped. The data is then used to create counterfeit cards.

7. Phishing and Vishing
Criminals send fake emails (phishing) or make phone calls (vishing) pretending to be 
legitimate institutions to trick people into revealing their card details.
"""
    },
    {
        "title": "How Machine Learning Detects Fraud",
        "category": "ml",
        "content": """
Machine learning has revolutionized fraud detection by enabling real-time analysis of 
thousands of transaction features simultaneously. Traditional rule-based systems required 
manual rules like "flag any transaction over $1000 abroad" — but fraudsters quickly learned 
to work around these rules.

Modern ML fraud detection works as follows:

Feature Engineering
Transaction data is transformed into meaningful features that help models distinguish 
fraud from legitimate transactions. Common features include:
- Transaction amount relative to historical average
- Time of transaction (unusual hours indicate higher risk)
- Geographic location and distance from previous transaction
- Merchant category and whether it matches spending history
- Transaction velocity (many transactions in a short time)
- Device fingerprinting for online transactions

Supervised Learning (XGBoost, LightGBM)
These models are trained on historical labeled data where each transaction is marked 
as fraud or legitimate. The model learns patterns that distinguish the two classes.
XGBoost and LightGBM use gradient boosting — building many decision trees where each 
tree corrects the errors of the previous one.

Unsupervised Learning (Autoencoder)
An autoencoder learns to reconstruct normal transaction patterns. When it encounters 
a fraudulent transaction, it cannot reconstruct it well, resulting in a high reconstruction 
error that signals potential fraud. This approach doesn't require labeled fraud data.

Class Imbalance Challenge
Fraud is extremely rare — typically only 0.1% to 0.5% of all transactions.
This creates a severe class imbalance that can cause models to simply predict everything 
as normal and achieve 99.9% accuracy while missing all fraud. Techniques like 
scale_pos_weight in XGBoost and class_weight in LightGBM address this by penalizing 
the model more heavily for missing fraud cases.
"""
    },
    {
        "title": "Understanding SHAP Values in Fraud Detection",
        "category": "xai",
        "content": """
SHAP (SHapley Additive exPlanations) is a method for explaining machine learning model 
predictions. In fraud detection, SHAP tells us exactly which transaction features 
contributed most to the fraud/normal prediction and by how much.

How SHAP Works
SHAP is based on game theory concepts. It calculates the contribution of each feature 
by comparing predictions with and without that feature across all possible feature combinations.
The result is a SHAP value for each feature that shows:
- Positive SHAP value: this feature pushed the prediction toward FRAUD
- Negative SHAP value: this feature pushed the prediction toward NORMAL
- Larger absolute value: stronger influence on the prediction

Example SHAP Explanation
For a transaction flagged as fraud:
- V14: -3.2 (SHAP: +0.45) — unusual pattern in transaction timing, strongly indicates fraud
- V10: -4.1 (SHAP: +0.38) — anomalous value in transaction characteristics
- Amount: 850 (SHAP: +0.22) — higher than normal spending amount
- V4: +2.1 (SHAP: -0.15) — this feature actually pushed toward normal

Why SHAP Matters for Fraud Detection
1. Transparency: Banks and regulators require explanations for declined transactions
2. Trust: Users can understand why their transaction was flagged
3. Debugging: Data scientists can identify model errors and biases
4. Improvement: Understanding which features matter most helps improve detection

SHAP vs LIME
Both SHAP and LIME explain individual predictions, but SHAP has stronger theoretical 
foundations (guaranteed consistency and local accuracy). LIME approximates the model 
locally with a simpler model, which can sometimes be less reliable.
"""
    },
    {
        "title": "Understanding LIME Explanations",
        "category": "xai",
        "content": """
LIME (Local Interpretable Model-agnostic Explanations) is another technique for explaining 
machine learning predictions. Unlike SHAP which is model-specific for tree models, 
LIME works with any machine learning model.

How LIME Works
LIME explains a prediction by:
1. Taking the transaction to be explained
2. Creating many slightly modified versions of that transaction (perturbations)
3. Getting the model's prediction for each modified version
4. Training a simple linear model on these perturbations
5. Using the linear model's coefficients as explanations

The result shows which features, when changed slightly, would most change the prediction.

LIME in Practice
For a fraud prediction, LIME might show:
- If Amount decreases from $850 to $100, prediction changes from FRAUD to NORMAL (high importance)
- If V14 increases from -3.2 to 0, prediction changes significantly (high importance)  
- If V22 changes, prediction barely changes (low importance)

Advantages of LIME
- Works with any model (model-agnostic)
- Intuitive explanations that non-technical users can understand
- Can handle text, images, and tabular data

Limitations of LIME
- Results can vary between runs due to random perturbation sampling
- The local linear approximation may not always be accurate
- Choosing the right neighborhood size is challenging
"""
    },
    {
        "title": "XGBoost for Fraud Detection",
        "category": "models",
        "content": """
XGBoost (Extreme Gradient Boosting) is one of the most popular machine learning algorithms 
for fraud detection due to its high accuracy, speed, and ability to handle imbalanced datasets.

How XGBoost Works
XGBoost builds an ensemble of decision trees sequentially. Each new tree focuses on 
correcting the mistakes of all previous trees. This process, called gradient boosting, 
results in a very powerful model that combines many weak learners into one strong learner.

Key Parameters for Fraud Detection
- n_estimators: number of trees (more trees = better accuracy but slower)
- max_depth: how deep each tree grows (deeper = more complex patterns)
- learning_rate: how much each tree contributes (smaller = more robust but slower)
- scale_pos_weight: handles class imbalance by weighting fraud cases more heavily
  (set to: number of normal transactions / number of fraud transactions)

Advantages for Fraud Detection
1. Handles missing values automatically
2. Built-in regularization prevents overfitting
3. Fast inference speed enables real-time fraud detection
4. Works well with the tabular transaction data typical in banking
5. scale_pos_weight directly addresses the class imbalance problem

In FraudX Analyst
Our XGBoost model achieved:
- Accuracy: 99.96%
- Precision: 88.42% (few false alarms)
- Recall: 85.71% (catches most fraud)
- F1 Score: 87.05%
- AUC-ROC: 98.46%

These results were achieved using Optuna hyperparameter tuning which searched 50 
different parameter combinations to find the optimal settings.
"""
    },
    {
        "title": "LightGBM for Fraud Detection",
        "category": "models",
        "content": """
LightGBM (Light Gradient Boosting Machine) is Microsoft's gradient boosting framework 
that is faster and more memory-efficient than XGBoost while achieving comparable or 
better accuracy. It is the best performing model in the FraudX Analyst system.

Key Differences from XGBoost
LightGBM uses leaf-wise tree growth instead of level-wise growth used by XGBoost.
This means LightGBM grows the leaf that will result in the biggest reduction in loss, 
leading to more accurate trees. However, it can overfit on small datasets — 
for large datasets like credit card transactions (284,807 records), it excels.

Why LightGBM Performs Better
- Faster training speed (13 seconds vs 38 seconds for XGBoost in our tests)
- Lower memory usage
- Supports categorical features natively
- Better accuracy on large datasets

Handling Class Imbalance
LightGBM uses class_weight parameter to address the severe imbalance in fraud data.
We set class_weight = {0: 1, 1: 577} meaning fraud cases are weighted 577 times 
more than normal cases — reflecting their rarity in the dataset.

In FraudX Analyst
Our LightGBM model achieved:
- Accuracy: 99.96%
- Precision: 90.32% (best precision of all three models)
- Recall: 85.71%
- F1 Score: 87.96% (best F1 of all three models)
- AUC-ROC: 97.07%

LightGBM is highlighted as the recommended model in the FraudX app dashboard.
"""
    },
    {
        "title": "Autoencoder for Fraud Detection",
        "category": "models",
        "content": """
An Autoencoder is a type of neural network used for unsupervised anomaly detection.
Unlike XGBoost and LightGBM which require labeled fraud/normal training data, 
an Autoencoder only needs normal transaction data to learn what normal looks like.

How It Works
1. Training phase: The autoencoder is trained only on normal (non-fraud) transactions
2. The network learns to compress and reconstruct normal transaction patterns
3. Detection phase: Any new transaction is passed through the autoencoder
4. Normal transactions are reconstructed accurately (low reconstruction error)
5. Fraudulent transactions look different from normal patterns, so reconstruction error is high
6. Transactions with reconstruction error above a threshold are flagged as fraud

Architecture in FraudX
Input (30 features) → Dense(32) → Dense(16) → Dense(8) [bottleneck] → Dense(16) → Dense(32) → Output (30 features)

The bottleneck layer (8 neurons) forces the network to learn a compressed representation 
of normal transaction patterns. Fraud cannot be compressed well by this representation.

Threshold Setting
The fraud threshold is set at the 95th or 99th percentile of reconstruction errors 
on normal transactions. Transactions above this threshold are classified as fraud.

Advantages
- Does not require labeled fraud data (useful when fraud labels are unavailable)
- Can detect novel fraud patterns not seen during training
- Provides reconstruction error as an interpretable anomaly score

Limitations
- Lower precision than supervised models (more false alarms)
- Requires careful threshold tuning
- Cannot leverage known fraud patterns the way supervised models can

In FraudX Analyst
- Accuracy: 99.82%
- Recall: 58.16% (catches about half of fraud cases)
- AUC-ROC: 94.82%
"""
    },
    {
        "title": "What to Do If You Suspect Fraud",
        "category": "action",
        "content": """
If you suspect your credit card has been used fraudulently, act immediately. 
The faster you respond, the better your chances of recovering lost funds and 
preventing further damage.

Immediate Steps (Within 24 Hours)
1. Contact your bank or card issuer immediately
   Call the number on the back of your card or use their emergency fraud line
   Most banks have 24/7 fraud reporting services
   
2. Freeze or cancel your card
   Ask your bank to immediately freeze the compromised card
   Request a replacement card with a new number

3. Review all recent transactions
   Go through your last 30-90 days of transactions carefully
   Note any unfamiliar charges, even small ones (fraudsters often test with small amounts first)

4. Change your passwords
   Change your online banking password immediately
   Update passwords for any account that uses the same email/password combination
   Enable two-factor authentication

5. File a fraud report
   File a report with your bank in writing (email or letter)
   You may also want to file a report with local police for documentation purposes
   In Malaysia, report to PDRM (Royal Malaysia Police) cybercrime unit

Follow-Up Actions (Within 1 Week)
- Check your credit report for any new accounts opened in your name
- Monitor your account daily for at least 30 days
- Keep records of all communications with your bank
- Follow up on disputed charges — banks typically have 45-90 days to investigate

Your Rights
Most countries have consumer protection laws that limit your liability for fraudulent charges.
In Malaysia, Bank Negara Malaysia (BNM) provides guidelines on consumer protection for electronic payments.
If you report unauthorized transactions promptly, you may be entitled to a full refund.
"""
    },
    {
        "title": "How to Protect Yourself from Credit Card Fraud",
        "category": "prevention",
        "content": """
Protecting yourself from credit card fraud requires awareness and good habits. 
Here are the most effective prevention strategies:

Online Security
- Only shop on websites with HTTPS (look for the padlock icon in your browser)
- Never save your card details on websites you don't fully trust
- Use virtual card numbers for online shopping if your bank offers this feature
- Use a separate card with a low limit specifically for online purchases
- Enable transaction notifications so you're alerted of every charge immediately

Physical Card Security
- Never let your card out of your sight during transactions
- Cover the keypad when entering your PIN at ATMs or shops
- Check ATMs for skimming devices — look for loose or unusual attachments
- Avoid using ATMs in poorly lit or isolated areas
- Sign the back of your card immediately when you receive it

Account Monitoring
- Review your bank statements at least weekly
- Set up real-time SMS or app notifications for all transactions
- Set transaction limits and international usage controls in your banking app
- Regularly check your credit report (free annually in most countries)

Digital Hygiene
- Never share card details over email or phone unless you initiated the call
- Be suspicious of unsolicited calls claiming to be from your bank
- Don't click links in emails asking you to verify card details
- Use strong, unique passwords for banking accounts
- Enable biometric authentication (fingerprint/face) on your banking app

Travelling
- Notify your bank before travelling internationally
- Carry more than one card from different networks
- Know your bank's international emergency number
- Avoid using public WiFi for banking transactions
"""
    },
    {
        "title": "Understanding Risk Scores in FraudX",
        "category": "app",
        "content": """
FraudX Analyst uses risk scores to quantify the likelihood that a transaction is fraudulent.
Understanding these scores helps you interpret simulation results correctly.

Risk Score Scale
- 0% - 25%: Low Risk — Transaction appears normal, highly unlikely to be fraud
- 26% - 50%: Medium Risk — Some unusual patterns detected, worth monitoring
- 51% - 75%: High Risk — Multiple suspicious indicators, transaction should be reviewed
- 76% - 100%: Very High Risk — Strong indicators of fraud, immediate action recommended

How Risk Scores Are Calculated
For XGBoost and LightGBM:
The risk score is the model's predicted probability that the transaction is fraudulent.
For example, a risk score of 87% means the model believes there is an 87% chance this 
transaction is fraud based on the pattern of features.

For the Autoencoder:
The risk score is derived from the reconstruction error — how differently the transaction 
looks compared to normal transaction patterns. Higher reconstruction error = higher risk score.

Confidence Score
The confidence score represents how certain the model is about its prediction.
- High confidence + FRAUD prediction = strong fraud signal, take action
- Low confidence + FRAUD prediction = uncertain, may want to verify manually
- High confidence + NORMAL prediction = transaction is clearly legitimate

Interpreting Results
Always consider both the prediction AND the risk score together:
- FRAUD at 95% risk: Very clear fraud signal, contact your bank immediately
- FRAUD at 55% risk: Borderline case, worth investigating further
- NORMAL at 2% risk: Clearly legitimate transaction
- NORMAL at 45% risk: Unusual transaction that passed the threshold, keep monitoring
"""
    },
    {
        "title": "The Kaggle Credit Card Fraud Dataset",
        "category": "dataset",
        "content": """
FraudX Analyst was trained on the widely-used Kaggle Credit Card Fraud Detection dataset,
which is a benchmark dataset in the fraud detection research community.

Dataset Overview
- Source: Real anonymized credit card transactions from European cardholders in September 2013
- Total transactions: 284,807
- Fraudulent transactions: 492 (0.172%)
- Normal transactions: 284,315 (99.828%)
- Time period: 2 days of transactions

Features
The dataset contains 30 features:
- Time: seconds elapsed between each transaction and the first transaction
- Amount: transaction amount in Euros
- V1-V28: PCA-transformed features (original features are confidential for privacy reasons)

The V1-V28 features are the result of Principal Component Analysis (PCA) transformation 
applied to the original transaction features. PCA was applied to protect confidentiality 
of the original data. While we don't know exactly what each V feature represents, 
the ML models and SHAP analysis can still identify which ones are most predictive of fraud.

Class Imbalance
The extreme class imbalance (0.17% fraud) is a key challenge. This is realistic — 
in real banking systems, fraud rates are typically between 0.1% and 0.5% of all transactions.
FraudX addresses this using scale_pos_weight (XGBoost) and class_weight (LightGBM).

Evaluation Metrics
Due to the class imbalance, accuracy alone is misleading (a model predicting everything 
as normal achieves 99.8% accuracy). Better metrics for fraud detection are:
- Precision: Of all transactions flagged as fraud, how many were actually fraud?
- Recall: Of all actual fraud transactions, how many did we catch?
- F1 Score: Harmonic mean of precision and recall
- AUC-ROC: Overall ability to discriminate between fraud and normal
"""
    },
    {
        "title": "Data set used for training the models",
        "category": "dataset",
        "content": """
The models in FraudX Analyst were trained on the Kaggle Credit Card Fraud Detection dataset,
which contains 284,807 transactions made by European cardholders in September 2013.
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. 
The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality
issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the
principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 
'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the 
transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable 
and it takes value 1 in case of fraud and 0 otherwise.

Given the class imbalance ratio, we recommend measuring the accuracy using the Area Under the Precision-Recall Curve (AUPRC). Confusion
matrix accuracy is not meaningful for unbalanced classification.
"""
    },
    {
        "title": "How Evaluation Metrics Are Calculated",
        "category": "metrics",
        "content": """
Evaluation metrics are mathematical formulas used to measure how well a fraud detection model
performs. In FraudX Analyst, we use five core metrics. Each is calculated from the confusion
matrix, which contains four values: True Positives (TP), True Negatives (TN), False Positives (FP),
and False Negatives (FN).

Accuracy
Accuracy = (TP + TN) / (TP + TN + FP + FN)
It measures the overall proportion of correct predictions out of all predictions. In fraud
detection, accuracy can be misleading because the dataset is highly imbalanced (99.83% normal,
0.17% fraud). A model that predicts everything as normal would still achieve 99.83% accuracy
while catching zero fraud. This is why we rely on additional metrics.

Reference: Hossin, M. and Sulaiman, M.N. (2015). A Review on Evaluation Metrics for Data
Classification Evaluations. International Journal of Data Mining and Knowledge Management
Process, 5(2), pp.1-11.

Precision
Precision = TP / (TP + FP)
Precision answers: "Of all transactions the model flagged as fraud, how many were actually fraud?"
High precision means fewer false alarms. In banking, low precision leads to too many legitimate
transactions being blocked, frustrating customers.

Recall (Sensitivity / True Positive Rate)
Recall = TP / (TP + FN)
Recall answers: "Of all actual fraud transactions, how many did the model catch?"
High recall means fewer missed fraud cases. In fraud detection, recall is often prioritized
because missing a fraud case (false negative) causes direct financial loss.

F1 Score
F1 = 2 * (Precision * Recall) / (Precision + Recall)
The F1 Score is the harmonic mean of precision and recall. It provides a single balanced metric
that accounts for both false positives and false negatives. F1 ranges from 0 (worst) to 1 (best).
We use F1 Score as the primary optimization metric in FraudX because it balances the tradeoff
between catching fraud (recall) and avoiding false alarms (precision).

Reference: Powers, D.M.W. (2011). Evaluation: From Precision, Recall and F-Measure to ROC,
Informedness, Markedness and Correlation. Journal of Machine Learning Technologies, 2(1), pp.37-63.

AUC-ROC (Area Under the Receiver Operating Characteristic Curve)
AUC-ROC measures the model's ability to discriminate between fraud and normal transactions across
all possible classification thresholds. It plots the True Positive Rate (recall) against the
False Positive Rate (FPR = FP / (FP + TN)) at various thresholds.
AUC = 1.0 means perfect discrimination; AUC = 0.5 means no better than random guessing.
AUC-ROC is threshold-independent, making it useful for comparing models overall.

Reference: Fawcett, T. (2006). An Introduction to ROC Analysis. Pattern Recognition Letters,
27(8), pp.861-874.

PR-AUC (Area Under the Precision-Recall Curve)
PR-AUC is especially suitable for imbalanced datasets like ours. It plots precision against recall
at various thresholds. Unlike ROC-AUC, PR-AUC focuses on the positive (fraud) class performance
and is not inflated by the large number of true negatives. FraudX reports PR-AUC alongside
AUC-ROC for a complete picture.

Reference: Saito, T. and Rehmsmeier, M. (2015). The Precision-Recall Plot Is More Informative
than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. PLoS ONE, 10(3),
e0118432.
"""
    },
    {
        "title": "How the XGBoost Model Was Built in FraudX",
        "category": "model_building",
        "content": """
XGBoost (Extreme Gradient Boosting) is a supervised learning algorithm used in FraudX Analyst
for binary classification of transactions as fraud or normal.

Algorithm Overview
XGBoost is an ensemble method that builds many decision trees sequentially. Each new tree
corrects the errors of the previous ones using gradient descent optimization. The final
prediction is the weighted sum of all tree predictions. XGBoost uses second-order gradient
approximations (Newton-Raphson) for faster convergence compared to standard gradient boosting.

Reference: Chen, T. and Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System.
Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data
Mining, pp.785-794.

Data Preparation
The Kaggle Credit Card Fraud dataset (284,807 transactions, 492 fraud) was split into:
- Training set: 70% (used to train the model)
- Validation set: 15% (used for hyperparameter tuning with Optuna)
- Test set: 15% (used ONCE for final evaluation, never seen during training)
Scalers were fit ONLY on the training set and then applied to validation and test sets
to prevent data leakage.

Handling Class Imbalance
XGBoost uses the scale_pos_weight parameter, calculated as the ratio of normal to fraud
samples in the training set (approximately 578:1). This makes the model penalize missed
fraud cases much more heavily than missed normal cases.

Hyperparameter Tuning
Optuna (a Bayesian optimization framework) was used to search for optimal hyperparameters
over 50 trials, evaluating on the validation set only. Key hyperparameters tuned:
- n_estimators: number of boosting rounds (100-500)
- max_depth: maximum tree depth (3-9)
- learning_rate: step size shrinkage (0.01-0.3, log scale)
- subsample: fraction of training data per tree (0.6-1.0)
- colsample_bytree: fraction of features per tree (0.6-1.0)
- min_child_weight: minimum sum of instance weight in a child (1-10)
- gamma: minimum loss reduction for a split (0-5)
- reg_alpha: L1 regularization (0-2)
- reg_lambda: L2 regularization (0-2)

Reference: Akiba, T. et al. (2019). Optuna: A Next-Generation Hyperparameter Optimization
Framework. Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery
and Data Mining, pp.2623-2631.

Explainability
After training, SHAP (SHapley Additive exPlanations) TreeExplainer was applied to compute
feature importance values for individual predictions. This enables FraudX to explain why
each transaction was flagged or cleared.

Reference: Lundberg, S.M. and Lee, S.I. (2017). A Unified Approach to Interpreting Model
Predictions. Advances in Neural Information Processing Systems 30 (NeurIPS), pp.4765-4774.
"""
    },
    {
        "title": "How the LightGBM Model Was Built in FraudX",
        "category": "model_building",
        "content": """
LightGBM (Light Gradient Boosting Machine) is a supervised learning algorithm used in FraudX
Analyst as the second gradient boosting model alongside XGBoost.

Algorithm Overview
LightGBM is a gradient boosting framework developed by Microsoft that uses two novel techniques:
Gradient-based One-Side Sampling (GOSS) which focuses on training instances with larger gradients,
and Exclusive Feature Bundling (EFB) which bundles mutually exclusive features together.
These make LightGBM significantly faster than traditional gradient boosting while maintaining
accuracy. LightGBM also grows trees leaf-wise (best-first) rather than level-wise, which
often produces deeper but more accurate trees.

Reference: Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree.
Advances in Neural Information Processing Systems 30 (NeurIPS), pp.3146-3154.

Data Preparation
Same as XGBoost: 70/15/15 train/validation/test split with scalers fit only on training data.
No data leakage.

Handling Class Imbalance
LightGBM uses class_weight parameter set to 'balanced', which automatically adjusts weights
inversely proportional to class frequencies. This achieves a similar effect to XGBoost's
scale_pos_weight but is computed automatically by scikit-learn's LGBMClassifier wrapper.

Hyperparameter Tuning
Optuna was used with 50 trials on the validation set. Key hyperparameters tuned:
- n_estimators: number of boosting iterations (100-500)
- max_depth: maximum tree depth (3-9)
- learning_rate: step size shrinkage (0.01-0.3, log scale)
- subsample: row sampling rate (0.6-1.0)
- colsample_bytree: feature sampling rate (0.6-1.0)
- min_child_samples: minimum data in a leaf (5-50)
- num_leaves: maximum number of leaves per tree (20-100)
- reg_alpha: L1 regularization (0-2)
- reg_lambda: L2 regularization (0-2)

LightGBM vs XGBoost
Both are gradient boosting algorithms but differ in tree construction:
- XGBoost grows level-wise (all nodes at same depth first)
- LightGBM grows leaf-wise (splits the leaf with maximum delta loss)
Leaf-wise growth can overfit on small datasets but often achieves better results on larger
datasets. LightGBM is also typically 2-5x faster to train.

Explainability
SHAP TreeExplainer is also applied to LightGBM for per-prediction feature importance,
using the same methodology as XGBoost.
"""
    },
    {
        "title": "How the Autoencoder Model Was Built in FraudX",
        "category": "model_building",
        "content": """
The Autoencoder is an unsupervised deep learning model used in FraudX Analyst as an
anomaly detection approach to fraud detection.

Algorithm Overview
An autoencoder is a neural network trained to reconstruct its input. It consists of:
- Encoder: compresses input into a lower-dimensional representation (bottleneck)
- Decoder: reconstructs the original input from the compressed representation
When trained only on normal transactions, the autoencoder learns to reconstruct normal
patterns well. Fraudulent transactions, which differ from normal patterns, produce high
reconstruction errors, indicating anomalies.

Reference: Hinton, G.E. and Salakhutdinov, R.R. (2006). Reducing the Dimensionality of Data
with Neural Networks. Science, 313(5786), pp.504-507.

Network Architecture in FraudX
Encoder: input_dim -> 32 neurons (ReLU) -> Dropout(0.2) -> 16 neurons (ReLU) -> 8 neurons (ReLU, bottleneck)
Decoder: 8 neurons -> 16 neurons (ReLU) -> Dropout(0.2) -> 32 neurons (ReLU) -> input_dim (linear activation)
Optimizer: Adam
Loss function: Mean Squared Error (MSE)

The bottleneck layer (8 neurons) forces the network to learn a compressed representation of
the 30 input features, capturing only the most essential patterns of normal transactions.

Data Preparation
Same 70/15/15 split as other models. Critically, the autoencoder is trained ONLY on normal
transactions from the training set. This ensures it learns exclusively what normal looks like.
The validation set (containing both normal and fraud) is used for threshold optimization.

Reconstruction Error Calculation
FraudX uses a weighted combination of three error metrics:
- Mean Squared Error (MSE): 50% weight - measures average squared difference
- Mean Absolute Error (MAE): 30% weight - measures average absolute difference
- Maximum Error: 20% weight - captures the worst feature reconstruction
Combined Error = 0.5 * MSE + 0.3 * MAE + 0.2 * Max_Error

Threshold Optimization
The fraud/normal threshold is determined by testing 200 evenly spaced values between the
minimum and maximum reconstruction errors on the validation set. The threshold that maximizes
F1 score is selected, balancing precision and recall.

Reference: An, J. and Cho, S. (2015). Variational Autoencoder Based Anomaly Detection Using
Reconstruction Probability. Special Lecture on IE, 2(1), pp.1-18.

Autoencoder vs Supervised Models
Advantages: Does not require labeled fraud data for training, can detect novel fraud patterns
not seen in training data.
Disadvantages: Generally lower precision and recall than supervised models because it only
learns what is normal, not what specifically looks like fraud. In FraudX, the autoencoder
achieves high accuracy but lower F1 compared to XGBoost and LightGBM.
"""
    },
    {
        "title": "Confusion Matrix and Its Role in Fraud Detection",
        "category": "metrics",
        "content": """
A confusion matrix is a table that summarizes how a classification model's predictions compare
to the actual outcomes. For binary fraud detection, it has four components:

True Positive (TP): Model correctly predicted FRAUD (actual was fraud)
True Negative (TN): Model correctly predicted NORMAL (actual was normal)
False Positive (FP): Model incorrectly predicted FRAUD (actual was normal) - a false alarm
False Negative (FN): Model incorrectly predicted NORMAL (actual was fraud) - a missed fraud

In FraudX Analyst's context:
- TP: A fraudulent transaction was correctly caught. This protects the cardholder.
- TN: A normal transaction was correctly allowed. The cardholder's purchase goes through.
- FP: A normal transaction was wrongly flagged as fraud. This inconveniences the cardholder
  and may block a legitimate purchase.
- FN: A fraudulent transaction was missed. The cardholder suffers financial loss.

Why FN is the Most Dangerous
In fraud detection, false negatives (missed fraud) are generally more costly than false
positives (false alarms). A missed fraud case means real money is stolen. A false alarm
merely requires a phone call to verify. This is why FraudX uses scale_pos_weight and
class_weight to penalize false negatives more heavily.

All five evaluation metrics (Accuracy, Precision, Recall, F1, AUC-ROC) are derived from
or related to the confusion matrix values.

Reference: Stehman, S.V. (1997). Selecting and Interpreting Measures of Thematic
Classification Accuracy. Remote Sensing of Environment, 62(1), pp.77-89.
"""
    },
    {
        "title": "Data Preprocessing and Leakage Prevention in FraudX",
        "category": "model_building",
        "content": """
Data preprocessing is a critical step in building reliable fraud detection models. FraudX
follows strict protocols to prevent data leakage, which would produce artificially inflated
performance metrics that do not reflect real-world performance.

Data Split Strategy
The dataset (284,807 transactions) is split into three subsets:
- Training set: 70% - used to fit the model parameters
- Validation set: 15% - used for hyperparameter tuning (Optuna) and threshold optimization
- Test set: 15% - used ONCE for final evaluation, never seen during any training or tuning

Feature Scaling
Standard scaling (z-score normalization) is applied to the Amount and Time features:
z = (x - mean) / standard_deviation
CRITICAL: The scaler is fit ONLY on the training set. The same fitted scaler is then applied
to the validation and test sets. This prevents information from the validation/test sets
from leaking into the training process.

Reference: Kaufman, S. et al. (2012). Leakage in Data Mining: Formulation, Detection, and
Avoidance. ACM Transactions on Knowledge Discovery from Data, 6(4), pp.1-21.

PCA Features (V1-V28)
The original dataset already has 28 features (V1-V28) that were PCA-transformed by the
dataset creators to protect cardholder privacy. These features are already scaled and do
not require additional preprocessing.

Why Leakage Prevention Matters
If a scaler is fit on the entire dataset (including test data), the model indirectly gains
information about the test set's distribution. This leads to overly optimistic metrics during
evaluation. FraudX's strict train-only scaling ensures that reported metrics accurately
reflect how the model would perform on completely unseen data.
"""
    },
    {
        "title": "Optuna Hyperparameter Optimization in FraudX",
        "category": "model_building",
        "content": """
Optuna is the hyperparameter optimization framework used in FraudX Analyst to find the best
configuration for XGBoost and LightGBM models.

What Is Hyperparameter Tuning?
Machine learning models have two types of parameters:
- Model parameters: learned from data during training (e.g., tree split values)
- Hyperparameters: set before training and control the learning process (e.g., learning_rate,
  max_depth, n_estimators)
Choosing good hyperparameters significantly impacts model performance.

How Optuna Works
Optuna uses Bayesian optimization with a Tree-structured Parzen Estimator (TPE) to
intelligently search the hyperparameter space. Unlike grid search (which tests every combination)
or random search, Optuna learns from previous trials to focus on promising regions of the
search space.

FraudX Optuna Configuration
- Number of trials: 50 per model
- Optimization target: maximize F1 score on the validation set
- Evaluation set: validation set ONLY (test set is never used during tuning)
- Direction: maximize (higher F1 is better)

After 50 trials, the best hyperparameter combination is used to train the final model on
the training set. The final model is then evaluated once on the test set.

Reference: Akiba, T. et al. (2019). Optuna: A Next-Generation Hyperparameter Optimization
Framework. Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery
and Data Mining, pp.2623-2631.
"""
    },
    {
        "title": "MLflow Experiment Tracking in FraudX",
        "category": "model_building",
        "content": """
MLflow is the experiment tracking platform used in FraudX Analyst to log, compare, and
manage machine learning model runs.

What MLflow Tracks in FraudX
For each model training run, MLflow records:
- Parameters: all hyperparameters (learning_rate, max_depth, n_estimators, etc.)
- Metrics: accuracy, precision, recall, F1 score, AUC-ROC, PR-AUC, training time
- Artifacts: trained model files, SHAP plots, evaluation plots (confusion matrix, ROC curve,
  precision-recall curve, feature importance)
- Metadata: run ID, timestamp, experiment name

FraudX Experiment Name
All models are logged under the experiment name "FraudX-Models-Fixed", indicating these are
the versions trained with proper data leakage prevention.

Why MLflow Matters
MLflow enables reproducibility and comparison across model versions. If a model is retrained
with different hyperparameters or a new dataset, all results are tracked and can be compared
side-by-side. This is essential for a production fraud detection system where model performance
must be monitored over time.

Reference: Zaharia, M. et al. (2018). Accelerating the Machine Learning Lifecycle with MLflow.
IEEE Data Engineering Bulletin, 41(4), pp.39-45.
"""
    },
]
