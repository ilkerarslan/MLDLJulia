using NovaML.Datasets
X, y = load_iris(return_X_y=true)

using NovaML.ModelSelection
Xtrn, Xtst, ytrn, ytst = train_test_split(X, y, test_size=0.2)

using NovaML.ModelSelection
Xtrn, Xtst, ytrn, ytst = train_test_split(X, y, test_size=0.2)

# Scale features
using NovaML.PreProcessing
scaler = StandardScaler()
scaler.fitted # false

# Fit and transform
Xtrnstd = scaler(Xtrn) 
# transform with the fitted model
Xtststd = scaler(Xtst)

# Train a model
using NovaML.LinearModel
lr = LogisticRegression(η=0.1, num_iter=100)

using NovaML.MultiClass: OneVsRestClassifier
ovr = OneVsRestClassifier(lr)

# Fit the model
ovr(Xtrnstd, ytrn)

# Make predictions
ŷtrn = ovr(Xtrnstd)
ŷtst = ovr(Xtststd)

# Evaluate the model
using NovaML.Metrics
acc_trn = accuracy_score(ytrn, ŷtrn);
acc_tst = accuracy_score(ytst, ŷtst);

println("Training accuracy: $acc_trn")
println("Test accuracy: $acc_tst")

data = load_boston()

X, y = load_boston(return_X_y=true)

