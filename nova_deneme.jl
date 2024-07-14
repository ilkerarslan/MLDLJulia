using Revise
using RDatasets, DataFrames
using Statistics, LinearAlgebra

iris = dataset("datasets", "iris")
X = iris[51:150, 1:4] |> Matrix
y = [(s == "versicolor") ? 0 : 1 for s ∈ iris[51:150, 5]]

using NovaML.ModelSelection: train_test_split
Xtrn, Xtst, ytrn, ytst = train_test_split(X, y, test_size=0.2)

# Scale features
using NovaML.PreProcessing: StandardScaler
scaler = StandardScaler()
scaler.fitted # false

# fit and transform
Xtrnstd = scaler(Xtrn) 
# transform with the fitted model
Xtststd = scaler(Xtst)

# Train a model
using NovaML.LinearModel: LogisticRegression
model = LogisticRegression(η=0.1, num_iter=100)

# fit the model
model(Xtrnstd, ytrn)

# Make predictions
ŷtrn = model(Xtrnstd)
ŷtst = model(Xtststd)

# Evaluate the model
using NovaML.Metrics: accuracy_score

acc_trn = accuracy_score(ytrn, ŷtrn);
acc_tst = accuracy_score(ytst, ŷtst);

println("Training accuracy: $acc_trn")
println("Test accuracy: $acc_tst")

# Data
using RDatasets, DataFrames
iris = dataset("datasets", "iris")
X = iris[:, 1:4] |> Matrix
y = iris.Species
map_species = Dict(
    "setosa" => 0,
    "versicolor" => 1,
    "virginica" => 2
)
y = [map_species[k] for k in y]

using NovaML.ModelSelection: train_test_split
Xtrn, Xtst, ytrn, ytst = train_test_split(X, y, test_size=0.2, random_state=1)

# Assuming X and y are your multiclass data
using NovaML.MultiClass: OneVsRestClassifier
lr = LogisticRegression()
ovr = OneVsRestClassifier(lr)

# fit the model
ovr(Xtrn, ytrn)

# Make predictions
ŷtrn = ovr(Xtrn)
ŷtst = ovr(Xtst)

using NovaML.Metrics: accuracy_score
accuracy_score(ytrn, ŷtrn)
accuracy_score(ytst, ŷtst)

using NovaML.Ensemble: RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, max_depth=5)
rf(Xtrn, ytrn)

ŷ = rf(Xtst)

using NovaML.Decomposition: PCA

pca = PCA(n_components=2)

# fit
pca(X)

# transform if fitted / fit & transform if not 
Xpca = pca(X)

# Inverse transform
Xorig = pca(Xpca, :inverse_transform)


sc = StandardScaler()
pca = PCA(n_components=2)
lr = LogisticRegression()

# transform the data and fit the model 
Xtrn |> sc |> pca |> X -> lr(X, ytrn)

# make predictions
ŷtst = Xtst |> sc |> pca |> lr

using NovaML.LinearModel: LinearRegression
using MLDatasets
boston = BostonHousing()
X = boston.features |> Matrix
y = boston.targets.MEDV

Xtrn, Xtst, ytrn, ytst = train_test_split(X, y, test_size=0.2)

linreg = LinearRegression()
linreg(Xtrn, ytrn)

ŷ = linreg(Xtst)

linreg.w
linreg.b

using NovaML.Metrics: r2_score, mse, adj_r2_score

r2_score(ytst, ŷ)
adj_r2_score(ytst, ŷ, n_features=size(Xtrn, 2))

(ytst .- ŷ).^2 |> mean
mse(ytst, ŷ)


using RDatasets, DataFrames

iris = dataset("datasets", "iris")
X = iris[:, 1:4] |> Matrix
y = iris.Species
map_species = Dict(
    "setosa" => 0,
    "versicolor" => 1,
    "virginica" => 2
)
y = [map_species[k] for k in y]

using NovaML.ModelSelection: train_test_split
Xtrn, Xtst, ytrn, ytst = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

println("Shape of Xtrn: ", size(Xtrn))
println("Shape of ytrn: ", size(ytrn))
println("Shape of Xtst: ", size(Xtst))
println("Shape of ytst: ", size(ytst))

using NovaML.PreProcessing: StandardScaler
stdscaler = StandardScaler()
Xtrnstd = stdscaler(Xtrn)
Xtststd = stdscaler(Xtst)

println("Shape of Xtrnstd: ", size(Xtrnstd))
println("Shape of Xtststd: ", size(Xtststd))

using NovaML.SVM: SVC
svm = SVC(kernel="linear", C=1.0, random_state=1, max_iter=1000, tol=1e-3)
svm(Xtrnstd, ytrn)
println("Shape of Xtrnstd: ", size(Xtrnstd))
println("Shape of ytrn: ", size(ytrn))
println("Unique values in ytrn: ", unique(ytrn))
ŷtst = svm(Xtststd) 
using NovaML.Metrics: accuracy_score
accuracy_score(ŷtst, ytst)

println("Shape of ŷtst: ", size(ŷtst))
println("Unique values in ŷtst: ", unique(ŷtst))

svm_rbf = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=1) 
svm_rbf(Xtrnstd, ytrn)

svm_poly = SVC(kernel="poly", C=1.0, degree=3, gamma="scale", random_state=1)
svm_poly(Xtrnstd, ytrn)



using NovaML.LinearModel: LogisticRegression
using NovaML.MultiClass: OneVsRestClassifier

lr = LogisticRegression()
ovr = OneVsRestClassifier(lr)

ovr(Xtrnstd, ytrn)
ŷovrtst = ovr(Xtststd)
accuracy_score(ŷovrtst, ytst)