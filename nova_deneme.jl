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

