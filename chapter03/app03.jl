using Random, Plots, DataFrames, StatsBase
using MLJ

# Load data
iris = load_iris()
selectrows(iris, 1:3) |> pretty
schema(iris)
iris = DataFrames.DataFrame(iris)
y, X = unpack(iris, ==(:target), rng=123)
X = X[:, 3:4]
first(X, 3) |> pretty

Random.seed!(1)
train, test = partition(eachindex(y), 0.7, stratify=y)
Xtrn, Xtst = X[train, :], X[test, :]
ytrn, ytst = y[train], y[test]

println("Labels count in y: ", values(countmap(y)))
println("Labels counts in ytrn: ", values(countmap(ytrn)))
println("Labels counts in ytst: ", values(countmap(ytst)))

# Define the StandardScaler equivalent
standardizer = Standardizer()
# Fit the standardizer to the training data
mach = machine(standardizer, Xtrn)
fit!(mach)
# Transform both training and test data
Xtrn_std = MLJ.transform(mach, Xtrn)
Xtst_std = MLJ.transform(mach, Xtst)

# Search for Perceptron
models(matching(X, y))
models("Perceptron")

doc("PerceptronClassifier", pkg="BetaML")
Model = @load PerceptronClassifier pkg=BetaML
ppn = Model()
mach = machine(ppn, Xtrn_std, ytrn);
fit!(mach)

X = vcat(Xtrn_std, Xtst_std)
y = vcat(ytrn, ytst)

ŷ = MLJ.predict(mach, X)

print("Misclassified examples: $(sum(y .!= mode.(ŷ)))")
accuracy(y, mode.(ŷ))
evaluate!(mach, resampling=Holdout(fraction_train=0.7),
         measures=[log_loss, accuracy],
         verbosity=0)

markers = [:circle, :rect, :utriangle, :dtriangle, :diamond]
colors = [:red, :blue, :lightgreen, :gray, :cyan]
test_idx = 105:150
resolution = 0.02
classifier = mach

# Plot the decision surface
x1_min, x1_max = minimum(X[:, 1]) - 1, maximum(X[:, 1]) + 1
x2_min, x2_max = minimum(X[:, 2]) - 1, maximum(X[:, 2]) + 1
xx1 = range(x1_min, x1_max, length=300)
xx2 = range(x2_min, x2_max, length=300)

x1 = minimum(xx1)
x2 = minimum(xx2)
predict_mode(mach, [x1 x2])[1]

Z = [predict_mode(classifier, [x1 x2])[1] for x1 in xx1, x2 in xx2]

p = contourf(xx1, xx2, Z, color=:RdYlBu, alpha=0.3, legend=false)

