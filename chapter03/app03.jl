using Random, Plots, DataFrames, StatsBase
using MLJ

# Load data
iris = MLJ.load_iris()
MLJ.selectrows(iris, 1:3) |> MLJ.pretty
MLJ.schema(iris)
iris = DataFrames.DataFrame(iris)
y, X = MLJ.unpack(iris, ==(:target), rng=123)
X = X[:, 3:4]
MLJ.first(X, 3) |> MLJ.pretty

Random.seed!(1)
train, test = MLJ.partition(eachindex(y), 0.7, stratify=y)
Xtrn, Xtst = X[train, :], X[test, :]
ytrn, ytst = y[train], y[test]

println("Labels count in y: ", values(StatsBase.countmap(y)))
println("Labels counts in ytrn: ", values(StatsBase.countmap(ytrn)))
println("Labels counts in ytst: ", values(StatsBase.countmap(ytst)))

# Define the StandardScaler equivalent
standardizer = MLJ.Standardizer()
# Fit the standardizer to the training data
mach = MLJ.machine(standardizer, Xtrn)
MLJ.fit!(mach)
# Transform both training and test data
Xtrn_std = MLJ.transform(mach, Xtrn)
Xtst_std = MLJ.transform(mach, Xtst)

# Search for Perceptron
models(matching(X, y))
models("Perceptron")

doc("PerceptronClassifier", pkg="BetaML")
Model = @load PerceptronClassifier pkg=BetaML
pcl = Model()
mach = machine(pcl, Xtrn_std, ytrn);
fit!(mach)

X = vcat(Xtrn_std, Xtst_std)
y = vcat(ytrn, ytst)

ŷ = MLJ.predict(mach, X)

print("Misclassified examples: $(sum(y .!= mode.(ŷ)))")
MLJ.accuracy(y, MLJ.mode.(ŷ))
evaluate!(mach, resampling=Holdout(fraction_train=0.7),
         measures=[log_loss, accuracy],
         verbosity=0)

resolution = 0.02
classifier = mach

# Plot the decision surface
x1_min, x1_max = minimum(X[:, 1]) - 1, maximum(X[:, 1]) + 1
x2_min, x2_max = minimum(X[:, 2]) - 1, maximum(X[:, 2]) + 1
xx1 = range(x1_min, x1_max, length=300)
xx2 = range(x2_min, x2_max, length=300)

Z = [predict_mode(mach, [x1 x2])[1] for x1 in xx1, x2 in xx2]
p = contourf(xx1, xx2, Z, color=:RdYlBu, alpha=0.3, legend=false)

#######################################################################
using Plots
test_idx = 105:150
markers = [:circle, :rect, :utriangle, :dtriangle, :diamond]
colors = [:red, :blue, :lightgreen, :gray, :cyan]

# Plot the decision surface
x1_min, x1_max = minimum(X[:, 1]) - 1, maximum(X[:, 1]) + 1
x2_min, x2_max = minimum(X[:, 2]) - 1, maximum(X[:, 2]) + 1
xx1 = range(x1_min, x1_max, length=300)
xx2 = range(x2_min, x2_max, length=300)

Z = [(predict_mode(mach, reshape([x1, x2], 1, 2))[1]) for x1 in xx1, x2 in xx2]

color_map = Dict("setosa" => 1, "versicolor" => 2, "virginica" => 3)
Z_numeric = [color_map[z] for z in Z]

p = contourf(xx1, xx2, Z_numeric, color=[:red, :blue, :lightgreen], levels=3, alpha=0.3, legend=false);

# Plot data points
for (i, cl) in enumerate(unique(y))
    idx = findall(y .== cl)
    scatter!(p, X[idx, 1], X[idx, 2], color=colors[i], marker=markers[i], label="Class $cl", ms=4)
end

# Highlight test examples
if !isempty(test_idx)
    X_test, y_test = X[test_idx, :], y[test_idx]
    scatter!(p, X_test[:, 1], X_test[:, 2],
             marker=:circle,
             mc=:black, ms=5,
             label="Test set",
             markersize=6)
end

xlabel!("Petal length (standardized)")
ylabel!("Petal width (standardized)")
title!("Decision Regions with Perceptron")
plot!(legend=:topleft)