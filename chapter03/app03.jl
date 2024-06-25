include("mod03.jl")
using .CHAPTER03
using Random, Plots, DataFrames, StatsBase
using MLJ

# Load data
iris = MLJ.load_iris()
MLJ.selectrows(iris, 1:3) |> MLJ.pretty
MLJ.schema(iris)
iris = DataFrames.DataFrame(iris)
y, X = MLJ.unpack(iris, ==(:target))
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

doc("KernelPerceptronClassifier", pkg="BetaML")
Model = @load KernelPerceptronClassifier pkg=BetaML
pcl = Model()
mach = machine(pcl, Xtrn_std, ytrn);
MLJ.fit!(mach)

ŷ = MLJ.predict(mach, Xtst_std)

print("Misclassified examples: $(sum(ytst .!= mode.(ŷ)))")
MLJ.accuracy(ytst, MLJ.mode.(ŷ))
evaluate!(mach, resampling=Holdout(fraction_train=0.7),
         measures=[log_loss, accuracy],
         verbosity=0)

X = vcat(Xtrn_std, Xtst_std)
y = vcat(ytrn, ytst)
plot_decision_regions(X, y, mach, test_idx=105:150, length=500)

# Logistic Regression

function sigmoid(z)
    return 1.0 ./ (1.0 .+ exp.(-z))
end

z = range(-7, 7, step=0.1)
sigma_z = sigmoid(z)

begin
    plot(z, sigma_z, label="σ(z)", legend=false)
    vline!([0.0], color=:black)
    ylims!(-0.1, 1.1)
    xlabel!("z")
    ylabel!("σ(z)")
    yticks!([0.0, 0.5, 1.0])
    plot!(grid=true, gridalpha=0.5, gridstyle=:dash, minorgrid=true)
    plot!(size=(600, 400), margin=3Plots.mm)        
end

loss_y(z, y) = y==1 ? -log(sigmoid(z)) : -log(1-sigmoid(z))

z = -10:0.1:10;
sigma_z = sigmoid(z)
c0 = [loss_y(x, 0) for x in z]
c1 = [loss_y(x, 1) for x in z]

plot(sigma_z, [c0, c1],
     label=["L(w,b) if y=1" "L(w,b) if y=0"],
     ylims=(0.0, 5.1),
     legend=:top,
     linestyle=[:solid :dash],
     linewidth=2)

idx = (ytrn .== "setosa") .|| (ytrn .== "versicolor")
Xtrn01_subset = Xtrn_std[idx, :] |> Matrix
ytrn01_subset = ytrn[idx]
ytrn01_subset = [x == "setosa" ? 0 : 1 for x in ytrn01_subset]

lrgd = LogisticRegression()
fitmodel!(lrgd, Xtrn01_subset, ytrn01_subset,
          η=0.3, num_iter=1000, seed=1)

plot_decision_regions(Xtrn01_subset, ytrn01_subset, lrgd)

# Logistic Regression with MLJ
models("Logistic")
MNModel = @load MultinomialClassifier pkg=MLJLinearModels
doc("MultinomialClassifier", pkg="MLJLinearModels")
solver = MLJLinearModels.LBFGS()
lr = MNModel(solver=solver)
mach = machine(lr, Xtrn_std, ytrn)
fit!(mach)

Xcomb_std = vcat(Xtrn_std, Xtst_std)
ycomb = vcat(ytrn, ytst)
plot_decision_regions(Xcomb_std, ycomb, mach; test_idx=105:150)