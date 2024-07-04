using Revise
using RDatasets, DataFrames, StatsBase
using Random, Plots, StatsBase

using Nova
using Nova.ModelSelection: train_test_split
using Nova.PreProcessing: StandardScaler
using Nova.LinearModel: Perceptron, MultiClassPerceptron
using Nova.Metrics: accuracy_score

# Data
iris = dataset("datasets", "iris")
X = iris[:, 3:4] |> Matrix
y = iris.Species
map_species = Dict(
    "setosa" => 0,
    "versicolor" => 1,
    "virginica" => 2
)
y = [map_species[k] for k in y]
print("Class labels: ", unique(y))

Xtrn, Xtst, ytrn, ytst = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

println("Labels count in y: ", values(StatsBase.countmap(y)))
println("Labels counts in ytrn: ", values(StatsBase.countmap(ytrn)))
println("Labels counts in ytst: ", values(StatsBase.countmap(ytst)))

stdscaler = StandardScaler()

stdscaler(Xtrn)
Xtrn_std = stdscaler(Xtrn)
Xtst_std = stdscaler(Xtst)


abstract type AbstractModel end

# Activation functions
"""Linear activation function"""
linearactivation(X) = X

# Common methods
"""Calculate net input"""
net_input(m::AbstractModel, x::AbstractVector) = m.W' * x .+ m.b
net_input(m::AbstractModel, X::AbstractMatrix) = X * m.W .+ m.b'

"""AbstractModel batch prediction"""
(m::AbstractModel)(X::AbstractMatrix) = [m(x) for x in eachrow(X)]

# Multiclass Perceptron
mutable struct MulticlassPerceptron <: AbstractModel
    # Parameters
    W::Matrix{Float64}
    b::Vector{Float64}
    losses::Vector{Float64}
    fitted::Bool
    classes::Vector{Any}

    # Hyper parameters
    η::Float64
    num_iter::Int
    random_state::Union{Nothing, Int64}
    optim_alg::Symbol
    batch_size::Int 
end

function MulticlassPerceptron(; η=0.01, num_iter=100, random_state=nothing,
            optim_alg=:SGD, batch_size=32)
    if !(optim_alg ∈ [:SGD, :Batch, :MiniBatch])
        throw("""`optim_alg` should be in [:SGD, :Batch, :MiniBatch]""")
    else        
        return MulticlassPerceptron(Matrix{Float64}(undef, 0, 0), Float64[], Float64[], false, [], η, num_iter, random_state, optim_alg, batch_size)
    end
end

# Model training
"""Multiclass Perceptron training model"""
function (m::MulticlassPerceptron)(X::Matrix, y::Vector)
    if m.random_state !== nothing
        Random.seed!(m.random_state)
    end
    empty!(m.losses)

    # Get unique classes
    m.classes = sort(unique(y))
    n_classes = length(m.classes)
    n_features = size(X, 2)

    # Initialize weights and bias
    m.W = randn(n_features, n_classes) ./ 100
    m.b = zeros(n_classes)

    # Create a dictionary to map classes to indices
    class_to_index = Dict(class => i for (i, class) in enumerate(m.classes))

    if m.optim_alg == :SGD
        n = length(y)
        for _ in 1:m.num_iter
            error = 0
            for i in 1:n
                xi, yi = X[i, :], y[i]
                ŷ_scores = net_input(m, xi)
                ŷ_index = argmax(ŷ_scores)
                yi_index = class_to_index[yi]
                
                if ŷ_index != yi_index
                    error += 1
                    # Update weights and bias for the correct class
                    m.W[:, yi_index] .+= m.η * xi
                    m.b[yi_index] += m.η
                    # Update weights and bias for the predicted class
                    m.W[:, ŷ_index] .-= m.η * xi
                    m.b[ŷ_index] -= m.η
                end
            end
            push!(m.losses, error)
        end
        m.fitted = true
    elseif m.optim_alg == :Batch
        # Implement batch gradient descent
    elseif m.optim_alg == :MiniBatch
        # Implement mini-batch gradient descent
    end
end

"""Multiclass Perceptron prediction with single observation"""
function (m::MulticlassPerceptron)(x::AbstractVector)
    if !m.fitted
        throw(ErrorException("Model is not fitted yet."))
    end
    scores = net_input(m, x)
    class_index = argmax(scores)
    return m.classes[class_index]
end

"""Multiclass Perceptron prediction with multiple observations"""
function (m::MulticlassPerceptron)(X::AbstractMatrix)
    if !m.fitted
        throw(ErrorException("Model is not fitted yet."))
    end
    scores = net_input(m, X)
    class_indices = [argmax(score) for score in eachrow(scores)]
    return [m.classes[i] for i in class_indices]
end


mcp = MulticlassPerceptron(η=0.1, random_state=1)
mcp(Xtrn_std, ytrn)
ŷ = mcp(Xtst_std)
print("Misclassified examples: $(sum(ytst .!= ŷ))")
accuracy_score(ytst, ŷ)

function plot_decision_region(model, X::Matrix, y::Vector, resolution::Float64=0.02)
    x1min, x1max = minimum(X[:, 1]) - 1, maximum(X[:, 1]) + 1 
    x2min, x2max = minimum(X[:, 2]) - 1, maximum(X[:, 2]) + 1

    x1_range, x2_range = x1min:resolution:x1max, x2min:resolution:x2max    
    z = [model([x1, x2]) for x1 in x1_range, x2 in x2_range]

    contourf(x1_range, x2_range, z', colorbar=false, colormap=:plasma, alpha=0.1)
    scatter!(X[:, 1], X[:, 2], group = y, markersize = 5, markerstrokewidth = 0.5)
end

Xcomb_std = vcat(Xtrn_std, Xtst_std)
ycomb = vcat(ytrn, ytst)

plot_decision_region(ppn, Xcomb_std, ycomb)

using MLJ, MLJModels, MLJLinearModels
using Distances, Optim
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
     linewidth=2,
     xlabel="σ(z)",
     ylabel="L(w,b)")

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


# Tackling overfitting via regularization
doc("LogisticClassifier", pkg="MLJLinearModels")
LogisticClassifier = @load LogisticClassifier pkg=MLJLinearModels

wpl = []
wpw = []
params = []
solver = MLJLinearModels.LBFGS()

for c in -5:0.5:5    
    model = LogisticClassifier(penalty=:l2, lambda=10.0^-c,
                               solver=solver)    
    mach = machine(model, Xtrn_std, ytrn) |> fit!;
    w = MLJ.fitted_params(mach).coefs
  
    push!(wpl, w[1][2][2])
    push!(wpw, w[2][2][2])
    push!(params, 10.0^c)
end

plot(params, [wpl wpw], xscale=:log10, 
     label = ["Petal length" "Petal width"],
     legend=:bottomleft,
     lw=2)

# Maximum margin classification with support vector machines
using LIBSVM
models("SVC")
SVC = @load ProbabilisticSVC pkg=LIBSVM
doc("ProbabilisticSVC", pkg="LIBSVM")
Random.seed!(1)
svm = SVC(kernel=LIBSVM.Kernel.Linear, cost=1.0)
mach = machine(svm, Xtrn_std, ytrn) |> fit!;
plot_decision_regions(X, y, mach, test_idx=105:150)

# Solving nonlinear problems using a kernel SVM
using Distributions

Random.seed!(1)
X_xor = rand(Normal(0, 1), (200, 2))
y_xor = xor.(X_xor[:, 1] .> 0, X_xor[:, 2] .> 0)
y_xor = Int.(y_xor) 

begin
    scatter(X_xor[y_xor.==1, 1], X_xor[y_xor.==1, 2],
            color=:royalblue, marker=:square, label="Class 1")
    scatter!(X_xor[y_xor.==0, 1], X_xor[y_xor.==0, 2],
             color=:tomato, marker=:circle, label="Class 0")
    
    # Set plot properties
    xlims!((-3, 3))
    ylims!((-3, 3))
    xlabel!("Feature 1")
    ylabel!("Feature 2")        
end
Random.seed!(1)
svm = SVC(kernel=LIBSVM.Kernel.RadialBasis, gamma=0.10, cost=10.0)
mach = machine(svm, X_xor, categorical(y_xor)) |> fit!;

begin
    markers = [:circle, :rect]
    x1_min, x1_max = minimum(X_xor[:, 1]) - 1, maximum(X_xor[:, 1]) + 1
    x2_min, x2_max = minimum(X_xor[:, 2]) - 1, maximum(X_xor[:, 2]) + 1
    xx1 = range(x1_min, x1_max, length=l)
    xx2 = range(x2_min, x2_max, length=l)
    
    Z = [predict_mode(mach, [x1 x2])[1] for x1 in xx1, x2 in xx2]

    p = contourf(xx1, xx2, Z, 
                 color=[:red, :blue],
                 levels=3, alpha=0.3, legend=false);

    for (i, cl) in enumerate(unique(y))
        idx = findall(y .== cl)
        scatter!(p, X[idx, 1], X[idx, 2], marker=markers[i], label="Class $cl", ms=4)
    end
    xlabel!("Feature 1")
    ylabel!("Feature 2")    
    plot!(legend=:topleft)    
end

Random.seed!(1)
svm = SVC(kernel=LIBSVM.Kernel.RadialBasis, gamma=0.2, cost=1.0)
mach = machine(svm, Xtrn_std, ytrn) |> fit!;
plot_decision_regions(Xcomb_std, ycomb, mach, test_idx=105:150)

Random.seed!(1)
svm = SVC(kernel=LIBSVM.Kernel.RadialBasis, gamma=100.0, cost=1.0)
mach = machine(svm, Xtrn_std, ytrn) |> fit!;
plot_decision_regions(Xcomb_std, ycomb, mach, test_idx=105:150)

# Decision Tree Learning
entropy(p) = -p*log2(p) - (1-p)*log2(1-p)

x = 0.01:0.01:1
ent = entropy.(x)
plot(x, ent,
     xlabel="Class membership probability p(i=1)",
     ylabel="Entropy",
     legend=false)

gini(p) = p*(1-p) + (1-p)*p
error(p) = 1- max(p, 1-p)

sc_ent = ent.*0.5
err = error.(x)

funcs = [ent, sc_ent, gini, err]
labs = ["Ent", "Ent sc", "Gini", "Miscl."]
lines = [:solid, :solid, :dash, :dashdot]
colors = [:black, :lightgray, :red, :green]

p = plot(ylims=[0.0, 1.1], xlabel="p(i=1)", ylabel="impurity index");
for (i, lab, ls, c) in zip(funcs, labs, lines, colors)
    plot!(p, x, i, label=lab, ls=ls, lw=2, color=c)
end
hline!(p, [0.5], lw=1, color=:black, ls=:dash, label=nothing);
hline!(p, [1], lw=1, color=:black, ls=:dash, label=nothing);
plot!(p, legend=:outertop, legendcolumns=4, margin=10Plots.mm)

## Building a decision tree
models("DecisionTree")

doc("DecisionTreeClassifier", pkg="BetaML")
DecisionTree = @load DecisionTreeClassifier pkg=BetaML
tree_model = DecisionTree(
    max_depth=4,
    splitting_criterion=BetaML.Utils.gini,
    rng=Random.seed!(1)
)
mach = machine(tree_model, Xtrn, ytrn) |> fit!;

Xcomb = vcat(Xtrn, Xtst)
ycomb = vcat(ytrn, ytst)

begin
    test_idx = 105:150
    markers = [:circle, :rect, :utriangle, :dtriangle, :diamond]
    colors = [:red, :lightblue, :lightgreen, :gray, :cyan]
    len=200
    # Plot the decision surface
    x1_min, x1_max = minimum(Xcomb[:, 1]) - 1, maximum(Xcomb[:, 1]) + 1
    x2_min, x2_max = minimum(Xcomb[:, 2]) - 1, maximum(Xcomb[:, 2]) + 1

    xx1 = range(x1_min, x1_max, length=len)
    xx2 = range(x2_min, x2_max, length=len)
    
    Z = [predict_mode(mach, table([x1 x2]))[1] for x1 in xx1, x2 in xx2]

    color_map = Dict(
        "setosa" => 1,
        "versicolor" => 2,
        "virginica" => 3
        )

    Z_numeric = [color_map[z] for z in Z]

    p = contourf(xx1, xx2, Z_numeric, 
                 color=[:red, :blue, :lightgreen],
                 levels=3, alpha=0.3, legend=false);

    # Plot data points
    for (i, cl) in enumerate(unique(y))
        idx = findall(y .== cl)
        scatter!(p, X[idx, 1], X[idx, 2], marker=markers[i], label="Class $cl", ms=4)
    end

    # Highlight test examples
    if !isempty(test_idx)
        X_test = Xcomb[test_idx, :]
        scatter!(p, X_test[:, 1], X_test[:, 2],
                 marker=:circle,
                 mc=:black, ms=2,
                 label="Test set",
                 markersize=6)
    end
    xlabel!("Petal length (cm)")
    ylabel!("Petal width (cm)")
    plot!(legend=:topleft)
end 

# Graph Visualization


# Random Forest Classifier
models("Forest")
RandomForest = @load RandomForestClassifier pkg=BetaBinomial
doc("RandomForestClassifier", pkg="BetaML")
forest = RandomForest(
    n_trees=25,
    max_depth=4,
    rng=Random.seed!(1)
)

mach = machine(forest, Xtrn, ytrn) |> fit!;
begin
    test_idx = 105:150
    markers = [:circle, :rect, :utriangle, :dtriangle, :diamond]
    colors = [:red, :lightblue, :lightgreen, :gray, :cyan]
    len=200
    # Plot the decision surface
    x1_min, x1_max = minimum(Xcomb[:, 1]) - 1, maximum(Xcomb[:, 1]) + 1
    x2_min, x2_max = minimum(Xcomb[:, 2]) - 1, maximum(Xcomb[:, 2]) + 1

    xx1 = range(x1_min, x1_max, length=len)
    xx2 = range(x2_min, x2_max, length=len)
    
    Z = [predict_mode(mach, table([x1 x2]))[1] for x1 in xx1, x2 in xx2]

    color_map = Dict(
        "setosa" => 1,
        "versicolor" => 2,
        "virginica" => 3
        )

    Z_numeric = [color_map[z] for z in Z]

    p = contourf(xx1, xx2, Z_numeric, 
                 color=[:red, :blue, :lightgreen],
                 levels=3, alpha=0.3, legend=false);

    # Plot data points
    for (i, cl) in enumerate(unique(y))
        idx = findall(y .== cl)
        scatter!(p, X[idx, 1], X[idx, 2], marker=markers[i], label="Class $cl", ms=4)
    end

    # Highlight test examples
    if !isempty(test_idx)
        X_test = Xcomb[test_idx, :]
        scatter!(p, X_test[:, 1], X_test[:, 2],
                 marker=:circle,
                 mc=:black, ms=2,
                 label="Test set",
                 markersize=6)
    end
    xlabel!("Petal length (cm)")
    ylabel!("Petal width (cm)")
    plot!(legend=:topleft)
end 

# K Nearest Neighbors
models("Neighbor")
KNN = @load KNNClassifier pkg=NearestNeighborModels
doc("KNNClassifier", pkg="NearestNeighborModels")

knn = KNN(
    K=5,
    metric=Distances.Minkowski(2),
    algorithm=:kdtree
)

mach = machine(knn, Xtrn, ytrn) |> fit!;
plot_decision_regions(Xcomb, ycomb, mach, test_idx=106:150)



