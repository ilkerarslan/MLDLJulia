using Revise
using RDatasets, DataFrames, StatsBase, LinearAlgebra
using Random, Plots, StatsBase

using NovaML.ModelSelection: train_test_split
using NovaML.PreProcessing: StandardScaler
using NovaML.LinearModel: MulticlassPerceptron, LogisticRegression
using NovaML.Metrics: accuracy_score
using NovaML.MultiClass: OneVsRestClassifier
using NovaML.Tree: DecisionTreeClassifier
using NovaML.Ensemble: RandomForestClassifier
using NovaML.Neighbors: KNeighborsClassifier

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
Xtrnstd = stdscaler(Xtrn)
Xtststd = stdscaler(Xtst)

# Multiclass Perceptron
mcp = MulticlassPerceptron(η=0.1, random_state=1)
mcp(Xtrnstd, ytrn)
ŷ = mcp(Xtststd)
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

function plot_decision_region(model::DecisionTreeClassifier, X::Matrix, y::Vector, resolution::Float64=0.02)
    x1min, x1max = minimum(X[:, 1]) - 1, maximum(X[:, 1]) + 1 
    x2min, x2max = minimum(X[:, 2]) - 1, maximum(X[:, 2]) + 1

    x1_range, x2_range = x1min:resolution:x1max, x2min:resolution:x2max    
    z = [model([x1 x2])[1] for x1 in x1_range, x2 in x2_range]

    contourf(x1_range, x2_range, z', colorbar=false, colormap=:plasma, alpha=0.1)
    scatter!(X[:, 1], X[:, 2], group = y, markersize = 5, markerstrokewidth = 0.5)
end

Xcomb_std = vcat(Xtrnstd, Xtststd)
ycomb = vcat(ytrn, ytst)

plot_decision_region(mcp, Xcomb_std, ycomb)

# Logistic Regression
begin
    function sigmoid(z)
        return 1.0 ./ (1.0 .+ exp.(-z))
    end
    z = range(-7, 7, step=0.1)
    sigma_z = sigmoid(z)    
    plot(z, sigma_z, label="σ(z)", legend=false)
    vline!([0.0], color=:black)
    ylims!(-0.1, 1.1)
    xlabel!("z")
    ylabel!("σ(z)")
    yticks!([0.0, 0.5, 1.0])
    plot!(grid=true, gridalpha=0.5, gridstyle=:dash, minorgrid=true)
    plot!(size=(600, 400), margin=3Plots.mm)        
end

begin
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
end

idx = (ytrn .== 0) .|| (ytrn .== 1)
Xtrn01_subset = Xtrnstd[idx, :] |> Matrix
ytrn01_subset = ytrn[idx]
ytrn01_subset = [x == 0 ? 0 : 1 for x in ytrn01_subset]

lrgd = LogisticRegression(η=0.01, num_iter=2000, random_state=1, solver=:batch)
lrgd(Xtrn01_subset, ytrn01_subset)
plot_decision_region(lrgd, Xtrn01_subset, ytrn01_subset)

lr = LogisticRegression()
ovr = OneVsRestClassifier(lr)
ovr(Xtrnstd, ytrn)
ovr.classifiers[3]

ŷ = ovr(Xtststd)
accuracy_score(ytst, ŷ)

ovr(Xtststd, type=:probs)

# Tackling overfitting via regularization 
weights = Matrix(undef, 0, 2)
params = []
for l in Int.(-5:5)
    lr = LogisticRegression(λ=10.0^l)
    ovr = OneVsRestClassifier(lr)
    ovr(Xtrnstd, ytrn)
    weights = vcat(weights, ovr.classifiers[2].w')
    push!(params, 10.0^l)
end

begin
    plot(params, weights[:, 1], label="Petal length", xaxis=:log)
    plot!(params, weights[:, 2], label="Petal width", xaxis=:log)
    xlabel!("λ")
    ylabel!("Weights")        
end

# Maximum margin classification with support vector machines
iris = dataset("datasets", "iris")
X = Matrix(iris[:, 1:4])
y = iris.Species


using NovaML.SVM: SVC
svm = SVC(kernel=:linear, C=1.0, gamma=:scale)
svm(Xtrnstd, ytrn)
ŷtst = svm(Xtst_std)

SVC = @load SVC pkg=LIBSVM
model = SVC(kernel=(x1, x2) -> x1'*x2)
mach = machine(model, Xtrn, ytrn) |> fit!;
ypreds = MLJ.predict(mach, Xtst)
sum(ypreds .!= ytst)

# Solving nonlinear problems using a kernel SVM
using Distributions
using LIBSVM
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
    test_idx = []
    len = 300
    markers = [:circle, :rect, :utriangle, :dtriangle, :diamond]
    colors = [:red, :lightblue, :lightgreen, :gray, :cyan]

    # Plot the decision surface
    x1_min, x1_max = minimum(X_xor[:, 1]) - 1, maximum(X_xor[:, 1]) + 1
    x2_min, x2_max = minimum(X_xor[:, 2]) - 1, maximum(X_xor[:, 2]) + 1
    xx1 = range(x1_min, x1_max, length=len)
    xx2 = range(x2_min, x2_max, length=len)        
    Z = [MLJ.predict(mach, [x1 x2])[1] for x1 in xx1, x2 in xx2]    

    p = contourf(xx1, xx2, Z, 
                 color=[:red, :blue, :lightgreen],
                 levels=3, alpha=0.3, legend=false);

    # Plot data points
    for (i, cl) in enumerate(unique(y_xor))
        idx = findall(y_xor .== cl)
        scatter!(p, X_xor[idx, 1], X_xor[idx, 2], marker=markers[i], label="Class $cl", ms=4)
    end
    scatter!()
    # Highlight test examples
end

Random.seed!(1)
svm = SVC(kernel=LIBSVM.Kernel.RadialBasis, gamma=0.2, cost=1.0)
mach = machine(svm, Xtrn_std, categorical(ytrn)) |> fit!;

begin
    test_idx = 105:150
    len = 300
    markers = [:circle, :rect, :utriangle, :dtriangle, :diamond]
    colors = [:red, :lightblue, :lightgreen, :gray, :cyan]

    # Plot the decision surface
    x1_min, x1_max = minimum(Xcomb_std[:, 1]) - 1, maximum(Xcomb_std[:, 1]) + 1
    x2_min, x2_max = minimum(Xcomb_std[:, 2]) - 1, maximum(Xcomb_std[:, 2]) + 1
    xx1 = range(x1_min, x1_max, length=len)
    xx2 = range(x2_min, x2_max, length=len)        
    Z = [MLJ.predict(mach, [x1 x2])[1] for x1 in xx1, x2 in xx2]    

    p = contourf(xx1, xx2, Z, 
                 color=[:red, :blue, :lightgreen],
                 levels=3, alpha=0.3, legend=false);

    # Plot data points
    for (i, cl) in enumerate(unique(ycomb))
        idx = findall(ycomb .== cl)
        scatter!(p, Xcomb_std[idx, 1], Xcomb_std[idx, 2], marker=markers[i], label="Class $cl", ms=4)
    end
    scatter!()
    # Highlight test examples
end

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

Xcomb = vcat(Xtrn, Xtst)
ycomb = vcat(ytrn, ytst)


using NovaML.Tree
tree = Tree.DecisionTreeClassifier(max_depth=4, random_state=1)
tree(Xtrn, ytrn)
ŷ = tree(Xtst)
sum(ytst .!= ŷ)
plot_decision_region(tree, Xcomb, ycomb)

# Random Forest Classifier
using NovaML.Ensemble: RandomForestClassifier
forest = RandomForestClassifier(n_estimators=25, random_state=1)
forest(Xtrn, ytrn)
ŷ = forest(Xtst)
sum(ŷ .!= ytst)

# K Nearest Neighbors
using NovaML.Neighbors: KNeighborsClassifier

knn = KNeighborsClassifier(
    K=5,   
    algorithm=:kdtree
)

mach = machine(knn, Xtrn, ytrn) |> fit!;

knn = KNeighborsClassifier(
    n_neighbors=5)

knn(Xtrn, ytrn)
ŷ = knn(Xtst)
sum(ŷ .!= ytst)

