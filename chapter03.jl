using Revise
using DataFrames, StatsBase, LinearAlgebra
using Random, Plots, StatsBase

# Data
using NovaML.Datasets
iris = load_iris()
X = iris["data"][:, 3:4]
y = iris["target"]

using NovaML.ModelSelection: train_test_split
Xtrn, Xtst, ytrn, ytst = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

println("Labels count in y: ", values(StatsBase.countmap(y)))
println("Labels counts in ytrn: ", values(StatsBase.countmap(ytrn)))
println("Labels counts in ytst: ", values(StatsBase.countmap(ytst)))

using NovaML.PreProcessing: StandardScaler
stdscaler = StandardScaler()
stdscaler(Xtrn)
Xtrnstd = stdscaler(Xtrn)
Xtststd = stdscaler(Xtst)

# Multiclass Perceptron
using NovaML.MultiClass: MulticlassPerceptron
mcp = MulticlassPerceptron(η=0.1, random_state=1)
mcp(Xtrnstd, ytrn)
ŷ = mcp(Xtststd)
print("Misclassified examples: $(sum(ytst .!= ŷ))")

using NovaML.Metrics: accuracy_score
accuracy_score(ytst, ŷ)

function plot_decision_region(model, X::Matrix, y::Vector, resolution::Float64=0.02)
    x1min, x1max = minimum(X[:, 1]) - 1, maximum(X[:, 1]) + 1 
    x2min, x2max = minimum(X[:, 2]) - 1, maximum(X[:, 2]) + 1

    x1_range, x2_range = x1min:resolution:x1max, x2min:resolution:x2max    
    z = [model([x1, x2]) for x1 in x1_range, x2 in x2_range]

    contourf(x1_range, x2_range, z', colorbar=false, colormap=:plasma, alpha=0.1)
    scatter!(X[:, 1], X[:, 2], group = y, markersize = 5, markerstrokewidth = 0.5)
end

function plot_decision_region(model, X::Matrix, y::Vector, resolution::Float64=0.02)
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

idx = (ytrn .== 1) .|| (ytrn .== 2)
Xtrn01_subset = Xtrnstd[idx, :]
ytrn01_subset = ytrn[idx]
ytrn01_subset = [x == 1 ? 0 : 1 for x in ytrn01_subset]

using NovaML.LinearModel: LogisticRegression
lrgd = LogisticRegression(η=0.01, num_iter=2000, random_state=1, solver=:batch)
lrgd(Xtrn01_subset, ytrn01_subset)
plot_decision_region(lrgd, Xtrn01_subset, ytrn01_subset)

using NovaML.MultiClass: OneVsRestClassifier
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

begin
    p = plot(ylims=[0.0, 1.1], xlabel="p(i=1)", ylabel="impurity index");
    for (i, lab, ls, c) in zip(funcs, labs, lines, colors)
        plot!(p, x, i, label=lab, ls=ls, lw=2, color=c)
    end
    hline!(p, [0.5], lw=1, color=:black, ls=:dash, label=nothing);
    hline!(p, [1], lw=1, color=:black, ls=:dash, label=nothing);
    plot!(p, legend=:outertop, legendcolumns=4, margin=10Plots.mm)
end

Xcomb = vcat(Xtrn, Xtst)
ycomb = vcat(ytrn, ytst)

using NovaML.Tree: DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=4, random_state=1)
tree(Xtrn, ytrn)
ŷ = tree(Xtst)
sum(ytst .!= ŷ)
plot_decision_region(tree, Xcomb, ycomb)
using NovaML.Metrics: accuracy_score, roc_auc_score
accuracy_score(ŷ, ytst)
ŷprobs = tree(Xtst, type=:probs)
roc_auc_score(ytst, ŷprobs, multiclass=:ovo)


# Random Forest Classifier
using NovaML.Ensemble: RandomForestClassifier
forest = RandomForestClassifier(n_estimators=1000, random_state=1)
forest(Xtrn, ytrn)
ŷ = forest(Xtst)
sum(ŷ .!= ytst)
accuracy_score(ytst, ŷ)

# K Nearest Neighbors
using NovaML.Neighbors: KNeighborsClassifier
knn = KNeighborsClassifier(
    n_neighbors=5,
    algorithm=:auto)

knn(Xtrn, ytrn)
ŷ = knn(Xtst)
sum(ŷ .!= ytst)
accuracy_score(ytst, ŷ)
knn(Xtst, type=:probs)