using Revise

using NovaML.LinearModel: Perceptron, Adaline

using LinearAlgebra, RDatasets, Plots
using DataFrames, Random, Statistics

v1 = [1, 2, 3]
v2 = 0.5 .* v1
acos( v1'*v2 / (norm(v1)*norm(v2)) )

# Data
iris = dataset("datasets", "iris")
X = iris[1:100, [:SepalLength, :PetalLength]] |> Matrix
y = (iris.Species[1:100] .== "setosa") .|> Int

scatter(X[:, 1], X[:, 2],
        group=iris[1:100, :Species],
        xlabel="Sepal Length (cm)",
        ylabel="Petal Length (cm)",
        title="Iris Dataset: Setosa vs. Versicolor")

pn = Perceptron(η=0.1, num_iter=10, optim_alg=:SGD)
pn(X, y)
pn(X)

begin
    plot(1:length(pn.losses), pn.losses,
    xlabel="Epochs", ylabel="Errors",
    title="Perceptron Training Errors",
    legend=false)
    scatter!(1:length(pn.losses), pn.losses)
end

function plot_decision_region(model, X::Matrix, y::Vector, resolution::Float64=0.02)
    x1min, x1max = minimum(X[:, 1]) - 1, maximum(X[:, 1]) + 1 
    x2min, x2max = minimum(X[:, 2]) - 1, maximum(X[:, 2]) + 1

    x1_range, x2_range = x1min:resolution:x1max, x2min:resolution:x2max    
    z = [model([x1, x2]) for x1 in x1_range, x2 in x2_range]

    contourf(x1_range, x2_range, z', colorbar=false, colormap=:plasma, alpha=0.1)
    scatter!(X[:, 1], X[:, 2], group = y, markersize = 5, markerstrokewidth = 0.5)
end

begin
    plot_decision_region(pn, X, y)
    xlabel!("Sepal Length (cm)")
    ylabel!("Petal Length (cm)")
    title!("Perceptron Decision Boundary")
end

# Adaline
ada1 = Adaline(η=0.1, num_iter=15, optim_alg=:Batch)
ada2 = Adaline(η=0.0001, num_iter=15, optim_alg=:Batch)

ada1(X, y)
ada2(X, y)

begin
    layout = @layout [a b]
    p1 = plot(1:length(ada1.losses), log10.(ada1.losses), 
              xlabel="Epochs", ylabel="log(mse)",
              title="Adaline - lr=0.1",
              legend=false);
    scatter!(p1, 1:length(ada1.losses), log10.(ada1.losses));
    
    p2 = plot(1:length(ada2.losses), ada2.losses, 
              xlabel="Epochs", ylabel="mse",
              title="Adaline - lr=0.0001",
              legend=false);
    scatter!(p2, 1:length(ada2.losses), ada2.losses);
    plot(p1, p2, layout=layout)        
end

# Adaline with standardized inputs
X_std = copy(X)
X_std[:, 1] = (X_std[:, 1] .- mean(X_std[:, 1])) ./ std(X_std[:, 1]) 
X_std[:, 2] = (X_std[:, 2] .- mean(X_std[:, 2])) ./ std(X_std[:, 2]) 

ada_gd = Adaline(η=0.5, num_iter=20, optim_alg=:Batch)
ada_gd(X_std, y)

begin
    layout = @layout [a b]
    p1 = plot_decision_region(ada_gd, X_std, y);
    xlabel!(p1, "Sepal Length (standardized)");
    ylabel!(p1, "Petal Length (standardized)");
    title!(p1, "Adaline - Gradient Descent");
    p2 = plot(1:length(ada_gd.losses), ada_gd.losses,
              xlabel="Epochs",
              ylabel="Mean squared error",
              legend=false);
    scatter!(p2, 1:length(ada_gd.losses), ada_gd.losses);
    plot(p1, p2, layout=layout)        
end


ada_sgd = Adaline(num_iter=15, η=0.01, random_state=1, optim_alg=:SGD)
ada_sgd(X_std, y)

begin
    layout = @layout [a b]
    p1 = plot_decision_region(ada_sgd, X_std, y);
    xlabel!(p1, "Sepal Length (standardized)");
    ylabel!(p1, "Petal Length (standardized)");
    title!(p1, "Adaline - Stochastic gradient descent");
    p2 = plot(1:length(ada_sgd.losses), ada_sgd.losses,
              xlabel="Epochs",
              ylabel="Average loss",
              legend=false);
    scatter!(p2, 1:length(ada_sgd.losses), ada_sgd.losses);
    plot(p1, p2, layout=layout)        
end
