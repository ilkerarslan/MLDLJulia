include("chapter02_module.jl")
using .CHAPTER02
using DataFrames, RDatasets, Plots, Colors, Random, Statistics

# Data
iris = dataset("datasets", "iris");
iris

X = iris[1:100, [:SepalLength, :PetalLength]] |> Matrix
y = (iris.Species[1:100] .== "setosa") .|> Int

scatter(X[:, 1], X[:, 2],
        group=iris[1:100, :Species],
        xlabel="Sepal Length (cm)",
        ylabel="Petal Length (cm)")

# Perceptron
pn = Perceptron()
fitSGD!(pn, X, y; η=0.1, num_iter=10)

begin
    plot(1:length(pn.errors), pn.errors)
    scatter!(1:length(pn.errors), pn.errors, legend=false)
    xlabel!("Epochs")
    ylabel!("Errors")
    
    plot_decision_region(pn, X, y)
    xlabel!("Sepal Length (cm)")
    ylabel!("Petal Length (cm)")                
end

# Adaline
ada1 = Adaline()
fitGD!(ada1, X, y, η=0.1, num_iter=15)
ada2 = Adaline()
fitGD!(ada2, X, y, η=0.0001, num_iter=15)

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

ada_gd = Adaline()
fitGD!(ada_gd, X_std, y, η=0.5, num_iter=20)

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

# Stochastic gradient Descent
ada_sgd = Adaline()
fitSGD!(ada_sgd, X_std, y, num_iter=15, η=0.01, random_seed=1)

begin
    layout = @layout [a b]
    p1 = plot_decision_region(ada_sgd, X_std, y);
    xlabel!(p1, "Sepal Length (standardized)");
    ylabel!(p1, "Petal Length (standardized)");
    title!(p1, "Adaline - SGD");
    p2 = plot(1:length(ada_sgd.losses), ada_sgd.losses,
              xlabel="Epochs",
              ylabel="Mean squared error",
              legend=false);
    scatter!(p2, 1:length(ada_sgd.losses), ada_sgd.losses);
    plot(p1, p2, layout=layout)        
end