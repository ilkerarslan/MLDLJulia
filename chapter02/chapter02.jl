include("chapter02_module.jl")
using .CHAPTER02
using DataFrames, RDatasets, Plots, Colors

iris = dataset("datasets", "iris");
iris

X = iris[1:100, [:SepalLength, :PetalLength]] |> Matrix
y = (iris.Species[1:100] .== "setosa") .|> Int

scatter(X[:, 1], X[:, 2],
        group=iris[1:100, :Species],
        xlabel="Sepal Length (cm)",
        ylabel="Petal Length (cm)")

pn = Perceptron()
fit!(pn, X, y; η=0.1, num_iter=10)

plot(1:length(pn.errors), pn.errors)
scatter!(1:length(pn.errors), pn.errors, legend=false)
xlabel!("Epochs")
ylabel!("Errors")

plot_decision_region(pn, X, y)
