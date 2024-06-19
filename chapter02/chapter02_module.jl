module CHAPTER02

using Random, ProgressBars, Plots, Colors

export Perceptron, fit!, plot_decision_region

"""Perceptron classifier"""
@kwdef mutable struct Perceptron
    w::Vector{Float64} = []
    b::Float64 = 0.0
    errors::Vector{Float64} = []
end

"""Return class label"""
(p::Perceptron)(X::Matrix) = (X * p.w .+ p.b) .≥ 0.0
(p::Perceptron)(x::Vector) = (x' * p.w + p.b) ≥ 0.0 ? 1 : 0

function fit!(model::Perceptron, X, y; 
              η=0.01, num_iter=100, random_seed=1)   
    Random.seed!(random_seed)
    model.w = randn(size(X, 2)) ./ 1_000
    model.b = 0.0
    model.errors = Float64[]

    for _ in ProgressBar(1:num_iter)
        error = 0.0
        for i in 1:length(y)
            xi = X[i, :]
            yi = y[i]
            ypred = model(xi)
            ∇w = (yi - ypred) * xi
            ∇b = (yi - ypred)
            model.w += η * ∇w
            model.b += η * ∇b
            err = (yi - ypred) != 0.0
            error += err
        end
        push!(model.errors, error)
    end    
end

function plot_decision_region(model, X, y, resolution=0.02)
    x1min, x1max = minimum(X[:, 1]) - 1, maximum(X[:, 1]) + 1 
    x2min, x2max = minimum(X[:, 2]) - 1, maximum(X[:, 2]) + 1

    a = x1min:resolution:x1max
    b = x2min:resolution:x2max
    c = zeros(length(a), length(b))'

    for i in 1:length(a), j in 1:length(b)
        c[j,i] = model([a[i], b[j]])
    end

    contourf(a, b, c, 
             colorbar=false, 
             colormap=:plasma, 
             alpha=0.1)
    scatter!(X[:, 1], X[:, 2],
            group = y,
            xlabel="Sepal Length (cm)",
            ylabel="Petal Length (cm)")
end


end # of Module CHAPTER02