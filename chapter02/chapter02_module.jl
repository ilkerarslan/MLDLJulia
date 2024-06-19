module CHAPTER02

using Random, ProgressBars, Plots, Statistics

export Perceptron, Adaline 
export fitGD!, fitSGD!, plot_decision_region

"""Perceptron classifier"""
@kwdef mutable struct Perceptron
    w::Vector{Float64} = []
    b::Float64 = 0.0
    errors::Vector{Float64} = []
end

"""Return class label"""
(m::Perceptron)(X::Matrix) = (X * m.w .+ m.b) .≥ 0.0
(m::Perceptron)(x::Vector) = (x' * m.w + m.b) ≥ 0.0 ? 1 : 0

function fitSGD!(model::Perceptron, X, y; 
              η=0.01, num_iter=50, random_seed=1)   
    Random.seed!(random_seed)
    model.w = randn(size(X, 2)) ./ 100
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
            group = y)
end

@kwdef mutable struct Adaline
    w::Vector{Float64} = []
    b::Float64 = 0.0
    losses::Vector{Float64} = []
end

(m::Adaline)(X::Matrix) = (X * m.w .+ m.b)
(m::Adaline)(x::Vector) = (x' * m.w + m.b) ≥ 0.5 ? 1 : 0


function fitGD!(model::Adaline, X, y; 
              η=0.01, num_iter=50, random_seed=1) 
    Random.seed!(random_seed)
    model.w = randn(size(X, 2)) ./ 100
    model.b = 0.0
    model.losses = Float64[]
    for _ in ProgressBar(1:num_iter)
        ypreds = model(X)
        errors = (y .- ypreds)
        model.w += η * 2.0 * (X' * errors) ./ size(X, 1)
        model.b += η * 2.0 * mean(errors)
        loss = mean(errors .^ 2)
        push!(model.losses, loss)
    end
end

function fitSGD!(model::Adaline, X, y; 
                 η=0.01, num_iter=10, shuffle=true, random_seed=1)   
    
    Random.seed!(random_seed)
    X_ = copy(X)
    y_ = copy(y)
    model.w = randn(size(X_, 2)) ./ 100
    model.b = 0.0
    model.losses = Float64[]

    for _ in ProgressBar(1:num_iter)
        if shuffle==true        
            perm = randperm(length(y))
            X_ = X_[perm, :]
            y_ = y_[perm]
        end
        losses = []
        for i in 1:length(y)
            xi = X_[i, :]
            yi = y_[i]
            ypred = xi' * model.w + model.b
            error = yi - ypred
            model.w += η * 2.0 * xi * error
            model.b += η * 2.0 * error
            loss = error^2
            push!(losses, loss)
        end        
        push!(model.losses, mean(losses))
    end    
end

end # of Module CHAPTER02