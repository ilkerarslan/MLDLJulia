module CHAPTER02
using Random, ProgressBars, Plots, Statistics, LinearAlgebra
export AbstractModel, Perceptron, Adaline, OptimAlgorithm
export fit!, predict, plot_decision_region

# Abstract type for models
abstract type AbstractModel end

# Optimization algorithms
@enum OptimAlgorithm begin
    SGD
    BatchGD
    MiniBatchGD
end

"""
    Perceptron

Perceptron classifier
"""
@kwdef mutable struct Perceptron <: AbstractModel
    w::Vector{Float64} = Float64[]
    b::Float64 = 0.0
    losses::Vector{Float64} = Float64[]
end

"""
    Adaline

Adaptive Linear Neuron classifier
"""
@kwdef mutable struct Adaline <: AbstractModel
    w::Vector{Float64} = Float64[]
    b::Float64 = 0.0
    losses::Vector{Float64} = Float64[]
end

"""Calculate net input"""
net_input(m::T, x::AbstractVector) where {T <: AbstractModel} = (x' * m.w + m.b)
net_input(m::T, X::AbstractMatrix) where {T <: AbstractModel} = (X * m.w .+ m.b)

"""Compute linear activation"""
linearactivation(X) = X

"""Return class label after unit step"""
predict(m::Perceptron, x::AbstractVector) = net_input(m, x) ≥ 0.0 ? 1 : 0
predict(m::Adaline, x::AbstractVector) = linearactivation(net_input(m, x)) ≥ 0.5 ? 1 : 0


"""
    fit!(model::AbstractModel, X::Matrix, y::Vector; kwargs...)

Fit the model to the data using the specified optimization algorithm.
"""
function fit!(model::AbstractModel, X::Matrix, y::Vector;
              η::Float64=0.01, num_iter::Int=50, batch_size::Int=32,
              optim_alg::OptimAlgorithm=SGD, random_seed::Int=1)
    Random.seed!(random_seed)
    model.w = randn(size(X, 2)) .* 0.01
    model.b = 0.0
    empty!(model.losses)

    if optim_alg == SGD
        _fit_sgd!(model, X, y, η, num_iter)
    elseif  optim_alg == BatchGD
        _fit_batch_gd!(model, X, y, η, num_iter)   
    end
end

function _fit_sgd!(model::Perceptron, X::Matrix, y::Vector, η::Float64, num_iter::Int)
    m = length(y)
    for _ in ProgressBar(1:num_iter)
        error = 0.0
        for i in 1:m
            xi, yi = X[i, :], y[i]
            ŷ = predict(model, xi)
            ∇ = η * (yi - ŷ)
            model.w += ∇ * xi
            model.b += ∇
            error += Int(∇ != 0.0)
        end
        push!(model.losses, error)
    end
end

function _fit_sgd!(model::Adaline, X::Matrix, y::Vector, η::Float64, num_iter::Int, shuffle=true)
    X_ = copy(X)
    y_ = copy(y)
    m = length(y_)

    for _ in ProgressBar(1:num_iter)
        if shuffle==true
            perm = randperm(m)
            X_ = X_[perm, :]
            y_ = y_[perm]
        end
        losses = []
        for i in 1:m
            xi, yi = X_[i,:], y_[i]
            error = yi - predict(model, xi)
            model.w += η * 2.0 * xi * error
            model.b += η * 2.0 * error 
            push!(losses, error^2)
        end
        push!(model.losses, mean(losses))
    end
end

function _fit_batch_gd!(model::AbstractModel, X::Matrix, y::Vector, η::Float64, num_iter::Int)
    m = length(y)
    for _ in ProgressBar(1:num_iter)
        output = linearactivation(net_input(model, X))        
        errors = (y .- output)
        model.w .+= η .* 2.0 .* X' * errors ./ m
        model.b += η * 2.0 * mean(errors)
        loss = mean(errors.^2)
        push!(model.losses, loss)
    end
end


"""
    plot_decision_region(model::AbstractModel, X::Matrix, y::Vector, resolution::Float64=0.02)

Plot the decision region for the model.
"""
function plot_decision_region(model::AbstractModel, X::Matrix, y::Vector, resolution::Float64=0.02)
    x1min, x1max = minimum(X[:, 1]) - 1, maximum(X[:, 1]) + 1 
    x2min, x2max = minimum(X[:, 2]) - 1, maximum(X[:, 2]) + 1

    x1_range = x1min:resolution:x1max
    x2_range = x2min:resolution:x2max
    
    z = [predict(model, [x1, x2]) for x1 in x1_range, x2 in x2_range]

    contourf(x1_range, x2_range, z', 
             colorbar=false, 
             colormap=:plasma, 
             alpha=0.1)
    scatter!(X[:, 1], X[:, 2],
            group = y,
            markersize = 5,
            markerstrokewidth = 0.5)
end



end # of Module