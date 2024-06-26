module CHAPTER03

using Plots, MLJ, Random, ProgressBars

export OptimAlgorithm, LogisticRegression
export fitmodel!, pred, plot_decision_regions

"""Optimization algorithm"""
@enum OptimAlgorithm begin
    SGD
    BatchGD 
    MiniBatchGD
end

"""
    LogisticRegression

Logistic Regression Classifier
"""
@kwdef mutable struct LogisticRegression
    w::Vector{Float64} = Float64[]
    b::Float64 = 0.0
    losses::Vector{Float64} = Float64[]
end

"""Calculate net input"""
net_input(m::LogisticRegression, x::AbstractVector) = x' * m.w + m.b
net_input(m::LogisticRegression, X::AbstractMatrix) = X * m.w .+ m.b


"""Compute logistic sigmoid activation"""
sigmoid(z) = 1 ./ (1 .+ exp.(-1 * clamp.(z, -250, 250)))

"""Return class label after unit step"""
pred(m::LogisticRegression, X::AbstractVector) = sigmoid(net_input(m, X)) ≥ 0.5 ? 1 : 0 

"""
    fitmodel!(model::LogisticRegression, X::Matrix, y::Vector; kwargs...)

Fit the model to the data using the specified optimization algorithm.
"""
function fitmodel!(model::LogisticRegression, X::Matrix, y::Vector;
              η::Float64=0.01, num_iter::Int=50,
              optim_alg::OptimAlgorithm=BatchGD, seed::Int=1)
    Random.seed!(seed)
    model.w = randn(size(X, 2)) ./ 100
    model.b = 0.0
    empty!(model.losses)
    
    if optim_alg==BatchGD
        _fit_batch_gd!(model, X, y, η, num_iter)
    end
end

function _fit_batch_gd!(model::LogisticRegression, X::Matrix, y::Vector,
                        η::Float64, num_iter::Int)
    m = length(y)
    for _ in ProgressBar(1:num_iter)        
        ŷ = sigmoid(net_input(model, X))
        errors = (y .- ŷ)
        model.w += 2*η .* X' * errors ./ m
        model.b += 2*η .* sum(errors) / m
        loss = (-y'*log.(ŷ) - (1 .- y)'*log.(1 .- ŷ)) / m
        push!(model.losses, loss)
    end
end



"""
    Plot Decision function

"""
function plot_decision_regions(X, y, mach; test_idx=Int[], length=300)
    markers = [:circle, :rect, :utriangle, :dtriangle, :diamond]
    colors = [:red, :lightblue, :lightgreen, :gray, :cyan]

    # Plot the decision surface
    x1_min, x1_max = minimum(X[:, 1]) - 1, maximum(X[:, 1]) + 1
    x2_min, x2_max = minimum(X[:, 2]) - 1, maximum(X[:, 2]) + 1
    xx1 = range(x1_min, x1_max, length=length)
    xx2 = range(x2_min, x2_max, length=length)
    
    Z = [(predict_mode(mach, reshape([x1, x2], 1, 2))[1])
         for x1 in xx1, x2 in xx2]

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
        X_test = X[test_idx, :]
        scatter!(p, X_test[:, 1], X_test[:, 2],
                 marker=:circle,
                 mc=:black, ms=2,
                 label="Test set",
                 markersize=6)
    end
end

function plot_decision_regions(X, y, m::LogisticRegression; test_idx=Int[], length=300)
    markers = [:circle, :rect, :utriangle, :dtriangle, :diamond]
    colors = [:red, :lightblue, :lightgreen, :gray, :cyan]
    x1_min, x1_max = minimum(X[:, 1]) - 1, maximum(X[:, 1]) + 1
    x2_min, x2_max = minimum(X[:, 2]) - 1, maximum(X[:, 2]) + 1
    xx1 = range(x1_min, x1_max, length=length)
    xx2 = range(x2_min, x2_max, length=length)
    x1 = maximum(xx1)
    x2 = maximum(xx2)
    
    Z = [pred(m, [x1, x2]) for x1 ∈ xx1, x2 ∈ xx2]
    p = contourf(xx1, xx2, Z, 
                     color=[:red, :blue, :lightgreen],
                     levels=3, alpha=0.3, legend=false);
    
    # Plot data points
    for (i, cl) in enumerate(unique(y))
        idx = findall(y .== cl)
        scatter!(p, X[idx, 1], X[idx, 2], marker=markers[i], label="Class $cl", ms=4)
    end
    
    # Highlight test examples
    if !isempty(test_idx)
        X_test = X[test_idx, :]
        scatter!(p, X_test[:, 1], X_test[:, 2],
                 marker=:circle,
                 mc=:black, ms=2,
                 label="Test set",
                 markersize=6)
    end
    
    xlabel!("Petal length (standardized)")
    ylabel!("Petal width (standardized)")    
    plot!(legend=:topleft)
end
 
end # of Module