using LinearAlgebra
using Optim
using Statistics
using Distances

"""
    SVC

Support Vector Classifier implementation for NovaML.

# Fields
- `kernel::Symbol`: Kernel type (:linear or :rbf)
- `C::Float64`: Regularization parameter
- `gamma::Union{Float64,Symbol}`: Kernel coefficient for RBF kernel
- `tol::Float64`: Tolerance for stopping criterion
- `max_iter::Int`: Maximum number of iterations
- `random_state::Union{Int,Nothing}`: Seed for random number generator
- `support_vectors_::Union{Matrix{Float64},Nothing}`: Support vectors
- `dual_coef_::Union{Matrix{Float64},Nothing}`: Coefficients of the support vectors in the decision function
- `intercept_::Union{Vector{Float64},Nothing}`: Constants in decision function
- `classes_::Union{Vector{Int},Nothing}`: Unique class labels
"""
mutable struct SVC
    kernel::Symbol
    C::Float64
    gamma::Union{Float64,Symbol}
    tol::Float64
    max_iter::Int
    random_state::Union{Int,Nothing}
    support_vectors_::Union{Matrix{Float64},Nothing}
    dual_coef_::Union{Matrix{Float64},Nothing}
    intercept_::Union{Vector{Float64},Nothing}
    classes_::Union{Vector{Int},Nothing}
    classifiers_::Union{Vector{SVC},Nothing}  # Add this line for multiclass

    function SVC(;
        kernel::Symbol=:rbf,
        C::Float64=1.0,
        gamma::Union{Float64,Symbol}=:scale,
        tol::Float64=1e-3,
        max_iter::Int=1000,
        random_state::Union{Int,Nothing}=nothing
    )
        @assert kernel in [:linear, :rbf] "Kernel must be either :linear or :rbf"
        @assert C > 0 "C must be positive"
        @assert tol > 0 "Tolerance must be positive"
        @assert max_iter > 0 "Maximum iterations must be positive"

        new(kernel, C, gamma, tol, max_iter, random_state, nothing, nothing, nothing, nothing, nothing)
    end
end

"""
    (svc::SVC)(X::Matrix{Float64}, y::Vector{Int})

Train the SVC model on the given data.

# Arguments
- `X::Matrix{Float64}`: Input features
- `y::Vector{Int}`: Target labels

# Returns
- `SVC`: Trained SVC model
"""
function (svc::SVC)(X::Matrix{Float64}, y::Vector{Int})
    # Input validation
    @assert size(X, 1) == length(y) "Number of samples in X and y must match"
    
    # Set random seed if specified
    if !isnothing(svc.random_state)
        Random.seed!(svc.random_state)
    end

    # Preprocess data
    svc.classes_ = sort(unique(y))
    n_classes = length(svc.classes_)
    
    if n_classes == 2
        return _fit_binary(svc, X, y)
    else
        return _fit_multiclass(svc, X, y)
    end
end

"""
    _fit_binary(svc::SVC, X::Matrix{Float64}, y::Vector{Int})

Fit binary SVC model.
"""
function _fit_binary(svc::SVC, X::Matrix{Float64}, y::Vector{Int})
    n_samples, n_features = size(X)
    
    # Compute kernel matrix
    K = _compute_kernel(svc, X, X)
    
    # Prepare labels
    y_binary = 2 * (y .== svc.classes_[2]) .- 1
    
    # Solve the dual optimization problem
    α = _solve_dual_problem(svc, K, y_binary)
    
    # Compute support vectors
    sv = α .> 1e-5
    svc.support_vectors_ = X[sv, :]
    svc.dual_coef_ = (α[sv] .* y_binary[sv])'  # Transpose to make it a row vector
    
    # Compute intercept
    svc.intercept_ = [_compute_intercept(svc, X[sv, :], y_binary[sv], α[sv])]
    
    return svc
end

"""
    _fit_multiclass(svc::SVC, X::Matrix{Float64}, y::Vector{Int})

Fit multiclass SVC model using one-vs-one strategy.
"""
function _fit_multiclass(svc::SVC, X::Matrix{Float64}, y::Vector{Int})
    n_classes = length(svc.classes_)
    n_classifiers = n_classes * (n_classes - 1) ÷ 2
    
    classifiers = Vector{SVC}(undef, n_classifiers)
    
    k = 1
    for i in 1:n_classes-1
        for j in i+1:n_classes
            sub_X = X[y .∈ [svc.classes_[i], svc.classes_[j]], :]
            sub_y = y[y .∈ [svc.classes_[i], svc.classes_[j]]]
            sub_y = (sub_y .== svc.classes_[j])
            
            classifiers[k] = SVC(kernel=svc.kernel, C=svc.C, gamma=svc.gamma,
                                 tol=svc.tol, max_iter=svc.max_iter)
            classifiers[k](sub_X, sub_y)
            k += 1
        end
    end
    
    svc.classifiers_ = classifiers
    return svc
end

"""
    _compute_kernel(svc::SVC, X::Matrix{Float64}, Y::Matrix{Float64})

Compute the kernel matrix for given data.
"""
function _compute_kernel(svc::SVC, X::Matrix{Float64}, Y::Matrix{Float64})
    if svc.kernel == :linear
        return X * Y'
    elseif svc.kernel == :rbf
        gamma = (svc.gamma == :scale) ? 1.0 / (size(X, 2) * var(X)) : svc.gamma
        return exp.(-gamma .* pairwise(Euclidean(), X', Y'))
    end
end

"""
    _solve_dual_problem(svc::SVC, K::Matrix{Float64}, y::Vector{Int})

Solve the dual optimization problem for SVC.
"""


function _solve_dual_problem(svc::SVC, K::Matrix{Float64}, y::Vector{Int})
    n_samples = length(y)
    
    function objective(α)
        return 0.5 * α' * (K .* (y * y')) * α - sum(α)
    end
    
    function gradient!(g, α)
        g .= (K .* (y * y')) * α .- 1
    end
    
    lower = zeros(n_samples)
    upper = fill(svc.C, n_samples)
    
    initial_x = fill(svc.C / 2, n_samples)  # Start from the middle of the box
    
    optimizer = Fminbox(LBFGS())
    
    res = optimize(objective, gradient!, lower, upper, initial_x, optimizer,
                   Optim.Options(iterations=svc.max_iter, g_tol=svc.tol))
    
    return Optim.minimizer(res)
end
"""
    _compute_intercept(svc::SVC, X::Matrix{Float64}, y::Vector{Int}, α::Vector{Float64})

Compute the intercept for SVC.
"""
function _compute_intercept(svc::SVC, X::Matrix{Float64}, y::Vector{Int}, α::Vector{Float64})
    K = _compute_kernel(svc, X, X)
    return mean(y .- K * (α .* y))
end

"""
    (svc::SVC)(X::Matrix{Float64})

Make predictions using the trained SVC model.

# Arguments
- `X::Matrix{Float64}`: Input features

# Returns
- `Vector{Int}`: Predicted class labels
"""
function (svc::SVC)(X::Matrix{Float64})
    if isnothing(svc.classes_)
        error("The model has not been fitted yet.")
    end
    
    if length(svc.classes_) == 2
        K = _compute_kernel(svc, X, svc.support_vectors_)
        # Ensure proper dimensions for matrix multiplication
        decision_values = K * svc.dual_coef_' .+ svc.intercept_[1]
        return [decision_value > 0 ? svc.classes_[2] : svc.classes_[1] for decision_value in decision_values]
    else
        n_samples = size(X, 1)
        votes = zeros(Int, n_samples, length(svc.classes_))
        
        for (i, classifier) in enumerate(svc.classifiers_)
            predictions = classifier(X)
            class_idx = [findfirst(==(pred), svc.classes_) for pred in predictions]
            votes[CartesianIndex.(1:n_samples, class_idx)] .+= 1
        end
        
        return svc.classes_[argmax.(eachrow(votes))]
    end
end

using  RDatasets, DataFrames
iris = dataset("datasets", "iris")
X = iris[:, 1:4] |> Matrix
y = iris.Species[:]
map_species = Dict(
    "setosa" => 0,
    "versicolor" => 1,
    "virginica" => 2
)
y = [map_species[k] for k in y]
using NovaML.ModelSelection: train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)

# Binary classification
svc = SVC(kernel=:rbf, C=1.0, gamma=:scale)
svc(X_train, y_train)  # Train the model
predictions = svc(X_test) |> vec # Make predictions 

using NovaML.Metrics: accuracy_score

accuracy_score(predictions, y_test)

# Multiclass classification
svc_multi = SVC(kernel=:linear, C=0.1)
svc_multi(X_train, y_train)  # Train the model
predictions_multi = svc_multi(X_test_multi)  # Make predictions

# Get decision function values
decision_values = decision_function(svc, X_test)
