module CHAPTER03

using Plots, MLJ, Random, ProgressBars

export plot_decision_regions

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