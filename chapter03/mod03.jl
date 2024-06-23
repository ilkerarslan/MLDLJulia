module CHAPTER03

using Plots
using StatsBase: countmap

export plot_decision_regions

function plot_decision_regions(X::Matrix, y::Vector, classifier;
                               test_idx=nothing, resolution=0.02)
    # Setup marker generator and color map
    markers = [:circle, :rect, :utriangle, :dtriangle, :diamond]
    colors = [:red, :blue, :lightgreen, :gray, :cyan]
    
    # Plot the decision surface
    x1_min, x1_max = minimum(X[:, 1]) - 1, maximum(X[:, 1]) + 1
    x2_min, x2_max = minimum(X[:, 2]) - 1, maximum(X[:, 2]) + 1
    xx1 = range(x1_min, x1_max, step=resolution)
    xx2 = range(x2_min, x2_max, step=resolution)
    
    Z = [predict(classifier, [x1, x2]) for x1 in xx1, x2 in xx2]
    
    p = contourf(xx1, xx2, Z, color=:RdYlBu, alpha=0.3)
    
    # Plot class examples
    unique_labels = unique(y)
    for (idx, cl) in enumerate(unique_labels)
        indices = findall(y .== cl)
        scatter!(X[indices, 1], X[indices, 2], 
                 color=colors[idx], marker=markers[idx], 
                 label="Class $cl", markerstrokecolor=:black)
    end
    
    # Highlight test examples
    if !isnothing(test_idx)
        X_test = X[test_idx, :]
        scatter!(X_test[:, 1], X_test[:, 2],
                 color=:white, marker=:circle, 
                 markersize=10, markerstrokecolor=:black, 
                 label="Test set")
    end
    
    xlabel!("Feature 1")
    ylabel!("Feature 2")
    title!("Decision Regions")
    
    return p
end
    
end