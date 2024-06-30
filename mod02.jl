module CHAPTER02
using Random, ProgressBars, Plots, Statistics, LinearAlgebra
export plot_decision_region

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