module CHAPTER04 

module PreProcessing

export fit!, transform, fit_transform!, inverse_transform
    
mutable struct MinMaxScaler
    min_vals::Vector{Float64}
    max_vals::Vector{Float64}
    feature_range::Tuple{Float64, Float64}

    function MinMaxScaler(feature_range::Tuple{Float64, Float64}=(0.0, 1.0))
        new([], [], feature_range)
    end
end

end # of module Preprocessing

end

end # of module chapter04