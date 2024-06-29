# Building Good Training Datasets - Data Preprocessing
## Dealing with missing data 
### Identifying missing values in tabular data 
using CSV, DataFrames

csv_data = """
A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0"""

df = CSV.read(IOBuffer(csv_data), DataFrame)
describe(df, :nmissing)

### Eliminating training examples or features with missing values 
dropmissing(df)

function remove_cols_with_missing(df)
    desc = describe(df, :nmissing)
    cols = desc[desc.nmissing .> 0, :variable]
    return df[:, Not(cols)]            
end

remove_cols_with_missing(df)

function remove_all_missing_cols(df)
    desc = describe(df, :nmissing)
    cols = desc[desc.nmissing .== size(df, 1), :variable]
    return df[:, Not(cols)]
end

df[!, :E] = [missing, missing, missing]
df = remove_all_missing_cols(df)

"""drop_rows
Drop rows with less than n nonmissing values
"""
function drop_rows(df, n=4)
    rows = Int[]
    for i in 1:size(df, 1)
        r = collect(df[i, :]) .|> !ismissing |> sum
        if r ≥ n push!(rows, i) end
    end
    return df[rows, :]
end

drop_rows(df, 4)

# only drop rows where missing appear in specific columns (here: :C)
dropmissing(df, :C)

# Imputing missing values 
using MLJ, Statistics
imputer = FillImputer()
mach = machine(imputer, df) |> fit!
MLJ.transform(mach, df)

# Categorical data encoding
df = DataFrame(
    color = ["green", "red", "blue"],
    size = ["M", "L", "XL"],
    price = [10.1, 13.5, 15.3],
    classlabel = ["class2", "class1", "class2"]
)

# Mapping ordinal features 
size_mapping = Dict(
    "XL" => 3,
    "L" => 2,
    "M" => 1
)
df.size = [get(size_mapping, s, missing) for s in df.size]
df

inv_size_mapping = Dict(v => k for (k, v) in size_mapping)  
df.size = [get(inv_size_mapping, s, missing) for s in df.size]

# Encoding class labels