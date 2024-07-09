# Building Good Training Datasets - Data Preprocessing
## Dealing with missing data 
### Identifying missing values in tabular data 
using Revise
using CSV, DataFrames
using Nova.Impute: SimpleImputer

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

"""Only drop rows where all columns are missing"""
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

r = [missing, 8.0, 6.0, 3.0]
push!(df, r, promote=true)

FillImputer = @load FillImputer pkg=MLJModels
imputer = FillImputer(
    continuous_fill = mean∘skipmissing,
    count_fill = mean∘skipmissing
)
mach = machine(imputer, df) |> fit!
MLJ.transform(mach, df)

df
impute = SimpleImputer(
    strategy = :mean
)

X = Matrix(df)
impute(X)

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
class_mapping = Dict(
    label => idx for (idx, label) in enumerate(unique(df.classlabel))
)

df.classlabel = [get(class_mapping, c, missing) for c in df.classlabel]
df

inv_class_mapping = Dict(
    v => k for (k, v) in class_mapping
)
df.classlabel = [get(inv_class_mapping, c, missing) for c in df.classlabel]
df

col = df.classlabel
labels = Dict(
    c => l for (l, c) ∈ enumerate(unique(col))
)
df

newcol = replace(col, labels...)

using Nova.PreProcessing: LabelEncoder
lblenc = LabelEncoder()
lblenc(df.size)
labels = ["M", "L", "XL", "M", "L", "M"]
lbls = lblenc(labels, :transform)
lblenc(lbls, :inverse_transform)

# One hot encoding
function col2_one_hot!(df, col)
    categories = unique(df[:, col])
    for category in categories
        onehot = zeros(Int8, size(df, 1))
        onehot[df[:, col] .== category] .= 1
        df[:, category] = onehot
    end
    select!(df, Not(col))
end
col2_one_hot!(df, :classlabel)
df

using Nova.PreProcessing: OneHotEncoder
ohe = OneHotEncoder()
ohe(df.size)
onehot = ohe(df.size, :transform)
ohe(onehot, :inverse_transform)

# Partitioning a dataset into separate training and test datasets 
using HTTP, Random

filelink = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
res = HTTP.get(filelink)
wine = CSV.read(res.body, DataFrame, header=false)
nms = [
    "Class label"
    "Alcohol"
 	"Malic acid"
 	"Ash"
	"Alcalinity of ash"  
 	"Magnesium"
	"Total phenols"
 	"Flavanoids"
 	"Nonflavanoid phenols"
 	"Proanthocyanins"
	"Color intensity"
 	"Hue"
 	"OD280/OD315 of diluted wines"
 	"Proline"
]

rename!(wine, nms)
wine
X, y = wine[:, 2:end], wine[:, 1]
X = Matrix(X)

using Nova.ModelSelection: train_test_split
Xtrn, Xtst, ytrn, ytst = train_test_split(X, y, 
                                          test_size=0.3, 
                                          random_state=0, 
                                          stratify=y)

using Nova.PreProcessing: MinMaxScaler
mmsc = MinMaxScaler()
mmsc(Xtrn)
Xtrn_norm = mmsc(Xtrn)
Xtst_norm = mmsc(Xtst)

using Nova.PreProcessing: StandardScaler
stdsc = StandardScaler()
stdsc(Xtrn)
Xtrn_std = stdsc(Xtrn)
Xtst_std = stdsc(Xtst)

# Selecting meaningful features
##Sparse solutions with L1 regularization
using Nova.MultiClass: OneVsRestClassifier
using Nova.LinearModel: LogisticRegression

ovr = OneVsRestClassifier()
ovr(Xtrn_std, ytrn)
ŷtrn = ovr(Xtrn_std)
ŷtst = ovr(Xtst_std)

using Nova.Metrics: accuracy_score
accuracy_score(ŷtrn, ytrn)
accuracy_score(ŷtst, ytst)

ovr.classifiers[1].w
[lr.b for lr in ovr.classifiers]

colors = ["blue", "green", "red", "cyan", "magenta",
          "yellow", "black", "pink", "lightgreen", "lightblue",
          "gray", "indigo", "orange"]
weights, params = [], []     

for λ ∈ -5:5
    ovr = OneVsRestClassifier(random_state=0, λ=10.0^λ)
    ovr(Xtrn_std, ytrn)
    push!(weights, ovr.classifiers[2].w)
    push!(params, 10.0^λ)
end

using Plots
plot()
for i in 1:length(weights)
    println(i)
    ys = [w[i] for w in weights]
    plot!(params, ys, label=nms[i+1], xaxis=:log)
end
plot!(legend=:bottomleft)

