begin
    using RDatasets, DataFrames
    iris = dataset("datasets", "iris")
    X = iris[51:150, 3:4] |> Matrix
    y = iris.Species[51:150]
    map_species = Dict(
        "setosa" => 0,
        "versicolor" => 1,
        "virginica" => 2
    )
    y = [map_species[k] for k in y]
    using NovaML.ModelSelection: train_test_split
    Xtrn, Xtst, ytrn, ytst = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y) 
end

using NovaML.SVM: SVC
svm = SVC()
@time svm(Xtrn, ytrn)
ŷ = svm(Xtst)
using NovaML.Metrics: accuracy_score
accuracy_score(ytst, ŷ)