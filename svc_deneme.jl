
begin
    using  RDatasets, DataFrames
    iris = dataset("datasets", "iris")
    X = iris[51:150, 1:4] |> Matrix
    y = iris.Species[51:150]
    map_species = Dict(
        "setosa" => 0,
        "versicolor" => 1,
        "virginica" => 2
    )
    y = [map_species[k] for k in y]
    using NovaML.ModelSelection: train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)
end

using NovaML.SVM: SVC

# Create an SVC instance
svc = SVC(kernel=:rbf, C=1.0, gamma=:scale)

# Train the model
svc(X_train, y_train)

# Make predictions
ŷtst = svc(X_test);
ŷtrn = svc(X_train);
using NovaML.Metrics: accuracy_score
accuracy_score(y_train, ŷtrn)
accuracy_score(y_test, ŷtst)