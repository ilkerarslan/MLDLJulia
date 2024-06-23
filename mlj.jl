using MLJ
iris = load_iris()
selectrows(iris, 1:3) |> pretty
schema(iris)

import DataFrames
iris = DataFrames.DataFrame(iris)
y, X = unpack(iris, ==(:target); rng=123)
first(X, 3) |> pretty
models()
doc("DecisionTreeClassifier", pkg="DecisionTree")

# Pkg.add("MLJDecisionTreeInterface)
Tree = @load DecisionTreeClassifier pkg=DecisionTree
tree = Tree()

evaluate(tree, X, y,
         resampling=CV(shuffle=true),
         measures=[log_loss, accuracy],
         verbosity=0)

typeof(y)
target_scitype(tree)
scitype(y)
yint = int.(y)
scitype(yint)
mach = machine(tree, X, y)
train, test = partition(eachindex(y), 0.7)
fit!(mach, rows=train)
yhat = predict(mach, X[test, :])
yhat[3:5]
log_loss(yhat, y[test])

broadcast(pdf, yhat[3:5], "virginica")
broadcast(pdf, yhat, y[test])[3:5]
mode.(yhat[3:5])
predict_mode(mach, X[test[3:5], :])
L = levels(y)
pdf(yhat[3:5], L)

v = Float64[1, 2, 3, 4]
stand = Standardizer()
mach2 = machine(stand, v)
fit!(mach2)
w = transform(mach2, v)
inverse_transform(mach2, w)

evaluate!(mach, resampling=Holdout(fraction_train=0.7),
          measures=[log_loss, accuracy],
          verbosity=0)
tree.max_depth = 3;
evaluate!(mach, resampling=Holdout(fraction_train=0.7),
          measures=[log_loss, accuracy],
          verbosity=0)

scitype(4.6)
scitype(42)
x1 = coerce(["yes", "no", "yes", "maybe"], Multiclass)
scitype(x1)
X = (x1=x1, x2=rand(4), x3=rand(4))
scitype(X)
schema(X)

i = info("DecisionTreeClassifier", pkg="DecisionTree")
i.input_scitype
i.target_scitype
MLJ_VERSION

models("Tree")
models("Perceptron")