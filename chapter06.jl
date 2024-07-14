using Revise
using HTTP, CSV, DataFrames
using Statistics, LinearAlgebra

filelink = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data";

data = HTTP.get(filelink)
df = CSV.read(data.body, DataFrame, header=false)

using NovaML.PreProcessing: LabelEncoder
X = df[:, 3:end] |> Matrix
y = df[:, 2]
le = LabelEncoder()
y = le(y)
le.classes
le.fitted
le(["M", "B"])
v = [1, 0, 0, 1, 1] 
le(v, :inverse_transform)

using NovaML.ModelSelection: train_test_split
Xtrn, Xtst, ytrn, ytst = train_test_split(X, y, 
                            test_size=0.20,
                            stratify=y,
                            random_state=1)

using NovaML.PreProcessing: StandardScaler
using NovaML.Decomposition: PCA 
using NovaML.LinearModel: LogisticRegression

sc = StandardScaler()
pca = PCA(n_components=2)
lr = LogisticRegression()
lr(Xtrn, ytrn)
using NovaML.Metrics: accuracy_score

ŷtrn = lr(Xtrn)
ŷtst = lr(Xtst)
accuracy_score(ŷtrn, ytrn)
accuracy_score(ŷtst, ytst)

Xtrn |> sc |> pca |> X -> lr(X, ytrn)
ŷtst = Xtst |> sc |> pca |> lr

accuracy_score(ytst, ŷtst)

using NovaML.ModelSelection: StratifiedKFold

kfold = StratifiedKFold(n_splits=10)
scores = []

for (i, (trn, tst)) in enumerate(kfold(y))
    sc = StandardScaler()
    pca = PCA(n_components=2)
    lr = LogisticRegression()
    xtrain, ytrain = X[trn, :], y[trn]
    xtest, ytest = X[tst, :], y[tst]
    xtrain |> sc |> pca |> x -> lr(x, ytrain);
    ŷtest = xtest |> sc |> pca |> lr
    score = accuracy_score(ŷtest, ytest)    
    push!(scores, score)    
end

scores
mean_acc, std_acc = mean(scores), std(scores);
println("CV accuracy: $mean_acc ± $std_acc")

using NovaML.Pipelines: Pipe

pipe = Pipe(
    StandardScaler(),
    PCA(n_components=2),
    LogisticRegression())
pipe(Xtrn, ytrn)
ŷ = pipe(Xtst)
accuracy_score(ytst, ŷ)

using NovaML.ModelSelection: cross_val_score

lr = LogisticRegression()

println("Accuracy: $(mean(scores)) ± $(std(scores))")

using Plots
using NovaML.ModelSelection: learning_curve
pipe = Pipe(StandardScaler(), LogisticRegression(max_iter=10_000))

trn_sizes, trn_scores, tst_scores = learning_curve(
    pipe, Xtrn, ytrn, 
    train_sizes = range(0.1, 1.0, length=10),
    cv = 10)

trn_mean = mean.(trn_scores)
trn_std = std.(trn_scores)
tst_mean = mean.(tst_scores)
tst_std = std.(tst_scores)

begin
    plot(trn_sizes, trn_mean, 
    color=:blue, marker=:circle, markersize=5, 
    label="Training accuracy", linewidth=2)

    plot!(trn_sizes, tst_mean, color=:green, 
          linestyle=:dash, marker=:square, markersize=5, 
          label="Validation accuracy", linewidth=2)

    # Add shaded areas for standard deviation
    plot!(trn_sizes, trn_mean + trn_std, 
         fillrange=trn_mean - trn_std, fillalpha=0.15, 
         color=:blue, label=nothing, legend=:bottomright)

    plot!(trn_sizes, tst_mean + tst_std, 
         fillrange=tst_mean - tst_std, fillalpha=0.15, 
         color=:green, label=nothing)

    # Customize the plot
    xlabel!("Number of training examples")
    ylabel!("Accuracy")
    title!("Learning Curve")
    ylims!(0.8, 1.03)
    xgrid!(true)
    ygrid!(true)   
end

