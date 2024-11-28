function ensemble_error(n_classifier::Int, ϵ::Float64)
    k_start = ceil(Int, n_classifier / 2)
    probs = [binomial(n_classifier, k) * ϵ^k * (1 - ϵ)^(n_classifier - k)
             for k ∈ k_start:n_classifier]
    return sum(probs)
end

ensemble_error(11, 0.25)

error_range = 0.0:0.01:1.0
ens_errors = [ensemble_error(11, ϵ) for ϵ ∈ error_range]

using Plots
begin
    plot(error_range, ens_errors, label="Ensemble error", linewidth=2)
    plot!(error_range, error_range, linestyle=:dash, label="Base Error", linewidth=2)
    xlabel!("Base error")
    ylabel!("Base/Ensemble Error")
end

using StatsBase, Statistics
arr = [0, 0, 1]
w = [0.2, 0.2, 0.6]
argmax(countmap(arr, w))

using Statistics, StatsBase
ex = [0.9 0.1;
    0.8 0.2;
    0.4 0.6]

w = [0.2, 0.2, 0.6]

p = mean(ex, Weights(w), dims=1)
argmax(vec(p))

# Combining classifiers via majority vote
using Revise

# Assume we have X_train, y_train, X_test, y_test
using NovaML.Datasets: load_iris
iris = load_iris()
X = iris["data"][51:150, [2, 3]]
y = (iris["target"][51:150] .== 2) .|> Int

using NovaML.ModelSelection: train_test_split
Xtrn, Xtst, ytrn, ytst = train_test_split(X, y,
    test_size=0.5,
    random_state=1,
    stratify=y)

# Create base classifiers
using NovaML.LinearModel
using NovaML.Tree
using NovaML.Neighbors

clf1 = LogisticRegression(random_state=1)
clf2 = DecisionTreeClassifier(max_depth=1, random_state=0)
clf3 = KNeighborsClassifier(n_neighbors=1)

# Create StandardScaler
using NovaML.PreProcessing
sc = StandardScaler()

# Create pipelines
using NovaML.Pipelines: pipe
pipe1 = pipe(sc, clf1)
pipe3 = pipe(sc, clf3)

# Create VotingClassifier
using NovaML.Ensemble
vc = VotingClassifier(
    estimators=[("lr", pipe1), ("dt", clf2), ("knn", pipe3)],
    voting=:soft)

vc(Xtrn, ytrn)
ŷ = vc(Xtst)

using NovaML.Metrics: accuracy_score, roc_auc_score
accuracy_score(ytst, ŷ)

# Get probability estimates
ŷprobs = vc(Xtst, type=:probs)
roc_auc_score(ytst, ŷprobs)

begin
    using Statistics
    using NovaML.ModelSelection: cross_val_score
    clf_labels = ["Logistic regression", "Decision tree", "KNN"]
    println("10-fold cross validation")

    for (clf, label) ∈ zip([pipe1, clf2, pipe3], clf_labels)
        scores = cross_val_score(
            clf, Xtrn, ytrn,
            cv=10,
            scoring=roc_auc_score
        )
        println("ROC AUC: $(round(mean(scores), digits=2)) (±$(round(std(scores), digits=2))) [$label]")
    end
end

vc = VotingClassifier(
    estimators=[("lr", pipe1), ("dt", clf2), ("knn", pipe3)],
    voting=:soft
)

push!(clf_labels, "Majority voting")
all_clf = [pipe1, clf2, pipe3, vc]
for (clf, label) ∈ zip(all_clf, clf_labels)
    scores = cross_val_score(
        clf, Xtrn, ytrn, cv=10,
        scoring=roc_auc_score
    )
    mn, st = mean(scores), std(scores)
    println("ROC AUC: $(round(mn,digits=2)) (±$(round(st, digits=2))) [$label]")
end

# Evaluating and tuning the the ensemble classifier
using NovaML.Metrics: auc, roc_curve
using Plots

colors = [:black, :orange, :blue, :green]
linestyles = [:solid, :dash, :dashdot, :dot, :dashdotdot]

p = plot(xlabel="False Positive Rate", ylabel="True Positive Rate",
    title="Receiver Operating Characteristic (ROC) Curve",
    legend=:bottomright);

for (clf, label, clr, ls) ∈ zip(all_clf, clf_labels, colors, linestyles)
    clf(Xtrn, ytrn)
    ŷ = clf(Xtst, type=:probs)[:, 2]
    fpr, tpr, _ = roc_curve(ytst, ŷ)
    roc_auc = auc(fpr, tpr)
    plot!(p, fpr, tpr, color=clr, linestyle=ls, label="$label (auc = $(round(roc_auc, digits=2)))")
end

plot!(p, [0, 1], [0, 1], color=:gray, linestyle=:dash, linewidth=2, label="Random");
display(p)

begin
    sc = StandardScaler()
    Xtrnstd = sc(Xtrn)
    Xtststd = sc(Xtst)

    # Set up the plot
    len = 300

    p = plot(layout=(2, 2), size=(800, 600))

    x1min, x1max = minimum(Xtrnstd[:, 1]) - 1, maximum(Xtrnstd[:, 1]) + 1
    x2min, x2max = minimum(Xtrnstd[:, 2]) - 1, maximum(Xtrnstd[:, 2]) + 1
    x1range = range(x1min, x1max, length=len)
    x2range = range(x2min, x2max, length=len)

    # Plot for each classifier
    for (i, model, tt) in zip(1:4, all_clf, clf_labels)
        # Train the model
        model(Xtrnstd, ytrn)

        # Create the decision boundary
        z = [model([x1 x2])[1] for x2 in x2range, x1 in x1range]

        # Plot
        contourf!(p[i], x1range, x2range, z,
            colorbar=false, color=[:red, :lightblue], alpha=0.25)
        scatter!(p[i], Xtrnstd[ytrn.==0, 1], Xtrnstd[ytrn.==0, 2],
            color=:blue, marker=:utriangle, label="Class 0")
        scatter!(p[i], Xtrnstd[ytrn.==1, 1], Xtrnstd[ytrn.==1, 2],
            color=:green, marker=:circle, label="Class 1")
        plot!(p[i], title=tt, legend=false)
    end
    # Display the plot
    display(p)
end

# Bagging Classifier
using NovaML.Datasets
wine = load_wine()
X, y = wine["data"], wine["target"]
X = X[:, [1, end - 1]]
idx = y .!= 1
X, y = X[idx, :], y[idx]

using NovaML.PreProcessing: LabelEncoder
le = LabelEncoder()
y = le(y)

using NovaML.ModelSelection: train_test_split
Xtrn, Xtst, ytrn, ytst = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

using NovaML.Tree: DecisionTreeClassifier
using NovaML.Ensemble: BaggingClassifier
tree = DecisionTreeClassifier(random_state=1)
bag = BaggingClassifier(
    n_estimators=1000,
    max_samples=1.0,
    max_features=1.0,
    bootstrap=true,
    bootstrap_features=false,
    random_state=1)

tree(Xtrn, ytrn)
ŷtrn = tree(Xtrn)
ŷtst = tree(Xtst)
acctrn = accuracy_score(ytrn, ŷtrn)
acctst = accuracy_score(ytst, ŷtst)

println("Decision tree accuracies:
Training set: $acctrn
testing set : $acctst")

bag(Xtrn, ytrn)
ŷtrn = bag(Xtrn)
ŷtst = bag(Xtst)
acctrn = accuracy_score(ytrn, ŷtrn)
acctst = accuracy_score(ytst, ŷtst)

println("Decision tree accuracies:
Training set: $acctrn
testing set : $acctst")

using Plots
begin
    len = 300
    p = plot(layout=(1, 2), size=(800, 600), xlabel="od280_od315_of_diluted_wines", ylabel="alcohol")

    x1min, x1max = minimum(Xtrn[:, 1]) - 1, maximum(Xtrn[:, 1]) + 1
    x2min, x2max = minimum(Xtrn[:, 2]) - 1, maximum(Xtrn[:, 2]) + 1
    x1range = range(x1min, x1max, length=len)
    x2range = range(x2min, x2max, length=len)

    # Plot for each classifier
    for (i, model, tt) in zip(1:2, [tree, bag], ["Decision tree", "Bagging"])
        # Train the model
        model(Xtrn, ytrn)

        # Create the decision boundary
        z = [model([x1 x2])[1] for x2 in x2range, x1 in x1range]

        # Plot
        contourf!(p[i], x1range, x2range, z,
            colorbar=false, color=[:red, :lightblue], alpha=0.25)
        scatter!(p[i], Xtrn[ytrn.==0, 1], Xtrn[ytrn.==0, 2],
            color=:blue, marker=:utriangle, label="Class 0")
        scatter!(p[i], Xtrn[ytrn.==1, 1], Xtrn[ytrn.==1, 2],
            color=:green, marker=:circle, label="Class 1")
        plot!(p[i], title=tt, legend=false)
    end
    display(p)
end

ŷ = tree(Xtst, type=:probs)[:, 2]

using NovaML.Metrics: auc, roc_curve
fpr, tpr, _ = roc_curve(ytst, ŷ)
roc_auc = auc(fpr, tpr)

ŷ = bag(Xtst, type=:probs)[:, 2]
fpr, tpr, _ = roc_curve(ytst, ŷ)
roc_auc = auc(fpr, tpr)

begin
    colors = [:black, :orange, :blue, :green]
    linestyles = [:solid, :dash, :dashdot, :dot, :dashdotdot]

    p = plot(xlabel="False Positive Rate", ylabel="True Positive Rate",
        title="Receiver Operating Characteristic (ROC) Curve",
        legend=:bottomright)
    models = [tree, bag]
    linestyles = [:solid, :dash]
    labels = [:tree, :bag]
    p = plot(xlabel="False Positive Rate", ylabel="True Positive Rate",
        title="Receiver Operating characteristic (ROC) Curve",
        legend=:bottomright)
    for (clf, lbl, ls) ∈ zip(models, labels, linestyles)
        clf(Xtrn, ytrn)
        ŷ = clf(Xtst, type=:probs)[:, 2]
        fpr, tpr, _ = roc_curve(ytst, ŷ)
        roc_auc = auc(fpr, tpr)
        plot!(p, fpr, tpr, linestyle=ls, label="$lbl (AUC:$(round(roc_auc, digits=2)))")
    end
    plot!([0, 1], [0, 1], color=:black, linewidth=2, linestyle=:dash, label=nothing)
    display(p)
end

# Adaptive Boosting
using Statistics
y = Int[1, 1, 1, -1, -1, -1, 1, 1, 1, -1]
ŷ = Int[1, 1, 1, -1, -1, -1, -1, -1, -1, -1]
correct = (y .== ŷ)
weights = fill(0.1, 10)
ε = mean(.!correct)
αⱼ = 0.5 * log((1 - ε) / ε)
w_correct = 0.1 * exp(-αⱼ * 1 * 1)
w_wrong = 01 * exp(-αⱼ * -1 * 1)
weights = ifelse.(correct .== 1, w_correct, w_wrong)
weights ./= sum(weights)

using NovaML.Ensemble: AdaBoostClassifier
using NovaML.Metrics: accuracy_score

tree = DecisionTreeClassifier(
    random_state=1,
    max_depth=1)

tree(Xtrn, ytrn)
ŷtrn = tree(Xtrn)
ŷtst = tree(Xtst)
treetrn = accuracy_score(ytrn, ŷtrn)
treetst = accuracy_score(ytst, ŷtst)

ada = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=500,
    learning_rate=0.1,
    random_state=1
)
ada(Xtrn, ytrn)
ŷtrn = ada(Xtrn)
ŷtst = ada(Xtst)
accuracy_score(ytrn, ŷtrn)
accuracy_score(ytst, ŷtst)

begin
    len = 300
    p = plot(layout=(1, 2), size=(800, 600), xlabel="od280_od315_of_diluted_wines", ylabel="alcohol")

    x1min, x1max = minimum(Xtrn[:, 1]) - 1, maximum(Xtrn[:, 1]) + 1
    x2min, x2max = minimum(Xtrn[:, 2]) - 1, maximum(Xtrn[:, 2]) + 1
    x1range = range(x1min, x1max, length=len)
    x2range = range(x2min, x2max, length=len)

    # Plot for each classifier
    for (i, model, tt) in zip(1:2, [tree, ada], ["Decision tree", "AdaBoost"])
        # Train the model
        model(Xtrn, ytrn)

        # Create the decision boundary
        z = [model([x1 x2])[1] for x2 in x2range, x1 in x1range]

        # Plot
        contourf!(p[i], x1range, x2range, z,
            colorbar=false, color=[:red, :lightblue], alpha=0.25)
        scatter!(p[i], Xtrn[ytrn.==0, 1], Xtrn[ytrn.==0, 2],
            color=:blue, marker=:utriangle, label="Class 0")
        scatter!(p[i], Xtrn[ytrn.==1, 1], Xtrn[ytrn.==1, 2],
            color=:green, marker=:circle, label="Class 1")
        plot!(p[i], title=tt, legend=false)
    end
    display(p)
end

# DecisionTreeRegression
using NovaML.Datasets
boston = load_boston()
X, y = boston["data"], boston["target"]
using NovaML.Tree
tree = DecisionTreeRegressor()
tree(X, y)
ŷ = tree(X)
using NovaML.Metrics: r2_score, adj_r2_score, mse
r2_score(y, ŷ)
adj_r2_score(y, ŷ, n_features=size(X, 2))
mse(y, ŷ)

# Gradient Boosting
using NovaML.Ensemble: GradientBoostingClassifier
gbc = GradientBoostingClassifier(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=4,
    random_state=1)
gbc(Xtrn, ytrn)
ŷtrn = gbc(Xtrn)
ŷtst = gbc(Xtst)

using NovaML.Metrics: accuracy_score, auc, roc_curve
accuracy_score(ytrn, ŷtrn)
accuracy_score(ytst, ŷtst)

ŷtrn = gbc(Xtrn, type=:probs)
fpr, tpr, _ = roc_curve(ytrn, ŷtrn)
roc_auc = auc(fpr, tpr)

ŷtst = gbc(Xtst, type=:probs)
fpr, tpr, _ = roc_curve(ytst, ŷtst)
roc_auc = auc(fpr, tpr)
