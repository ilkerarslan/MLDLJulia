function ensemble_error(n_classifier::Int, ϵ::Float64)
    k_start = ceil(Int, n_classifier/2)
    probs = [binomial(n_classifier, k)*ϵ^k*(1-ϵ)^(n_classifier-k)
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

# Define the input array
ex = [0.9 0.1;
      0.8 0.2;
      0.4 0.6]

# Define weights
w = [0.2, 0.2, 0.6]

# Calculate weighted average
p = mean(ex, Weights(w), dims=1)
argmax(vec(p))


# Combining classifiers via majority vote
using Revise
using NovaML.Datasets
using NovaML.Ensemble
using NovaML.LinearModel
using NovaML.Tree
using NovaML.Neighbors
using NovaML.PreProcessing
using NovaML.ModelSelection
using NovaML.Pipelines

# Assume we have X_train, y_train, X_test, y_test
iris = load_iris()
X = iris["data"][51:150, [2,3]] 
y = (iris["target"][51:150] .== 2) .|> Int 

Xtrn, Xtst, ytrn, ytst = train_test_split(X, y, 
                                          test_size=0.5, 
                                          random_state=1,
                                          stratify=y)
# Create base classifiers
clf1 = LogisticRegression()
clf2 = DecisionTreeClassifier(max_depth=1, random_state=0)
clf3 = KNeighborsClassifier(n_neighbors=1)

# Create StandardScaler
sc = StandardScaler()

# Create pipelines
pipe1 = pipe(sc, clf1)
pipe3 = pipe(sc, clf3)

# Create VotingClassifier
vc = VotingClassifier(
    estimators=[("lr", pipe1), ("dt", clf2), ("knn", pipe3)],
    voting=:soft
)

# Fit the VotingClassifier
vc(Xtrn, ytrn)

# Make predictions
ŷ = vc(Xtst)

using NovaML.Metrics: accuracy_score, roc_auc_score
accuracy_score(ytst, ŷ)

# Get probability estimates
ŷprobs = vc(Xtst, type=:probs)
roc_auc_score(ytst, ŷprobs)


for (name, clf) in [("lr", pipe1), ("dt", clf2), ("knn", pipe3)]
    clf(Xtrn, ytrn)
    ŷ_individual = clf(Xtst)
    acc = accuracy_score(ytst, ŷ_individual)
    println("$name accuracy: $acc")
    
    if :probs in propertynames(clf)
        ŷprobs_individual = clf(Xtst, type=:probs)
        auc = roc_auc_score(ytst, ŷprobs_individual, multiclass=:ovr)
        println("$name AUC: $auc")
    end
    println()
end