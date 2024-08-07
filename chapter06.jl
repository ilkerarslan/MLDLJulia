using Revise

using NovaML.Datasets: load_breast_cancer
X, y = load_breast_cancer(return_X_y=true)

using NovaML.ModelSelection: train_test_split
Xtrn, Xtst, ytrn, ytst = train_test_split(X, y, test_size=0.2,
                                          stratify=y, random_state=1)

using NovaML.PreProcessing: StandardScaler
using NovaML.Decomposition: PCA 
using NovaML.LinearModel: LogisticRegression

sc = StandardScaler()
pca = PCA(n_components=2)
lr = LogisticRegression()
lr(Xtrn, ytrn)

using NovaML.Metrics: accuracy_score
ŷtrn, ŷtst = lr(Xtrn), lr(Xtst)
accuracy_score(ŷtrn, ytrn), accuracy_score(ŷtst, ytst)

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

using Statistics, LinearAlgebra
mean_acc, std_acc = mean(scores), std(scores);
println("CV accuracy: $mean_acc ± $std_acc")

using NovaML.Pipelines: pipe
p = pipe(
    StandardScaler(),
    PCA(n_components=2),
    LogisticRegression())

p(Xtrn, ytrn)
ŷ = p(Xtst)
accuracy_score(ytst, ŷ)

using NovaML.ModelSelection: cross_val_score
lr = LogisticRegression()

println("Accuracy: $(mean(scores)) ± $(std(scores))")

using Plots
using NovaML.ModelSelection: learning_curve
p = pipe(StandardScaler(), LogisticRegression(max_iter=10_000))

trn_sizes, trn_scores, tst_scores = learning_curve(
    p, Xtrn, ytrn, 
    train_sizes = range(0.1, 1.0, length=10),
    cv = 10);

trn_mean, trn_std = mean.(trn_scores), std.(trn_scores)
tst_mean, tst_std = mean.(tst_scores), std.(tst_scores)

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

using NovaML.ModelSelection: validation_curve

# Define the parameter range
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

# Create a pipeline with StandardScaler and LogisticRegression
model = pipe(
    StandardScaler(),
    LogisticRegression(num_iter=1000))

# Perform validation curve analysis
train_scores, test_scores = validation_curve(
    model,
    Xtrn,
    ytrn,
    param_dict=Dict(:LogisticRegression => :λ),
    param_range=param_range,
    cv=10,
    scoring="accuracy")

train_mean = mean(train_scores, dims=2)
train_std = std(train_scores, dims=2)
test_mean = mean(test_scores, dims=2)
test_std = std(test_scores, dims=2)

begin
    plot(param_range, train_mean, yerr=train_std,
         color=:blue, marker=:circle, markersize=5, 
         label="Training accuracy", xscale=:log10)

    plot!(param_range, test_mean, yerr=test_std,
          color=:green, linestyle=:dash,
          marker=:square, markersize=5, 
          label="Validation accuracy")

    # Add error bands
    plot!(param_range, train_mean + train_std, 
          fillrange=train_mean - train_std, 
          alpha=0.15, color=:blue, label=nothing)

    plot!(param_range, test_mean + test_std, 
          fillrange=test_mean - test_std, 
          alpha=0.15, color=:green, label=nothing)

    # Customize the plot
    xlabel!("Parameter λ")
    ylabel!("Accuracy")
    title!("Validation Curve")
    ylims!(0.5, 1.0)
    xgrid!(true)
    plot!(legend=:bottomleft)    
end

scaler = StandardScaler()
Xtrnstd = scaler(Xtrn)
Xtststd = scaler(Xtst)

using NovaML.SVM: SVC
svm = SVC(kernel=:rbf, C=1.0, gamma=:scale)
svm(X, y)
ŷ = svm(Xtststd)

using NovaML.Metrics: accuracy_score
accuracy_score(ŷ, ytst)
ŷtrn = svm(Xtrnstd)
accuracy_score(ŷtrn, ytrn)

# GridSearchCV
using NovaML.ModelSelection: GridSearchCV

using NovaML.Metrics: accuracy_score

scaler = StandardScaler()
svc = SVC()
pipe_svc = pipe(scaler, svc)
param_range = [100.0, 1000.0]
param_grid = [
    [svc, (:C, param_range), (:kernel, [:linear])],
    [svc, (:C, param_range), (:gamma, param_range), (:kernel, [:rbf])]
]

gs = GridSearchCV(
    estimator=pipe_svc,
    param_grid=param_grid,
    scoring=accuracy_score,
    cv=5,
    refit=true)

gs(Xtrn, ytrn)

gs.best_score
bp = gs.best_params
bp[1]  # best model
bp[2:end] # best parameters
clf  = gs.best_params[1]

# RandomSearchCV
using NovaML.ModelSelection: RandomSearchCV
svc = SVC()
sc = StandardScaler()
pipe_svc = pipe(sc, svc)

param_range = [100.0, 1000.0]
param_grid = [
    [svc, (:C, param_range), (:kernel, [:linear])],
    [svc, (:C, param_range), (:gamma, param_range), (:kernel, [:rbf])]
]

rs = RandomSearchCV(
    estimator=pipe_svc,
    param_grid = param_grid,
    scoring=accuracy_score,
    cv=5,
    refit=true,
    n_iter=10,
    random_state=1
)

rs(Xtrn, ytrn)
rs.best_score
rs.best_params
rs.best_params[1]
rs.best_params[2:end]

# Add nested cross-validation with grid search cv

# Confusion matrix
using NovaML.Metrics: confusion_matrix, display_confusion_matrix, accuracy_score

svc = SVC()
sc = StandardScaler()
pipe_svc = pipe(sc, svc)
@time pipe_svc(Xtrn, ytrn)
ŷ = pipe_svc(Xtst)
confmat = confusion_matrix(ytst, ŷ)
display_confusion_matrix(confmat)
accuracy_score(ytst, ŷ)

using Plots
using Plots.PlotMeasures

function plot_confusion_matrix(confmat::Matrix)
    n = size(confmat, 1)
    
    # Create heatmap
    heatmap(confmat, 
            c=:Blues, 
            alpha=0.3, 
            aspect_ratio=:equal, 
            size=(300, 300),
            xrotation=0,
            xticks=1:n, 
            yticks=1:n,
            xlims=(0.5, n+0.5), 
            ylims=(0.5, n+0.5),
            right_margin=5mm,
            xlabel="Predicted label",
            ylabel="True label",
            xmirror=true, # Mirror xticks to the top
            framestyle=:box, # Add lines to all sides of the plot
            legend=nothing)
    
    # Add text annotations
    for i in 1:n, j in 1:n
        annotate!(j, i, text(string(confmat[i,j]), :center, 10))
    end
    
    # Flip y-axis to match Python's matshow
    plot!(yflip=true)
    
    # Display the plot
    display(current())
end

plot_confusion_matrix(confmat)

using NovaML.Metrics: precision_score, recall_score, f1_score, matthews_corrcoef
precision_score(ytst, ŷ)
recall_score(ytst, ŷ)
f1_score(ytst, ŷ)
matthews_corrcoef(ytst, ŷ)

####################################################
using NovaML.Metrics: roc_curve, auc
using Statistics: mean, std
using Plots

# Create a pipeline with StandardScaler and LogisticRegression
clf = pipe(StandardScaler(), 
           PCA(n_components=2),
           LogisticRegression(solver=:lbfgs, λ=0.01, random_state=1))

clf(Xtrn, ytrn)
ŷprobs = clf(Xtst, type=:probs)

# Initialize variables to store results
cv = StratifiedKFold(n_splits=3, shuffle=true, random_state=42)
tprs = []
aucs = []
mean_fpr = range(0, 1, length=100)

# Create a plot
p = plot(xlabel="False Positive Rate", ylabel="True Positive Rate",
         title="Receiver Operating Characteristic (ROC) Curve",
         legend=:bottomright);

# Custom linear interpolation function
function linear_interp(x, x_values, y_values)
    i = searchsortedfirst(x_values, x)
    if i > length(x_values)
        return y_values[end]
    elseif i == 1
        return y_values[1]
    else
        x0, x1 = x_values[i-1], x_values[i]
        y0, y1 = y_values[i-1], y_values[i]
        return y0 + (x - x0) * (y1 - y0) / (x1 - x0)
    end
end

# Perform cross-validation and plot ROC curves
for (i, (train, test)) in enumerate(cv(y))
    clf(X[train, :], y[train])
    y_score = clf(X[test, :], type=:probs)[:, 2]
    
    fpr, tpr, _ = roc_curve(y[test], y_score)
    roc_auc = auc(fpr, tpr)
    push!(aucs, roc_auc)
    
    plot!(p, fpr, tpr, alpha=0.3, label="ROC fold $i (AUC = $(round(roc_auc, digits=2)))")
    
    interp_tpr = [linear_interp(x, fpr, tpr) for x in mean_fpr]
    interp_tpr[1] = 0.0
    push!(tprs, interp_tpr)
end

# Plot the mean ROC curve
mean_tpr = mean(tprs)
mean_tpr[1] = 0.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = std(aucs)
plot!(p, mean_fpr, mean_tpr, color=:blue, linewidth=2,
      label="Mean ROC (AUC = $(round(mean_auc, digits=2)) ± $(round(std_auc, digits=2)))");

# Plot the diagonal line
plot!(p, [0, 1], [0, 1], color=:red, linestyle=:dash, label="Random");

# Display the plot
display(p)

# Resampling
# Create imbalanced dataset
X = rand(500, 2)
y = vcat(ones(Int, 450), zeros(Int, 50))

# Print initial class distribution
println("Number of class 1 examples before: ", sum(y .== 1))
println("Number of class 0 examples before: ", sum(y .== 0))

# Upsample minority class
X_minority = X[y .== 0, :]
y_minority = y[y .== 0]

using NovaML.Utils: resample
X_upsampled, y_upsampled = resample(X_minority, y_minority, 
                                    replace=true, 
                                    n_samples=sum(y .== 1), 
                                    random_state=123)

# Combine upsampled minority class with majority class
X_balanced = vcat(X[y .== 1, :], X_upsampled)
y_balanced = vcat(y[y .== 1], y_upsampled)

# Print final class distribution
println("Number of class 1 examples after: ", sum(y_balanced .== 1))
println("Number of class 0 examples after: ", sum(y_balanced .== 0))

##############################################################

clf = pipe(StandardScaler(), 
           PCA(n_components=2),
           LogisticRegression(solver=:lbfgs, λ=0.01, random_state=1))

clf(Xtrn, ytrn)
ŷprobs = clf(Xtst, type=:probs)
