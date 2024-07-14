# Chapter 05: Compressing Data via Dimensionality Reduction
## Unsupervised dimensionality reduction via principal component analysis
### Extracting the principal components step by step
using Revise
using DataFrames, HTTP, Random, CSV

using NovaML.ModelSelection: train_test_split

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
CSV.write("data/wine.csv", wine)
X, y = wine[:, 2:end], wine[:, 1]
X = Matrix(X)

Xtrn, Xtst, ytrn, ytst = train_test_split(X, y, 
										  test_size=0.3, 
										  random_state=0, 
										  stratify=y)

using NovaML.PreProcessing: StandardScaler
sc = StandardScaler()
sc(Xtrn)
Xtrnstd, Xtststd = sc(Xtrn), sc(Xtst)

using Statistics
covmat = cov(Xtrnstd)

using  LinearAlgebra
eigen_decomposition = eigen(covmat)
eigen_vals, eigen_vecs = eigen_decomposition.values, eigen_decomposition.vectors

#### Total and explained variance
tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sort(eigen_vals, rev=true)]
cum_var_exp = cumsum(var_exp)

using Plots 
begin
	bar(1:13, var_exp, label="Individual explained variance")
	plot!(1:13, cum_var_exp, linetype=:steppost, 
		  label="Cumulative explained variance",
		  color="blue", linewidth=2)
	xlabel!("Explained variance ratio")
	ylabel!("Principal component index")
	xticks!(1:13)
end

#### Feature transformation

# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(abs(eigen_vals[i]), eigen_vecs[:, i]) 
			   for i ∈ eachindex(eigen_vals)]
sort!(eigen_pairs, by= k -> k[1], rev=true)
w = [eigen_pairs[1][2] eigen_pairs[2][2]]
Xtrnstd[1, :]' * w
Xtrnpca = Xtrnstd * w

begin
	colors = ["red", "blue", "green"]
	scatter()
	for (l, c) ∈ zip(unique(ytrn), colors)
		scatter!(Xtrnpca[ytrn.==l, 1], Xtrnpca[ytrn.==l, 2], 
				color=c, label="Class $l")
	end
	scatter!()	
end

# Principal component analysis in scikit learning
function plot_decision_regions(X, y, mach, test_idx=[], len=300)	
	colors = ["red", "blue", "lightgreen", "gray", "cyan"]
	x1_min, x1_max = minimum(X[:, 1]) - 1, maximum(X[:, 1]) + 1
	x2_min, x2_max = minimum(X[:, 2]) - 1, maximum(X[:, 2]) + 1
	xx1 = range(x1_min, x1_max, length=len)
	xx2 = range(x2_min, x2_max, length=len)
	z = [MLJ.predict_mode(mach, [x1 x2])[1] for x1 ∈ xx1, x2 ∈ xx2]

	p = contourf(xx1, xx2, z, 
				 color=[:red, :blue, :lightgreen],
				 levels=3, alpha=0.3, legend=false)

	# plot data points 
	for (i, cl) in enumerate(unique(y))
		idx = findall(y .== cl)
		scatter!(p, X[idx, 1], X[idx, 2], 
				 label="Class $cl", ms=4, color=colors[i])
	end
	scatter!()
end

using MLJ
models("Multinomial")
doc("MultinomialClassifier", pkg="MLJLinearModels")
MultinomialClassifier = @load MultinomialClassifier pkg=MLJLinearModels

models("PCA")
PCA = @load PCA pkg="MultivariateStats"
doc("PCA", pkg="MultivariateStats")
pca = PCA(maxoutdim=2)
dfXtrnstd = DataFrame(Xtrnstd, :auto)
mach = machine(pca, dfXtrnstd) |> fit!;
Xtrnpca = MLJ.transform(mach, Xtrnstd) |> DataFrame;
Xtstpca = MLJ.transform(mach, Xtststd) |> DataFrame;

ytrncat = categorical(ytrn)
ytstcat = categorical(ytst)

using MLJLinearModels, Optim
ovr = MultinomialClassifier(solver=MLJLinearModels.LBFGS(
	optim_options = Optim.Options(time_limit = 20),
    )
)
mach = machine(ovr, Xtrnpca, ytrncat) |> fit!;
plot_decision_regions(Xtrnpca, ytrncat, mach)

using NovaML.MultiClass: OneVsRestClassifier
using NovaML.LinearModel: LogisticRegression
using NovaML.Decomposition: PCA

pca = PCA(n_components=2)
pca(Xtrnstd)
Xtrnpca = pca(Xtrnstd)
Xtstpca = pca(Xtststd)

lr = LogisticRegression()
ovr = OneVsRestClassifier(lr)
ovr(Xtrnpca, ytrn)
ŷtst = ovr(Xtstpca)
sum(ytst .!= ŷtst)

function plot_decision_regions(X, y, model, length=300)
	colors = ["red", "blue", "lightgreen", "green", "cyan"]
	x1_min, x1_max = minimum(X[:, 1])-1, maximum(X[:, 1])+1
	x2_min, x2_max = minimum(X[:, 2])-1, maximum(X[:, 2])+1
	xx1 = range(x1_min, x1_max, length=length)
	xx2 = range(x2_min, x2_max, length=length)
	z = [model([x1 x2])[1] for x1 ∈ xx1, x2 ∈ xx2]
	p = contourf(xx1, xx2, z,
				 color=[:red, :blue, :green],
				 levels=3, alpha=0.3, legend=false)
	# plot data points 
	for (i, cl) ∈ enumerate(unique(y))
		idx = findall(ytrn.==cl)
		scatter!(p, X[idx, 1], X[idx, 2],
				 label="Class $(cl)", color=colors[i])
	end
	scatter!()	
end

begin
	plot_decision_regions(Xtrnpca, ytrn, ovr)
	xlabel!("PC 1")
	ylabel!("PC 2")
	plot!(legend=:bottomleft)	
end

# Assessing feature contributions
loadings = eigen_vecs .* sqrt.(eigen_vals)
begin
	bar(1:13, loadings[:, 1], xrotation=60)
	ylabel!("Loadings for PC 1")
	xticks!(1:13, nms[2:end])
end