# Chapter 05: Compressing Data via Dimensionality Reduction
## Unsupervised dimensionality reduction via principal component analysis
### Extracting the principal components step by step
using Revise
using DataFrames, HTTP, Random, CSV

using NovaML.Datasets
wine = load_wine()
X, y = wine["data"], wine["target"]
nms = wine["feature_names"]

using NovaML.ModelSelection: train_test_split
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

using NovaML.Metrics: accuracy_score
accuracy_score(ytst, ŷtst)

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