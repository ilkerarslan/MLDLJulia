# Chapter 05: Compressing Data via Dimensionality Reduction
## Unsupervised dimensionality reduction via principal component analysis
### Extracting the principal components step by step

using DataFrames, HTTP, Random, CSV

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

using Nova.ModelSelection: train_test_split
Xtrn, Xtst, ytrn, ytst = train_test_split(X, y, 
										  test_size=0.3, 
										  random_state=0, 
										  stratify=y)

using Nova.PreProcessing: StandardScaler										  
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

colors = ["red", "blue", "green"]

begin
	scatter()
	for (l, c) ∈ zip(unique(ytrn), colors)
		scatter!(Xtrnpca[ytrn.==l, 1], Xtrnpca[ytrn.==l, 2], 
				color=c, label="Class $l")
	end
	scatter!()	
end


# Principal component analysis in scikit learning
