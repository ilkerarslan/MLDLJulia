using Revise
using CSV, DataFrames, HTTP

url = "https://jse.amstat.org/v19n3/decock/AmesHousing.txt"
response = HTTP.get(url)
df = CSV.read(IOBuffer(String(response.body)), DataFrame, delim="\t");
columns = ["Overall Qual", "Overall Cond", "Gr Liv Area", "Central Air", "Total Bsmt SF", "SalePrice"]
df = df[:, columns]
size(df)

rename!(df, ["OverallQual", "OverallCond", "GrLivArea", "CentralAir", "TotalBsmtSF", "SalePrice"])
transform!(df, :CentralAir => ByRow(x -> x == "Y" ? 1 : 0) => :CentralAir)
describe(df)[:, [:variable, :nmissing]]
dropmissing!(df)

using Plots
using StatsPlots
using Statistics

p = @df df corrplot(cols(1:6), size=(800, 700), alpha=0.5, markersize=2);
plot!(p, 
    tickfontsize=6,
    xrot=45,
    guidefontsize=8,
    titlefontsize=10)

cm = cor(Matrix(df))
heatmap(
    cm,
    xticks=(1:size(cm,2), names(df)),
    yticks=(1:size(cm,1), names(df)),
    xrotation=90,
    yrotation=0,
    aspect_ratio=:equal,
    color=:viridis,
    clim=(-1,1),
    title="Correlation Heatmap",
    size=(800, 700))
for i in 1:size(cm,1)
    for j in 1:size(cm,2)
        annotate!(j, i, text(round(cm[i,j], digits=2), 8, :black, :center))
    end
end
plot!(colorbar=false)