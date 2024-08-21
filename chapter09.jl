using Revise
using CSV, DataFrames, HTTP

# import data
begin
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
end

using Plots
using StatsPlots
using Statistics

begin
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
end

# Linear regression
X = df.GrLivArea
y = df.SalePrice

using NovaML.PreProcessing
scx = StandardScaler()
scy = StandardScaler()

Xstd = scx(X)
ystd = scy(y)

using NovaML.LinearModel
lr = LinearRegression(solver=:batch, η=0.1)
lr(Xstd, ystd)
lr(Xstd)

function lin_regplot(X, y, model)
    scatter(X, y, color=:steelblue,
            legend=nothing, markersize=7,
            markerstrokecolor=:white)
    plot!(X, model(X), linewidth=2, color=:black)
end

begin
    lin_regplot(Xstd, ystd, lr)
    xlabel!("Living area above ground (std)")
    ylabel!("Sale price (std)")        
end

feature_std = scx([2500])
target_std = lr(feature_std)
target_reverted = scy(target_std, type=:inverse_transform)

lr.b, lr.w

slr = LinearRegression()
slr(X, y)
ŷ = slr(X)
slr.b, slr.w

begin
    lin_regplot(X, y, slr)
    xlabel!("Living area above ground in square feet")
    ylabel!("Sale price in US dollars")
end

ransac = RANSACRegression(
    estimator = LinearRegression(),
    max_trials = 100,
    min_samples = 0.95,
    residual_threshold = nothing,
    random_state=123    
)

X = reshape(X, length(X), 1)
ransac(X, y)

begin
    # Assuming ransac is your fitted RANSACRegressor
    inlier_mask = ransac.inlier_mask_
    outlier_mask = .!inlier_mask

    line_X = 3:10
    line_y_ransac = ransac(reshape(line_X, :, 1))

    # Create the plot
    p = scatter(X[inlier_mask], y[inlier_mask],
                color=:steelblue, markerstrokecolor=:white,
                marker=:circle, label="Inliers")

    scatter!(p, X[outlier_mask], y[outlier_mask],
             color=:limegreen, markerstrokecolor=:white,
             marker=:square, label="Outliers")

    #plot!(p, line_X, line_y_ransac, color=:black, linewidth=2)

    xlabel!("Living area above ground in square feet")
    ylabel!("Sale price in U.S. dollars")    

    # Adjust the layout
    plot!(p, size=(800, 600), margin=5Plots.mm)

    # Display the plot
    display(p)    
end

println("Slope: $(ransac.estimator_.w[1])")
println("Intercept: $(ransac.estimator_.b)")

function mean_absolute_deviation(data)
    return mean(abs.(data .- mean(data)))
end

mean_absolute_deviation(y)

