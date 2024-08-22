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

# correlation plot
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

# linear regression plot
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

# linear regression plot
begin
    lin_regplot(X, y, slr)
    xlabel!("Living area above ground in square feet")
    ylabel!("Sale price in US dollars")
end

ransac = RANSACRegression(
    estimator = LinearRegression(),
    max_trials = 100,
    min_samples = 0.95,
    residual_threshold = 65000,
    random_state=123    
)

X = reshape(X, length(X), 1)
ransac(X, y)

begin
    # Assuming ransac is your fitted RANSACRegressor
    inlier_mask = ransac.inlier_mask_
    outlier_mask = .!inlier_mask

    line_ransac = ransac(X[inlier_mask, :])
    
    # Create the plot
    p = scatter(X[inlier_mask], y[inlier_mask],
                color=:steelblue, markerstrokecolor=:white,
                marker=:circle, label="Inliers")

    scatter!(p, X[outlier_mask], y[outlier_mask],
             color=:limegreen, markerstrokecolor=:white,
             marker=:square, label="Outliers")

    plot!(p, X[inlier_mask], line_ransac, color=:black, linewidth=2, label="Ransac Regressor")

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

using NovaML.ModelSelection
X = df[:, Not(:SalePrice)] |> Matrix
y = df[:, :SalePrice]

Xtrn, Xtst, ytrn, ytst = train_test_split(X, y, test_size=0.3, random_state=123)

slr = LinearRegression()
slr(Xtrn, ytrn)
ŷtrn = slr(Xtrn)
ŷtst = slr(Xtst)

#Residual plots of trn and tst data
begin       
    x_max = maximum([maximum(ŷtrn), maximum(ŷtst)])
    x_min = minimum([minimum(ŷtrn), minimum(ŷtst)])
    
    p = plot(layout=(1,2), size=(800, 400), link=:y)
    
    scatter!(p[1], ŷtst, ŷtst .- ytst,
             color=:limegreen, markershape=:square, 
             markeralpha=0.8, markerstrokecolor=:white,
             label="Test data")
    
    scatter!(p[2], ŷtrn, ŷtrn .- ytrn,
             color=:steelblue, markershape=:circle, 
             markeralpha=0.8, markerstrokecolor=:white,
             label="Training data")
    
    ylabel!(p[1], "Residuals")
    xlabel!(p[1], "Predicted values")
    xlabel!(p[2], "Predicted values")
    
    hline!(p[1], [0], color=:black, linewidth=2, label="")
    hline!(p[2], [0], color=:black, linewidth=2, label="")
    
    xlims!(p[1], (x_min-100, x_max+100))
    xlims!(p[2], (x_min-100, x_max+100))
    
    plot!(p[1], legend=:topleft)
    plot!(p[2], legend=:topleft)
    
    display(p)
end

using NovaML.Metrics
msetrn = mse(ytrn, ŷtrn)
msetst = mse(ytst, ŷtst)

maetrn = mae(ytrn, ŷtrn)
maetst = mae(ytst, ŷtst)

lasso = Lasso(α=10.)
lasso(Xtrn, ytrn)
ŷtrn = lasso(Xtrn)
ŷtst = lasso(Xtst)
mse(ytrn, ŷtrn)
mse(ytst, ŷtst)

using NovaML.Datasets
X, y = load_boston(return_X_y=true)

lr = LinearRegression()
lasso = Lasso(α=1.)
ridge = Ridge(α=100.)

lr(X, y)
ŷlr = lr(X);
lasso(X, y)
ŷlasso = lasso(X);
ridge(X, y)
ŷridge = ridge(X);

r2_score(y, ŷlr)
r2_score(y, ŷlasso)
r2_score(y, ŷridge)

lr.b, lr.w
lasso.b, lasso.w
ridge.b, ridge.w

plot(1:size(X, 2), lr.w, label="linreg")
plot!(1:size(X, 2), lasso.w, label="lasso")
plot!(1:size(X, 2), ridge.w, label="ridge")
