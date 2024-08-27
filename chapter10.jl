using Revise

using NovaML.Datasets
X, y = make_blobs(
    n_samples=150,
    n_features=2,
    centers=3,
    cluster_std=0.5,
    shuffle=true,
    random_state=123)

using Plots
begin
    scatter(X[:, 1], X[:, 2],
            color=:white,
            markerstrokecolor=:black,
            markersize=6,
            xlabel="Feature 1",
            ylabel="Feature 2",
            grid=true,
            legend=false)
    
    plot!(size=(600, 400), margin=5Plots.mm)
end

using NovaML.Cluster
km = KMeans(
    n_clusters=3,
    init="random",
    n_init=10,
    max_iter=300,
    tol=1e-04,
    random_state=0)

km(X)
km.labels_
km.inertia_
km.cluster_centers_

ykm = km.labels_

begin
    scatter(X[ykm.==1, 1], X[ykm.==1, 2],
            markersize=6, color=:lightgreen,
            markershape=:square, markerstrokecolor=:black,
            label="cluster 1")
    scatter!(X[ykm.==2, 1], X[ykm.==2, 2],
             markersize=6, color=:orange,
             markershpe=:circle, markerstrokecolor=:black,
             label="Cluster 2")
    scatter!(X[ykm.==3, 1], X[ykm.==3, 2],
             markersize=6, color=:lightblue,
             markershape=:utriangle, markerstrokecolor=:black,
             label="Cluster 3")
    scatter!(km.cluster_centers_[:, 1],
             km.cluster_centers_[:, 2],
             markersize=14, color=:red,
             markershape=:star5, markerstrokecolor=:black,
             label="Centroids")
    xlabel!("Feature 1")
    ylabel!("Feature 2")
    plot!(legend=:topright)
    plot!(size=(800, 600), dpi=300)
end

println("Distortion: $(round(km.inertia_, digits=2))")

distortions = Float16[]
for i ∈ 1:10
    km = KMeans(
        n_clusters=i,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=123
    )
    km(X)
    push!(distortions, km.inertia_)
end

plot(1:10, distortions, markershape=:circle,
     xlabel="Number of clusters",
     ylabel="Distortion",
     legend=nothing)

km = KMeans(
    n_clusters=3,
    init="k-means++",
    n_init=10,
    max_iter=300,
    tol=1e-04,
    random_state=0)

km(X)

using Statistics
using ColorSchemes
using NovaML.Metrics

function plot_silhoutte(km::KMeans)
    ykm = km.labels_
    cluster_labels = sort(unique(km.labels_))
    n_clusters = length(cluster_labels)
    silhouette_vals = silhouette_samples(X, km.labels_, metric="euclidean")
    δ = 1. / (length(silhouette_vals)+20)
    yval = 10δ
    
    p = plot(xlabel="Silhouette coefficient",label="Cluster", title="Silhouette Plot", legend=false, ylims=(0.0, 1.0), xlims=(0.0, 1.0), ylabel="Cluster");
    for (i, c) in enumerate(cluster_labels)
        c_silhouette_vals = silhouette_vals[ykm.==c]
        sort!(c_silhouette_vals)
        color = get(ColorSchemes.jet, i/n_clusters)
        for xval in c_silhouette_vals
            plot!(p, [0, xval], [yval, yval], color=color)
            yval += δ
        end
    end
    silhouette_avg = mean(silhouette_vals)
    vline!([silhouette_avg], color=:red, linestyle=:dash, lw=2)
    
    start = (1-20δ)/6
    stop = (1 - 10δ) - (1-20δ)/6
    
    yticks!(p, range(start, stop, length=n_clusters), string.(cluster_labels))    
end

plot_silhoutte(km)

km = KMeans(
    n_clusters=2, 
    init="k-means++",
    n_init=10,
    max_iter=300,
    tol=1e-04,
    random_state=123)

km(X)
ykm = km.labels_

begin
    scatter(X[ykm.==1, 1], X[ykm.==1, 2],
            markersize=6, color=:lightgreen,
            markerstrokecolor=:black, marker=:circle,
            label="Cluster 1")
    scatter!(X[ykm.==2, 1], X[ykm.==2, 2],
             markersize=6, color=:orange,
             markerstrokecolor=:black, marker=:rect,
             label="Cluster 2")
    scatter!(km.cluster_centers_[:, 1],
             km.cluster_centers_[:, 2],
             markersize=12, color=:red,
             marker=:star, label="Centroids")
    xlabel!("Feature 1")
    ylabel!("Feature 2")
end

plot_silhoutte(km)

variables = [:X, :Y, :Z]
labels = ["ID_0", "ID_1", "ID_2", "ID_3", "ID_4"]
X = rand(5,3).*10

using DataFrames
df = DataFrame(X, :auto)
rename!(df, variables)


Vector(df[1, :])