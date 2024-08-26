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
ykm = km.labels_
using NovaML.Metrics
using Distances

cluster_labels = unique(km.labels_)
n_clusters = length(cluster_labels)
silhouette_vals = silhouette_samples(X, km.labels_, metric="euclidean")
yaxlower, yaxupper = 0, 0
yticks = []

using Statistics
using ColorSchemes

function plot_silhouette(X::AbstractMatrix, labels::AbstractVector)
    

    return p
end
i, c = 1,1

p = plot(size=(800, 600), 
             xlabel="Silhouette coefficient", 
             ylabel="Cluster", 
             title="Silhouette Plot", 
             legend=false,
             xlims=(-0.1, 1.1),  # Extend x-axis slightly
             yticks=(1:n_clusters, string.(cluster_labels)),
             yflip=true)
c_silhouette_vals = silhouette_vals[ykm .== c] |> sort

begin
    p = plot(size=(800, 600), 
             xlabel="Silhouette coefficient", 
             ylabel="Cluster", 
             title="Silhouette Plot", 
             legend=false,
             xlims=(-0.1, 1.1),  # Extend x-axis slightly
             yticks=(1:n_clusters, string.(cluster_labels)),
             yflip=true)
    cluster_labels = sort(unique(ykm))
    n_clusters = length(cluster_labels)
    silhouette_vals = silhouette_samples(X, ykm, metric="euclidean")
    yaxlower, yaxupper = 0, 0

    i, c = 1,1
    c_silhouette_vals = silhouette_vals[ykm.==c]
    sort!(c_silhouette_vals)
    yaxupper += length(c_silhouette_vals)
    color = get(ColorSchemes.tab10, i / n_clusters)
    for (dy,sv) ∈ enumerate(c_silhouette_vals) 
        plot!([0, i+dy], [sv, i+dy])
    end
    plot!()
    
end