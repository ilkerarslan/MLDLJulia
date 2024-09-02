using Test
using LinearAlgebra
using Statistics

# Assuming your AgglomerativeClustering is in a module called Cluster
using NovaML.Cluster: AgglomerativeClustering

@testset "AgglomerativeClustering Tests" begin
    
    @testset "Basic Functionality" begin
        X = [1 2; 1 4; 1 0; 4 2; 4 4; 4 0]
        clustering = AgglomerativeClustering(n_clusters=2)
        clustering(X)
        
        @test length(unique(clustering.labels_)) == 2
        @test length(clustering.labels_) == size(X, 1)
        @test clustering.n_leaves_ == size(X, 1)
        @test clustering.n_connected_components_ == 2
    end

    @testset "Distance Computation" begin
        X = [1 2; 1 4; 1 0; 4 2; 4 4; 4 0]
        clustering = AgglomerativeClustering(n_clusters=2, compute_distances=true)
        clustering(X)
        
        @test length(clustering.distances_) == size(X, 1) - 1
        @test all(clustering.distances_ .>= 0)
    end

    @testset "Linkage Methods" begin
        X = [1 2; 1 4; 1 0; 4 2; 4 4; 4 0]
        for linkage in ["single", "complete", "average"]
            clustering = AgglomerativeClustering(n_clusters=2, linkage=linkage)
            clustering(X)
            @test length(unique(clustering.labels_)) == 2
        end
    end

    @testset "Different Metrics" begin
        X = [1 2; 1 4; 1 0; 4 2; 4 4; 4 0]
        for metric in ["euclidean", "manhattan"]
            clustering = AgglomerativeClustering(n_clusters=2, metric=metric)
            clustering(X)
            @test length(unique(clustering.labels_)) == 2
        end
    end

    @testset "Custom Metric" begin
        X = [1 2; 1 4; 1 0; 4 2; 4 4; 4 0]
        custom_metric(a, b) = sum(abs.(a .- b))
        clustering = AgglomerativeClustering(n_clusters=2, metric=custom_metric)
        clustering(X)
        @test length(unique(clustering.labels_)) == 2
    end

    @testset "Distance Threshold" begin
        X = [1 2; 1 4; 1 0; 4 2; 4 4; 4 0]
        clustering = AgglomerativeClustering(distance_threshold=3.0, compute_distances=true)
        clustering(X)
        @test all(clustering.distances_ .<= 3.0)
    end

    @testset "Children Structure" begin
        X = [1 2; 1 4; 1 0; 4 2; 4 4; 4 0]
        clustering = AgglomerativeClustering(n_clusters=2)
        clustering(X)
        @test size(clustering.children_) == (size(X, 1) - 1, 2)
        @test all(clustering.children_ .>= 1)
        @test all(clustering.children_ .<= 2*size(X, 1) - 1)
    end

    @testset "Edge Cases" begin
        # Single point
        X = [1 2]
        clustering = AgglomerativeClustering(n_clusters=1)
        clustering(X)
        @test clustering.labels_ == [1]

        # All points the same
        X = [1 2; 1 2; 1 2]
        clustering = AgglomerativeClustering(n_clusters=1)
        clustering(X)
        @test all(clustering.labels_ .== 1)
    end

    @testset "Fit Predict" begin
        X = [1 2; 1 4; 1 0; 4 2; 4 4; 4 0]
        clustering = AgglomerativeClustering(n_clusters=2)
        labels = clustering(X, :fit_predict)
        @test length(labels) == size(X, 1)
        @test length(unique(labels)) == 2
    end

    @testset "Error Handling" begin
        @test_throws ErrorException AgglomerativeClustering(n_clusters=nothing, distance_threshold=nothing)
        @test_throws ErrorException AgglomerativeClustering(n_clusters=2, distance_threshold=1.0)
        @test_throws ErrorException AgglomerativeClustering(linkage="ward", metric="manhattan")
    end
end