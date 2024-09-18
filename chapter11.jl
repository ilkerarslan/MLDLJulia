using MLDatasets, DataFrames, Plots, Random

Xtrn, ytrn = MNIST(split=:train)[:]
Xtst, ytst = MNIST(split=:test)[:]

begin
    # Create a 2x5 grid of plots
    plot_array = Array{Plots.Plot}(undef, 2, 5)
    
    for i in 0:9
        # Find the first instance of digit i
        idx = findfirst(==(i), ytrn)
        
        # Extract, reshape, and transpose the image
        img = permutedims(Xtrn[:, :, idx], (2, 1))
        
        # Create a heatmap for the digit
        p = heatmap(img, 
                    aspect_ratio=:equal, 
                    c=:grays, 
                    axis=false, 
                    colorbar=false, 
                    title="Digit $i",
                    yflip=true)  # Flip the y-axis to match Python's imshow
        
        # Store the plot in our array
        plot_array[fld(i, 5) + 1, rem(i, 5) + 1] = p
    end
    
    # Combine all plots into a single figure
    final_plot = plot(plot_array..., layout=(2, 5), size=(800, 400))
    
    # Display the plot
    display(final_plot)
end

begin
    # Create a 5x5 grid of plots
    plot_array = Array{Plots.Plot}(undef, 5, 5)
    
    # Find indices of all '7' digits
    seven_indices = findall(==(7), ytrn)
    
    for i in 1:25
        # Extract and reshape the image
        img = permutedims(Xtrn[:, :, seven_indices[i]], (2,1))
        
        # Create a heatmap for the digit
        p = heatmap(img, 
                    aspect_ratio=:equal, 
                    c=:grays, 
                    axis=false, 
                    colorbar=false,
                    title="",
                    yflip=true)  # Flip the y-axis to correct orientation
        
        # Store the plot in our array
        plot_array[fld(i-1, 5) + 1, rem(i-1, 5) + 1] = p
    end
    
    # Combine all plots into a single figure
    final_plot = plot(plot_array..., layout=(5, 5), size=(800, 800), link=:all)
    
    # Display the plot
    display(final_plot)
end

Xtrn = reshape(Xtrn, :, size(Xtrn, 3))
Xtst = reshape(Xtst, :, size(Xtst, 3))

valindex = 1:5000
Xval = Xtrn[:, valindex]
yval = ytrn[valindex]

Xtrn = Xtrn[:, Not(valindex)]
ytrn = ytrn[Not(valindex)]


sigmoid(z) = 1. ./ (1. .+ exp.(-z))

function int_to_onehot(y, nclass)
    ary = zeros(Float32, nclass, size(y, 1))
    for (i, label) ∈ enumerate(y)
        ary[label+1, i] = 1
    end
    return ary
end

mutable struct NeuralNetMLP
    n_out::Int
    wʰ::Matrix{Float64}
    bʰ::Vector{Float64}
    wᵒ::Matrix{Float64}
    bᵒ::Vector{Float64}

    function NeuralNetMLP(;n_in::Int, n_hid::Int, n_out::Int, random_seed::Int=123)
        rng = MersenneTwister(random_seed)
        
        wʰ = randn(rng, Float64, (n_hid, n_in)) .* sqrt(2.0 / (n_in + n_hid))
        bʰ = zeros(n_hid)
        
        wᵒ = randn(rng, Float64, (n_out, n_hid)) .* sqrt(2.0 / (n_hid + n_out))
        bᵒ = zeros(n_out)
        
        new(n_out, wʰ, bʰ, wᵒ, bᵒ)
    end
end
    
function (nn::NeuralNetMLP)(X::Matrix{Float32})
    zʰ = nn.wʰ * X .+ nn.bʰ
    aʰ = sigmoid.(zʰ)
    
    zᵒ = nn.wᵒ * aʰ .+ nn.bᵒ
    aᵒ = sigmoid.(zᵒ)
    
    #println("Size ah: $(size(aʰ))")
    #println("Size ao: $(size(aᵒ))")
    return aʰ, aᵒ
end

function (nn::NeuralNetMLP)(X::Matrix{Float32}, aʰ::Matrix{Float64}, aᵒ::Matrix{Float64}, y::Vector{Int})
    yoh = int_to_onehot(y, nn.n_out)
    
    # Output layer
    ∂L∂aᵒ = 2 .* (aᵒ .- yoh) ./ size(X, 2)
    ∂aᵒ∂zᵒ = aᵒ .* (1 .- aᵒ)
    δᵒ = ∂L∂aᵒ .* ∂aᵒ∂zᵒ

    ∂L∂wᵒ = δᵒ * aʰ'
    ∂L∂bᵒ = vec(sum(δᵒ, dims=2))

    # Hidden layer
    ∂L∂aʰ = nn.wᵒ' * δᵒ
    ∂aʰ∂zʰ = aʰ .* (1 .- aʰ)
    δʰ = ∂L∂aʰ .* ∂aʰ∂zʰ
    ∂L∂wʰ = δʰ * X'
    ∂L∂bʰ = vec(sum(δʰ, dims=2))
    
    return (∂L∂wᵒ, ∂L∂bᵒ, ∂L∂wʰ, ∂L∂bʰ)    
end

model = NeuralNetMLP(
    n_in = 28*28,
    n_hid = 50,
    n_out = 10,
)

num_epochs = 50
minibatch_size = 100

function minibatch_generator(X::Matrix{Float32}, 
                             y::Vector{Int}, 
                             minibatch_size::Int)
    indices = shuffle(1:size(X, 1))
    batches = Iterators.partition(indices, minibatch_size)
    return ((X[:, batch], y[batch]) for batch in batches)
end

for i in 1:num_epochs
    minibatch_gen = minibatch_generator(Xtrn, ytrn, minibatch_size)
    for (Xtrn_mini, ytrn_mini) in minibatch_gen
        println("Minibatch X shape: ", size(Xtrn_mini))
        println("Minibatch y shape: ", size(ytrn_mini))
        break
    end
    break    
end

using Statistics

function mse_loss(y, probs, nclass=10)
    yohe = int_to_onehot(y, nclass)
    ε = yohe .- probs
    return mean(ε.^2)    
end

accuracy(y, ŷ) = sum(y .== ŷ) / length(y)

function compute_mse_and_acc(model::NeuralNetMLP, X, y;
                             nclass=10, minibatch_size=100)
    mse, correct_pred, num_examples = 0.0, 0, 0
    minibatch_gen = minibatch_generator(X, y, minibatch_size)
    i = 0
    for (_, (features, targets)) ∈ enumerate(minibatch_gen)
        _, probas = model(features)    
        predicted_labels = getindex.(vec(argmax(probas, dims=1)), 1) .- 1
        onehot_targets = int_to_onehot(targets, nclass)
        loss = mean((onehot_targets .- probas).^2)
        correct_pred += sum((predicted_labels .== targets))
        num_examples += length(targets)
        mse += loss 
        i += 1        
    end
    mse /= i 
    acc = correct_pred/num_examples
    return mse, acc
end

mse, acc = compute_mse_and_acc(model, Xval, yval)

function train(model, Xtrn, ytrn, Xval, yval, num_epochs; η=0.1)
    epoch_loss = []
    epoch_train_acc = []
    epoch_valid_acc = []

    for e in 1:num_epochs
        minibatch_gen = minibatch_generator(Xtrn, ytrn, minibatch_size)
        for (Xtrn_mini, ytrn_mini) in minibatch_gen
            ah, aout = model(Xtrn_mini)
            ∂L∂wᵒ, ∂L∂bᵒ, ∂L∂wʰ, ∂L∂bʰ = model(Xtrn_mini, ah, aout, ytrn_mini)

            model.wʰ -= η * ∂L∂wʰ
            model.bʰ -= η * ∂L∂bʰ
            model.wᵒ -= η * ∂L∂wᵒ
            model.bᵒ -= η * ∂L∂bᵒ
        end

        train_mse, train_acc = compute_mse_and_acc(model, Xtrn, ytrn)
        valid_mse, valid_acc = compute_mse_and_acc(model, Xval, yval)

        train_acc, valid_acc = train_acc*100, valid_acc*100
        push!(epoch_train_acc, train_acc)
        push!(epoch_valid_acc, valid_acc)
        push!(epoch_loss, train_mse)

        println("Epoch: $(lpad(e, 3, '0'))/$(lpad(num_epochs, 3, '0')) " *
        "| Train MSE: $(round(train_mse, digits=2)) " *
        "| Train Acc: $(round(train_acc, digits=2))% " *
        "| Valid Acc: $(round(valid_acc, digits=2))%")
    end
    return epoch_loss, epoch_train_acc, epoch_valid_acc
end

Random.seed!(123)
epoch_loss, epoch_train, epoch_valid = train(model, Xtrn, 
                                                 ytrn, Xval, 
                                                 yval, 200, η=0.1)


begin
    plot(1:length(epoch_loss), epoch_loss, label=nothing)
    xlabel!("Epoch")
    ylabel!("Mean squared error")
end                              

begin
    plot(1:length(epoch_train), epoch_train, label="Training")
    plot!(1:length(epoch_valid), epoch_valid, label="Validation")
    xlabel!("Epochs")
    ylabel!("Accuracy")    
end