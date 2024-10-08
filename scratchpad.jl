# This will prompt if neccessary to install everything, including CUDA:
using Flux, CUDA, Statistics, ProgressMeter

# Generate some data for the XOR problem: vectors of length 2, as columns of a matrix:
noisy = rand(Float32, 2, 1000)                                    # 2×1000 Matrix{Float32}
truth = [xor(col[1]>0.5, col[2]>0.5) for col in eachcol(noisy)]   # 1000-element Vector{Bool}

# Define our model, a multi-layer perceptron with one hidden layer of size 3:
model = Chain(
    Dense(2 => 3, tanh),   # activation function inside layer
    BatchNorm(3),
    Dense(3 => 2)) |> gpu        # move model to GPU, if available

# The model encapsulates parameters, randomly initialised. Its initial output is:
out1 = model(noisy |> gpu) |> cpu                                 # 2×1000 Matrix{Float32}
probs1 = softmax(out1)      # normalise to get probabilities

# To train the model, we use batches of 64 samples, and one-hot encoding:
target = Flux.onehotbatch(truth, [true, false])                   # 2×1000 OneHotMatrix
loader = Flux.DataLoader((noisy, target) |> gpu, batchsize=64, shuffle=true);
# 16-element DataLoader with first element: (2×64 Matrix{Float32}, 2×64 OneHotMatrix)

optim = Flux.setup(Flux.Adam(0.01), model)  # will store optimiser momentum, etc.

# Training loop, using the whole data set 1000 times:
losses = []
@showprogress for epoch in 1:1_000
    for (x, y) in loader
        loss, grads = Flux.withgradient(model) do m
            # Evaluate model and loss inside gradient context:
            y_hat = m(x)
            Flux.logitcrossentropy(y_hat, y)
        end
        Flux.update!(optim, model, grads[1])
        push!(losses, loss)  # logging, outside gradient context
    end
end

# or 
@showprogress for epoch ∈ 1:1_000
    Flux.train!(model, loader, optim) do m, x, y 
        ŷ = m(x)
        Flux.logitcrossentropy(ŷ, y)
    end
end


optim # parameters, momenta and output have all changed
out2 = model(noisy |> gpu) |> cpu  # first row is prob. of true, second row p(false)
probs2 = softmax(out2)      # normalise to get probabilities
mean((probs2[1,:] .> 0.5) .== truth)  # accuracy 94% so far!

using Plots  # to draw the above figure
begin
    p_true = scatter(noisy[1,:], noisy[2,:], zcolor=truth, title="True classification", legend=false)
    p_raw =  scatter(noisy[1,:], noisy[2,:], zcolor=probs1[1,:], title="Untrained network", label="", clims=(0,1))
    p_done = scatter(noisy[1,:], noisy[2,:], zcolor=probs2[1,:], title="Trained network", legend=false)
    
    plot(p_true, p_raw, p_done, layout=(1,3), size=(1000,330))    
end

begin
    plot(losses; xaxis=(:log10, "iteration"),
         yaxis="loss", label="per batch")
    n = length(loader)
    plot!(n:n:length(losses), mean.(Iterators.partition(losses, n)), label="epoch mean", dpi=200)
end

#Fitting a line 
using Flux
actual(x) = 4x + 2
xtrn, xtst = hcat(0:5...), hcat(0:10...)
ytrn, ytst = actual.(xtrn), actual.(xtst)

model = Dense(1 => 1)
model.weight
model.bias
predict = Dense(1 => 1)
predict(xtrn)

using Statistics
loss(model, x, y) = mean(abs2.(model(x) .- y))
loss(predict, xtrn, ytrn)

opt = Descent()
data = [(xtrn, ytrn)]
predict.weight
predict.bias
Flux.train!(loss, predict, data, opt)
loss(predict, xtrn, ytrn)
predict.weight, predict.bias

for epoch in 1:200
    Flux.train!(loss, predict, data, opt)
end
loss(predict, xtrn, ytrn)
predict.weight, predict.bias
predict(xtst)
ytst

# Gradients and Layers 
using Flux 
f(x) = 3x^2 + 2x + 1;
df(x) = Flux.gradient(f, x)[1]
df(2)
d2f(x) = Flux.gradient(df, x)[1]
d2f(2)

f(x, y) = sum((x .- y).^2)
gradient(f, [2,1], [2, 0])
nt = (a=[2,1], b=[2,0], c=tanh)
g(x::NamedTuple) = sum(abs2, x.a .- x.b)
g(nt)
dg_nt = gradient(g, nt)[1]
gradient((x,y) -> sum(abs2, x.a ./ y .- x.b), nt, [1,2])

gradient(nt, [1,2]) do x, y 
    z = x.a ./ y 
    sum(abs2, z .- x.b)
end
Flux.withgradient(g, nt)

predict(W, b, x) = W*x .+ b 
function loss(W, b, x, y)
    ŷ = predict(W, b, x)
    sum((y .- ŷ).^2)
end
x, y = rand(5), rand(2)
w = rand(2, 5)
b = rand(2)
loss(w, b, x, y)
∂w, ∂b = gradient((w,b) -> loss(w, b, x, y), w, b)
w .-= 0.1 .* ∂w 
loss(w, b, x, y)

w1 = rand(3, 5)
b1 = rand(3)
layer1(x) = w1 * x .+ b1
w2 = rand(2,3)
b2 = rand(2)
layer2(x) = w2 * x .+ b2
model(x) = layer2(sigmoid.(layer1(x)))
model(rand(5))
function linear(in, out)
    w = randn(out, in)
    b = randn(out)
    x -> w*x .+ b 
end
linear1 = linear(5,3)
linear2 = linear(3,2)
model(x) = linear2(sigmoid.(linear1(x)))
model(rand(5))
struct Affine
    w 
    b
end

Affine(in::Integer, out::Integer) = Affine(randn(out,in), zeros(out))

# Overload call, so the object can be used as a function
(m::Affine)(x) = m.w * x .+ m.b
a = Affine(10, 5)
a(rand(10))

layers = [Dense(10 => 5, relu), Dense(5 => 2), softmax]
model(x) = foldl((x, m)-> m(x), layers, init=x)
model(Float32.(rand(10)))

model2 = Chain(
    Dense(10 => 5, relu),
    Dense(5 => 2),
    softmax)
model2(rand(10))
m = Dense(5 => 2) ∘ Dense(10 => 5, σ)
m(rand(10))

m = Chain(x -> x^2, x -> x+1)
