using Flux, CUDA, Statistics, ProgressMeter

CUDA.devices()
Flux.GPU_BACKEND

# Building a linear regression model
## Custom model
Xtrn = 1:10 .|> Float32
ytrn = [1.0, 1.3, 3.1, 2.0, 5.0, 6.3, 6.6,7.4, 8.0, 9.0] .|> Float32

Xtrn, ytrn = Xtrn', ytrn'

using Plots
plot(Xtrn', ytrn', 
     seriestype=:scatter, 
     markersize=10, 
     xlabel="x", 
     ylabel="y", 
     legend=false)

Xtrn_norm = (Xtrn .- mean(Xtrn)) ./ std(Xtrn)

custom_model(W, b, x) = @. W*x + b
W = rand(Float32, 1, 1)
b = [0.0f0]

function custom_loss(W, b, x, y)
     ŷ = custom_model(W, b, x)
     sum((y .- ŷ).^2) / length(x)
end
custom_loss(W, b, Xtrn_norm, ytrn)

learning_rate = 0.1
num_epochs = 200
log_epochs = 10

∂L∂W, ∂L∂b, _, _ = Flux.gradient(custom_loss, W, b, Xtrn_norm, ytrn)
W .-= learning_rate .* ∂L∂W
b .-= learning_rate .* ∂L∂b
custom_loss(W, b, Xtrn_norm, ytrn)

function train_custom_model()
     ∂L∂W, ∂L∂b, _, _ = Flux.gradient(custom_loss, W, b, Xtrn_norm, ytrn)
     @. W -= learning_rate * ∂L∂W
     @. b -= learning_rate * ∂L∂b     
end
train_custom_model();
W, b, custom_loss(W, b, Xtrn_norm, ytrn)

for i ∈ 1:num_epochs
     train_custom_model()
     if i % log_epochs == 0
          println("Epoch: $i  Loss $(custom_loss(W, b, Xtrn_norm, ytrn))")
     end
end
W, b

Xtst = range(1, 10, length=100) .|> Float32
Xtst_norm  = (Xtst .- mean(Xtrn)) ./ std(Xtrn)
ŷ = custom_model(W, b, Xtst_norm)

begin
     scatter(Xtrn_norm', ytrn', markersize=10)
     plot!(Xtst_norm, ŷ, style=:dash, lw=3)
end

## Flux model 
model = Dense(1 => 1)
function loss(model, x, y)
     ŷ = model(Xtrn_norm)
     Flux.mse(ŷ, y)
end
loss(model, Xtrn_norm, ytrn)

function train_model()
     ∂L∂m, _, _ = gradient(loss, model, Xtrn_norm, ytrn)
     @. model.weight = model.weight - learning_rate * ∂L∂m.weight
     @. model.bias = model.bias - learning_rate * ∂L∂m.bias
end

for i ∈ 1:num_epochs
     train_model()
     if i % log_epochs == 0
          println("Epoch: $i  Loss $(loss(model, Xtrn_norm, ytrn))")
     end
end
model.weight, model.bias


# Building a multilayer perceptron for classifying flowers in the Iris dataset
