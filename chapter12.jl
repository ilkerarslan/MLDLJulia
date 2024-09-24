using CUDA
using Flux

CUDA.functional()
Flux.GPU_BACKEND

W = cu(rand(2, 5))
b = cu(rand(2))

predict(x) = W*x .+ b
loss(x,y) = sum((predict(x) .- y).^2)
x, y = cu(rand(5)), cu(rand(2))
loss(x,y)