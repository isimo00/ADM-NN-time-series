using Flux, Plots, Statistics, MLDatasets

train_x, train_y = MNIST(split=:train)[:]
test_x, test_y = MNIST(split = :test)[:]

colorview(Gray, train_x[:, :, 1]')
println("hello world")
