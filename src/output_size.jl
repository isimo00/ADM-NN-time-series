using Flux, Plots, Statistics, MLDatasets, Images, TimeSeries, CSV, DataFrames

data = CSV.read("BAJAJFINSV.csv", DataFrame)

output_size = parse(Int, ARGS[1])

# Keep only the close price, and convert the datatype to Float32,
# Float32 is NECESSARY for recurrent models in Flux
vwap = Float32.(data.VWAP)
# Create a vector of features (our input) and a vector of labels (our target output)
X_original = vwap[1:2838]
Y = vwap[2:2839]
# Reshape input data into Flux recurrent data format
X = [[x] for x ∈ X_original]

# Model definition
model = Chain(
    RNN(1 => 32, relu),
    Dense(32 => output_size, identity)
)

# Train the model
epochs = 100
opt = ADAM()
θ = Flux.params(model) # Keep track of the model parameters
for epoch ∈ 1:epochs # Training loop
    Flux.reset!(model) # Reset the hidden state of the RNN
    # Compute the gradient of the mean squared error loss
    ∇ = gradient(θ) do
        model(X[1]) # Warm-up the model
        sum(Flux.Losses.mse.([model(x)[1] for x ∈ X[2:end]], Y[2:end]))
    end
    Flux.update!(opt, θ, ∇) # Update the parameters
end

# Reshape result from Flux data format to Julia standard
Prediction = [model(x)[1] for x ∈ X]
acc = [abs(X_original[i]-Prediction[i])/X_original[i]*100 for i in 1:size(X_original)[1]]
accuracy=100-mean(acc[300:end])
io = open("out/Output_size_$output_size.txt", "w");
write(io, string(accuracy));
close(io);
