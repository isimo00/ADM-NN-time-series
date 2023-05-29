using Flux, Plots, Statistics, MLDatasets, Images, TimeSeries, CSV, DataFrames

HL_size = 32;

data = CSV.read("BAJAJFINSV.csv", DataFrame)
vwap = Float32.(data.VWAP)
X_original = vwap[1:2838]
Y = vwap[2:2839]
# Reshape input data into Flux recurrent data format
X = [[x] for x ∈ X_original]

# Model definition
model = Chain(
    RNN(1 => HL_size, relu),
    Dense(HL_size => 1, identity)
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

plot(data.Date[1:2838], [X_original Prediction], label=["VWAP" "Prediction"], linewidth=3)
