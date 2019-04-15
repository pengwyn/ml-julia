module MLBase

"""Utilities for the neural network modules
"""

using Statistics
using Plots
using ArgCheck

# Note: all of these functions are defined for single element only as per julia style

# identity is already a function in Base

logistic(x) = 1 / (1 + exp(-x))

# tanh is already a function

relu(x) = x > 0 ? x : zero(x)

# Not sure about this - I think the input is not what is described in the problem.
function softmax(vec)
    out = exp.(vec)
    out /= sum(out)
end

ACTIVATIONS = Dict(:identity => identity,
                   :tanh => tanh,
                   :logistic => logistic,
                   :relu => relu,
                   :softmax => softmax)

			   

# All derivatives can be found with automatic differentiation
using ForwardDiff

DERIVATIVES = Dict(key => x -> ForwardDiff.derivative(func, x)
                   for (key,func) in ACTIVATIONS)


# Convenience function for 2-arg vectorisation
# prod here is always (-y_true * y_pred)

loss(func, prod) = mean(func, prod)

squaredLoss(prod) = (1 - prod)^2
logLoss(prod) = 1/log(2) * log(1 + exp(prod))
binaryLogLoss(prod) = sign(prod)

for func in [:squaredLoss, :logLoss, :binaryLogLoss]
    @eval $func(y_true,y_pred) = loss($func, -y_true*y_pred)
end


# Really these don't belong to anything
correct(y_true, y_pred) = y_true .== y_pred
misclassified(y_true, y_pred) = .!(correct(y_true, y_pred))
score(y_true, y_pred) = mean(y_true .== y_pred)


######################################
# * ClassifierMixin
#------------------------------------

export ClassifierMixin
abstract type ClassifierMixin end

export initialiseWeights!
function initialiseWeights! end
export forwardPass
function forwardPass end
export predict
function predict end

export plotFit
function plotFit(self::ClassifierMixin, X, y_true ; full_grid=true)
    n_features = size(X, 2)
    @argcheck n_features == 2

    y_pred = predict(self, X)

    s = score(y_pred, y_true)
		
    xlim = extrema(X[:,1]) .* 1.1
    ylim = extrema(X[:,2]) .* 1.1

    p = plot(xlims=xlim,
             ylims=ylim,
             legend=false,
             title="Score: $s")

    # Find the line separating these
    m = -self.w[1] / self.w[2]
    c = -self.b / self.w[2]

    func = x -> m*x + c

    plot!([func, func],
          fillrange=[ylim[1] ylim[2]],
          fillcolor=[:green :yellow],
          fillalpha=0.2,
          color=:black,
          linewidth=2)

    inds = misclassified(y_true,y_pred)
    cols = ifelse.(inds, :red, :blue)
    scatter!(X[:,1], X[:,2], color=cols)
end


end

