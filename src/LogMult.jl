module LogMult

using Reexport
@reexport using MLBase
using Data

using Distributions

export LogisticClassifierMultinomial
Base.@kwdef mutable struct LogisticClassifierMultinomial <: ClassifierMixin
    learning_rate::Float64 = 0.05
    max_iter::Int = 200
    λ1::Float64 = 0.0
    λ2::Float64 = 0.0
    w = nothing
    b = nothing
end

import MLBase: initialiseWeights!

initialiseWeights!(self::LogisticClassifierMultinomial, cont::DataContainer) = initialiseWeights!(self, cont.n_features, cont.n_targets)
function initialiseWeights!(self::LogisticClassifierMultinomial, n_features, n_targets)
    self.w = rand(Uniform(-1,1), n_features, n_targets)
    self.b = rand(Uniform(-1,1), 1, n_targets)
end

import MLBase: forwardPass
calcZ(self::LogisticClassifierMultinomial, X) = X*self.w .+ self.b
forwardPass(self::LogisticClassifierMultinomial, X, z=calcZ(self,X)) = mapslices(softmax, z, dims=2)
import MLBase: predict
function predict(self::LogisticClassifierMultinomial, X)
    y = forwardPass(self, X)
    
    labels = mapslices(argmax, y, dims=2)
end

import MLBase: lossFunc
function lossFunc(self::LogisticClassifierMultinomial, X, y_true)
    y_pred = forwardPass(self, X)
    logLoss(y_true, y_pred)
end

function computeLossGrad(self::LogisticClassifierMultinomial, X, δ)
    ∇w = X' * δ
    ∇w += self.λ2 * self.w
    ∇w += self.λ1 * sign.(self.w)

    # ∇b = dropdims(sum, δ, dims=1)
    # TODO: Are these correct??
    ∇b = float(sum(δ))
    ∇b += self.λ2 * self.b
    ∇b += self.λ1 * sign(self.b)

    return ∇w,∇b
end

function updateParams!(self::LogisticClassifierMultinomial, ∇w, ∇b)
    self.w -= self.learning_rate * ∇w
    self.b -= self.learning_rate * ∇b
end

const logisticDeriv = MLBase.DERIVATIVES[:logistic]
import MLBase: fit!
function fit!(self::LogisticClassifierMultinomial, X, y_true)

    optimal_loss = Inf
    optimal_w = nothing
    optimal_b = nothing
    
    for iter in 1:self.max_iter
        z = calcZ(self, X)
        y_pred = forwardPass(self, X, z)
        δ = ∇y_binaryLogLoss(y_true, y_pred) .* logisticDeriv.(z)

        ∇ = computeLossGrad(self, X, δ)
        updateParams!(self, ∇...)

        l = lossFunc(self, X, y_true)

        if l < optimal_loss
            optimal_loss = l
            optimal_w = self.w
            optimal_b = self.b
        end

        @debug "After iteration $iter" self.w self.b l δ ∇
    end

    self.w = optimal_w
    self.b = optimal_b
end

end
