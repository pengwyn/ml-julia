module LogBin

using Reexport
@reexport using MLBase

using Distributions

export LogisticClassifierBinary
Base.@kwdef mutable struct LogisticClassifierBinary <: ClassifierMixin
    learning_rate::Float64 = 0.05
    max_iter::Int = 200
    λ1::Float64 = 0.0
    λ2::Float64 = 0.0
    w = nothing
    b = nothing
    optimal_w = nothing
    optimal_b = nothing
end

import MLBase: initialiseWeights!

function initialiseWeights!(self::LogisticClassifierBinary, X)
    n_features = size(X,2)
    self.w = rand(Uniform(-1,1), n_features)
    self.b = rand(Uniform(-1,1))
end

import MLBase: forwardPass
calcZ(self::LogisticClassifierBinary, X) = X*self.w .+ self.b
forwardPass(self::LogisticClassifierBinary, X, z=calcZ(self,X)) = logistic.(z)
import MLBase: predict
function predict(self::LogisticClassifierBinary, X)
    # I'm guessing n_targets needs to be 2 for now, until we introduce one_hot stuff.
    y = forwardPass(self, X)
    
    labels = ifelse.(y .< 0.5, 0, 1)
end

import MLBase: lossFunc
function lossFunc(self::LogisticClassifierBinary, X, y_true)
    y_pred = forwardPass(self, X)
    binaryLogLoss(y_true, y_pred)
end

function computeLossGrad(self::LogisticClassifierBinary, X, δ)
    ∇w = X' * δ
    ∇b = dropdims(sum, δ, dims=1)

    return ∇w,∇b
end

function updateParams!(self::LogisticClassifierBinary, ∇w, ∇b)
    self.w -= self.learning_rate * ∇w
    self.b -= self.learning_rate * ∇b
end

const logisticDeriv = MLBase.DERIVATIVES[:logistic]
import MLBase: fit!
function fit!(self::LogisticClassifierBinary, X, y_true)

    for iter in 1:self.max_iter
        z = calcZ(self, X)
        y_pred = forwardPass(self, X, z)
        δ = ∇y_binaryLogLoss(y_true, y_pred) .* logisticDeriv.(z)
        # δ = 1/length(y_true) * (y_pred - y_true)

        ∇ = computeLossGrad(self, X, δ)
        updateParams!(self, ∇...)

        @debug "After iteration $iter" self.w self.b loss(self, X, y_true) δ ∇
    end
end

end