module Data

# module DataGenerator

using Distributions, Random

using ArgCheck, Parameters
using DataFrames 

randomRadius(radius, n_samples, noise) = rand(Normal(radius, noise), n_samples)

function distributeSamples(n_samples, n_targets)
    target_samples = fill(n_samples ÷ n_targets, n_targets)

    N_leftover = n_samples % n_targets
    target_samples[1:N_leftover] .+= 1

    y = map(eachindex(target_samples)) do id
        fill(id, target_samples[id])
    end

    y = vcat(y...)

    return target_samples, y
end

export makeDonut
makeDonut(n_samples::Int=100, n_targets=2, noise=0.) = makeDonut(1:n_targets, n_samples, n_targets, noise)
function makeDonut(radii::AbstractVector, n_samples=100, n_targets=2, noise=0.)
    @argcheck length(radii) == n_targets

    target_samples,y = distributeSamples(n_samples, n_targets)

    X = Matrix{Int}(undef,0,2)
    for t = 1:n_targets
        sample_size = target_samples[t]
        r = radii[t]

        θ = rand(Uniform(0,2π), sample_size)
        X0 = randomRadius(r, sample_size, noise) .* cos.(θ)
        X1 = randomRadius(r, sample_size, noise) .* sin.(θ)

        X = [X ; [X0 X1]]
    end

    return X,y
end

export makeCloud
function makeCloud(n_targets=2 ; n_features=2, kwds...)
    centres = map(1:n_targets) do i
        rand(Uniform(-1.0, 1.0), n_features)
    end
    noises = rand(Uniform(0, 0.2), n_targets)

    makeCloud(centres, noises ; n_features=n_features, kwds...)
end

function makeCloud(centres, noises ; n_samples=100, n_features)
        """
        Generates Gaussian clouds of dimension n_features

        A simple toy dataset to visualize clustering and classification
        algorithms.

        Parameters
        ----------
        n_samples : int, optional (default=100)
            The total number of points generated.
        n_targets : int, optional (default=2)
            The number of target classes (spiral arms)
        noise : double or None (default=None)
            Standard deviation of Gaussian noise added to the data.
        n_features : int, optional (default=2)
            Dimension of each cloud
        random_state : int, RandomState instance or None (default)
            Determines random number generation for dataset shuffling and noise.
            Pass an int for reproducible output across multiple function calls.

        Returns
        -------
        X : array of shape [n_samples, n_features]
            The generated samples.
        y : array of shape [n_samples]
            The integer labels (0, 1, ..., targets) for class membership of each sample.
        """

    @argcheck length(centres) == length(noises)
    n_targets = length(centres)

    target_samples,y = distributeSamples(n_samples, n_targets)

    X = map(y) do ind
        point = rand.(Normal.(centres[ind], noises[ind]))
    end
    X = hcat(X...) |> permutedims

    return X,y
end
        
        
export makeSpiral
function makeSpiral(n_targets=2 ; n_samples=100, noise=0.05, inner_radius=0, outer_radius=2)
    # n_features is fixed to 2 here.
    target_samples,y = distributeSamples(n_samples, n_targets)

    phases = LinRange(0, 2π, n_targets+1)[1:end-1]

    X = map(target_samples,phases) do target_n,phase
        # Parameteric with variable p
        p = rand(target_n)

        r = @. inner_radius + p * (outer_radius - inner_radius)
        θ = @. phase + 1.75 * p * 2π

        X1 = @. r * sin(θ)
        X2 = @. r * cos(θ)

        X1 += rand(Normal(0, noise), target_n)
        X2 += rand(Normal(0, noise), target_n)

        [X1 X2]
    end

    X = vcat(X...)

    return X,y
end

export makeXor
function makeXor(n_features=2 ; n_samples=100)
    # n_targets is fixed to 2 here.

    # This would be a lot easier if we could just generate points and classify
    # rather than setting the number of samples per target a priori.

    # Yes I'm going to do that instead...

    XorClassify(point) = xor((point .> 0)...) ? 1 : 2

    X = rand(Uniform(-1, 1), n_samples, n_features)

    y = mapslices(XorClassify, X, dims=2)
    y = vec(y)
        
    return X,y
end

export makeMoons
function makeMoons(; n_samples=100, noise=0.05)
    # n_features and n_targets are fixed to 2 here.
    target_samples,y = distributeSamples(n_samples, 2)

    funcs = [θ -> 2*[cos(θ), sin(θ)] - [1,1],
             θ -> 2*[cos(θ), -sin(θ)] + [1,1]]
    X = map(target_samples,funcs) do target_n,func
        θ = rand(Uniform(0,π), target_n)
        hcat(func.(θ)...)
    end
    X = hcat(X...) |> permutedims

    return X,y
end

export DataContainer
@with_kw mutable struct DataContainer
    data
    data_df::DataFrame = storeDataAsDF(data)

    n_samples::Int = size(data_df,1)
    n_features::Int = size(data_df,2) - 1
    feature_names::Vector{Symbol} = names(data_df)
    n_targets::Int = length(unique(data_df[:,end]))

    shuffled::Bool = false

    scales = [ones(n_features), zeros(n_features)]
end

function DataContainer(data, shuffle_data=true)
    obj = DataContainer(data=data)
    shuffle_data && shuffle!(obj)
    return obj
end

storeDataAsDF(data::DataFrame) = data
function storeDataAsDF(data)
    X,y = data
    n_samples,n_features = size(X)

    columns = map(1:n_features) do col
        Symbol("X$col") => X[:,col]
    end

    df = DataFrame(; columns..., y=y)
end

function extractArrays(self::DataContainer)
    X = df[:, 1:self.n_features]
    y = df[:, end]
    return X, y
end

function shuffle!(self::DataContainer)
    self.data_df = self.data_df[randperm(self.n_samples),:]
    self.shuffled = true
end

function trainTestSplit(self::DataContainer, frac=0.8)
        """Split the dataset self.data_df into training and test sets


        Parameters
        ----------
        frac : float, optional (default=0.8)
            Fraction of the dataset assigned to the training set (with the remaining
            (1 - frac) assigned to the test set

        Returns
        -------
        X_train : array of shape [frac*n_samples, n_features]
            The training samples.
        y_train : array of shape [n_samples, 1]
            The training targets.
        X_test : array of shape [(1 - frac)*n_samples, n_features]
            The test samples.
        y_test : array of shape [n_samples, 1]
            The test targets
        """

    n_train = round(Int, self.n_samples * frac)

    X,y = extractArrays(self)

    X_train = X[1:n_train, :]
    y_train = y[1:n_train]

    X_test = X[n_train+1:end, :]
    y_test = y[n_train+1:end]

    return X_train, y_train, X_test, y_test
end
    
using RecipesBase
@recipe function plot(self::DataContainer)
    seriestype := :scatter
    markercolors := self.data_df[:,end]
    xlabel --> self.feature_names[1]
    ylabel --> self.feature_names[2]

    if self.n_features == 3
        zlabel --> self.feature_names[3]
    end

    # Bit of a mess! Does the necessary conversion from DataFrame to tuple with cols...
    self.data_df[:,1:end-1] |> Matrix |> eachcol |> x -> tuple(x...)
end

    # def scale(self, X=None, inplace=True):
    #     """
    #     Scale the input to have mean=0 and sd=1 for each feature

    #     Parameters
    #     ----------
    #     X : array (optional)
    #         Optional array to scale. If None (default) self.data_df is scaled
    #     inplace : Boolean (optional, default=True)
    #         Boolean argument to determine whether the scaled array should replace the original

    #     Returns
    #     -------
    #     X_scaled : array
    #         Scaled array where each feature has mean=0 and sd=1
    #     """

    # def back_transform(self, X_scaled=None, scales=None):
    #     """
    #     Back transform the scaled array to the untransformed original

    #     Parameters
    #     ----------

    #     Returns
    #     -------
    #     self : Restores unscaled input features
    #     """

    # def add_polynomial_features(self, degree=2, terms=None, powers_only=False, interaction_only=False, inplace=True):
    #     """
    #     Transform the original dataset X by adding polynomial and interaction features, e.g. X0**2, X1*X2

    #     Parameters
    #     ----------
    #     degree : int (default=2)
    #         The degree of polynomial and interaction terms to be added to the original dataset. For example, if the
    #         original dataset had three features: X0; X1; and X2 and we set the degree = 2, the new dataset would contain
    #         the additional features: X0**2, X1**2, and X2**2 and X0*X1, X0*X2 and X1*X2.
    #     terms : list (default = None)
    #         A list of the features to be transformed, i.e. if only a subset of features are of interest
    #     powers_only: Boolean (default = False)
    #         Include only polynomial terms, e.g. X0**3
    #     interaction_only: Boolean (default = False)
    #         Include only interaction terms, e.g. X0*X1
    #     in_place : Boolean (default = True)
    #         Store the modified (i.e. extended) array in place of the original dataset

    #     Returns
    #     -------
    #     X_new : array
    #         Extended array containing additional polynomial and interaction features
    #     """

    # def one_hot_enc(self, y=None):
    #     """
    #     Convert one-dimensional target lists or arrays to one-hot encoded arrays

    #     Parameters
    #     ----------
    #     y : list/array (default = None)
    #         List/array of targets

    #     Returns
    #     -------
    #     y_one_hot : array, shape = (n_samples, n_targets)
    #         One-hot encoded target array
    #     """

    # def add_white_noise(self):
    #     """
    #     Add a new feature of random white noise

    #     ...

    #     Returns
    #     -------
    #     X : array of shape [n_samples, n_features + 1]
    #         The generated samples plus a new column of white noise.
    #     """

# end

# using .DataGenerator

# Stuff stolen from my library

end
