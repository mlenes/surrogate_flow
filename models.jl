# The standard RNN model
struct BaseModel <: Lux.AbstractLuxLayer
    n_pad::Int
    core::Any
end

function BaseModel(n_filter, n_in, n_hidden, n_out)
    n_pad = n_filter ÷ 2

    model_flux = Chain(
        WrappedFunction(x -> pad_circular(x, n_pad; dims=1)),
        Conv((n_filter,), n_in=>n_hidden, swish),
        
        WrappedFunction(x -> pad_circular(x, n_pad; dims=1)),
        Conv((n_filter,), n_hidden=>n_hidden, swish),
        
        WrappedFunction(x -> pad_circular(x, n_pad; dims=1)),
        Conv((n_filter,), n_hidden=>n_hidden, swish),
        
        WrappedFunction(x -> pad_circular(x, n_pad; dims=1)),
        Conv((n_filter,), n_hidden=>n_out)
    )

    return BaseModel(n_pad, SkipConnection(model_flux, +))
end

Lux.initialparameters(rng::AbstractRNG, m::BaseModel) = Lux.initialparameters(rng, m.core)
Lux.initialstates(rng::AbstractRNG, m::BaseModel) = Lux.initialstates(rng, m.core)

(m::BaseModel)(x, ps, st) = m.core(x, ps, st);


# Custom convolution layer with static kernel
struct StaticConv1D <: Lux.AbstractLuxLayer
    kernel
end

function StaticConv1D(kernel::AbstractArray)
    return StaticConv1D(() -> copy(kernel))
end

Lux.initialparameters(::AbstractRNG, layer::StaticConv1D) = NamedTuple()
Lux.initialstates(::AbstractRNG, layer::StaticConv1D) = (kernel = layer.kernel(),)
function (l::StaticConv1D)(x, ps, st)
    y = NNlib.conv(x, st.kernel)
    return y, st
end

# This model predicts fluxes as its next to last step, and then predicts the updated velocity field from those fluxes
struct FluxModel <: Lux.AbstractLuxLayer
    n_pad::Int
    core::Any
end

function FluxModel(n_filter, n_in, n_hidden, n_out)
    n_pad = n_filter ÷ 2

    model_flux = Chain(
        WrappedFunction(x -> pad_circular(x, n_pad; dims=1)),
        Conv((n_filter,), n_in=>n_hidden, swish),
        
        WrappedFunction(x -> pad_circular(x, n_pad; dims=1)),
        Conv((n_filter,), n_hidden=>n_hidden, swish),
        
        WrappedFunction(x -> pad_circular(x, n_pad; dims=1)),
        Conv((n_filter,), n_hidden=>n_hidden, swish),
        
        WrappedFunction(x -> pad_circular(x, n_pad; dims=1)),
        Conv((n_filter,), n_hidden=>n_out)
    )

    div_kernel = reshape(Float32[0, -1, 1], :, 1, 1)
    div_layer = Chain(
        WrappedFunction(x -> pad_circular(x, 1; dims=1)),
        StaticConv1D(div_kernel)
    )

    return FluxModel(n_pad, SkipConnection(Chain(model_flux, div_layer), +))
end

Lux.initialparameters(rng::AbstractRNG, m::FluxModel) = Lux.initialparameters(rng, m.core)
Lux.initialstates(rng::AbstractRNG, m::FluxModel) = Lux.initialstates(rng, m.core)

(m::FluxModel)(x, ps, st) = m.core(x, ps, st)
