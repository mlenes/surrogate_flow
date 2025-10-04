# The standard RNN model
struct BaseModel <: Lux.AbstractLuxLayer
    n_pad::Int
    core::Any
end

function BaseModel(n_filter, n_in, n_hidden, n_out)
    n_pad = n_filter ÷ 2

    core = Chain(
        WrappedFunction(x -> pad_circular(x, n_pad; dims=1)),
        Conv((n_filter,), n_in=>n_hidden, swish),
        
        WrappedFunction(x -> pad_circular(x, n_pad; dims=1)),
        Conv((n_filter,), n_hidden=>n_hidden, swish),
        
        WrappedFunction(x -> pad_circular(x, n_pad; dims=1)),
        Conv((n_filter,), n_hidden=>n_hidden, swish),
        
        WrappedFunction(x -> pad_circular(x, n_pad; dims=1)),
        Conv((n_filter,), n_hidden=>n_out)
    )

    return BaseModel(n_pad, SkipConnection(core, +))
end

Lux.initialparameters(rng::AbstractRNG, m::BaseModel) = Lux.initialparameters(rng, m.core)
Lux.initialstates(rng::AbstractRNG, m::BaseModel) = Lux.initialstates(rng, m.core)

(m::BaseModel)(x, ps, st) = m.core(x, ps, st)


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


struct FluxModelSmall <: Lux.AbstractLuxLayer
    n_pad::Int
    core::Any
end

function FluxModelSmall(n_filter, n_in, n_hidden, n_out)
    n_pad = n_filter ÷ 2

    model_flux = Chain(
        WrappedFunction(x -> pad_circular(x, n_pad; dims=1)),
        Conv((n_filter,), n_in=>n_hidden, swish),
        
        WrappedFunction(x -> pad_circular(x, n_pad; dims=1)),
        Conv((n_filter,), n_hidden=>n_out)
    )

    div_kernel = reshape(Float32[0, -1, 1], :, 1, 1)
    div_layer = Chain(
        WrappedFunction(x -> pad_circular(x, 1; dims=1)),
        StaticConv1D(div_kernel)
    )

    return FluxModelSmall(n_pad, SkipConnection(Chain(model_flux, div_layer), +))
end

Lux.initialparameters(rng::AbstractRNG, m::FluxModelSmall) = Lux.initialparameters(rng, m.core)
Lux.initialstates(rng::AbstractRNG, m::FluxModelSmall) = Lux.initialstates(rng, m.core)

(m::FluxModelSmall)(x, ps, st) = m.core(x, ps, st)


# Custom layers for handling tuple input for variable timestepping
struct TupleConv{C} <: Lux.AbstractLuxLayer
    conv::C
end

function TupleConv(kernel_size::Tuple, n_in_out::Pair, activation)
    n_in, n_out = n_in_out
    return TupleConv(Conv(kernel_size, n_in => n_out, activation))
end

function TupleConv(kernel_size::Tuple, n_in_out::Pair)
    n_in, n_out = n_in_out
    return TupleConv(Conv(kernel_size, n_in => n_out))
end

Lux.initialparameters(rng::AbstractRNG, m::TupleConv) = (conv=Lux.initialparameters(rng, m.conv),)
Lux.initialstates(rng::AbstractRNG, m::TupleConv) = (conv=Lux.initialstates(rng, m.conv),)

function (m::TupleConv)((x, Δt), ps, st)
    y, st_conv = m.conv(x, ps.conv, st.conv)
    return (y, Δt), (conv=st_conv,)
end


struct TuplePadCircular <: Lux.AbstractLuxLayer
    n_pad::Int

    function TuplePadCircular(n_filter::Int)
        return new(n_filter ÷ 2)
    end
end

Lux.initialparameters(::AbstractRNG, m::TuplePadCircular) = NamedTuple()
Lux.initialstates(::AbstractRNG, m::TuplePadCircular) = NamedTuple()

function (m::TuplePadCircular)((x, Δt), ps, st)
    n_pad = m.n_pad
    nx, nc, nb = size(x)
    
    y = vcat(
        x[end-n_pad+1:end, :, :],
        x,
        x[1:n_pad, :, :]
    )

    return (y, Δt), st
end


struct TupleSkipConnection{L} <: Lux.AbstractLuxLayer
    layer::L
end

Lux.initialparameters(rng::AbstractRNG, m::TupleSkipConnection) = (layer = Lux.initialparameters(rng, m.layer),)
Lux.initialstates(rng::AbstractRNG, m::TupleSkipConnection) = (layer = Lux.initialstates(rng, m.layer),)

function (m::TupleSkipConnection)((x, Δt), ps, st)
    (y, Δt), st_layer = m.layer((x, Δt), ps.layer, st.layer)

    return (x .+ y, Δt), (layer=st_layer,)
end

struct TupleDrop <: Lux.AbstractLuxLayer
end

Lux.initialparameters(::AbstractRNG, m::TupleDrop) = NamedTuple()
Lux.initialstates(::AbstractRNG, m::TupleDrop) = NamedTuple()

function (m::TupleDrop)((x, Δt), ps, st)
    return x, st
end


struct FiLMBlock{G, B} <: Lux.AbstractLuxLayer
    γ_net::G
    β_net::B
end

function FiLMBlock(n_hidden::Int)
    return FiLMBlock(
        Chain(Dense(1, n_hidden), Dense(n_hidden, n_hidden)),
        Chain(Dense(1, n_hidden), Dense(n_hidden, n_hidden))
    )
end

function FiLMBlock(n_hidden::Int, activation)
    return FiLMBlock(
        Chain(Dense(1, n_hidden, activation), Dense(n_hidden, n_hidden)),
        Chain(Dense(1, n_hidden, activation), Dense(n_hidden, n_hidden))
    )
end

Lux.initialparameters(rng::AbstractRNG, m::FiLMBlock) = (γ_net=Lux.initialparameters(rng, m.γ_net), β_net=Lux.initialparameters(rng, m.β_net))
Lux.initialstates(rng::AbstractRNG, m::FiLMBlock) = (γ_net=Lux.initialstates(rng, m.γ_net), β_net=Lux.initialstates(rng, m.β_net))

function (m::FiLMBlock)((x, Δt), ps, st)
    γ, st_γ = m.γ_net(Δt, ps.γ_net, st.γ_net)
    β, st_β = m.β_net(Δt, ps.β_net, st.β_net)

    B = size(x, 3)
    
    γ = reshape(γ, 1, :, B)
    β = reshape(β, 1, :, B)

    return (x .* γ .+ β, Δt), (γ_net=st_γ, β_net=st_β)
end


struct TupleStaticConv <: Lux.AbstractLuxLayer
    kernel
end

function TupleStaticConv(kernel::AbstractArray)
    return TupleStaticConv(() -> copy(kernel))
end

Lux.initialparameters(::AbstractRNG, m::TupleStaticConv) = NamedTuple()
Lux.initialstates(::AbstractRNG, m::TupleStaticConv) = (kernel = m.kernel(),)

function (m::TupleStaticConv)((x, Δt), ps, st)
    y = NNlib.conv(x, st.kernel)
    return (y, Δt), st
end


struct TimeModel{M} <: Lux.AbstractLuxLayer
    main::M
end

function TimeModel(n_filter::Int, n_hidden::Int)
    core = Chain(
        TuplePadCircular(n_filter),
        TupleConv((n_filter,), 1 => n_hidden, swish),
        FiLMBlock(n_hidden, swish),

        TuplePadCircular(n_filter),
        TupleConv((n_filter,), n_hidden => n_hidden, swish),
        FiLMBlock(n_hidden, swish),

        TuplePadCircular(n_filter),
        TupleConv((n_filter,), n_hidden => n_hidden, swish),
        FiLMBlock(n_hidden, swish),

        TuplePadCircular(n_filter),
        TupleConv((n_filter,), n_hidden => 1)
        )

    main = Chain(TupleSkipConnection(core), TupleDrop())
    return TimeModel(main)
end

Lux.initialparameters(rng::AbstractRNG, m::TimeModel) = (main=Lux.initialparameters(rng,m.main),)
Lux.initialstates(rng::AbstractRNG, m::TimeModel) = (main=Lux.initialstates(rng,m.main),)

function (m::TimeModel)((x, Δt), ps, st)
    y, st_main = m.main((x, Δt), ps.main, st.main)
    return y, (main=st_main,)
end


struct TimeFluxModel{M} <: Lux.AbstractLuxLayer
    main::M
end

function TimeFluxModel(n_filter::Int, n_hidden::Int)
    core = Chain(
        TuplePadCircular(n_filter),
        TupleConv((n_filter,), 1 => n_hidden, swish),
        FiLMBlock(n_hidden, swish),

        TuplePadCircular(n_filter),
        TupleConv((n_filter,), n_hidden => n_hidden, swish),
        FiLMBlock(n_hidden, swish),

        TuplePadCircular(n_filter),
        TupleConv((n_filter,), n_hidden => n_hidden, swish),
        FiLMBlock(n_hidden, swish),

        TuplePadCircular(n_filter),
        TupleConv((n_filter,), n_hidden => 1)
        )

    div_kernel = reshape(Float32[0, -1, 1], :, 1, 1)
    flux_layer = Chain(core, TuplePadCircular(n_filter), TupleStaticConv(div_kernel))

    main = Chain(TupleSkipConnection(flux_layer), TupleDrop())
    return TimeFluxModel(main)
end

Lux.initialparameters(rng::AbstractRNG, m::TimeFluxModel) = (main=Lux.initialparameters(rng,m.main),)
Lux.initialstates(rng::AbstractRNG, m::TimeFluxModel) = (main=Lux.initialstates(rng,m.main),)

function (m::TimeFluxModel)((x, Δt), ps, st)
    y, st_main = m.main((x, Δt), ps.main, st.main)
    return y, (main=st_main,)
end


struct RLift{C} <: Lux.AbstractLuxLayer
    conv::C
end

function RLift(kernel_size::Tuple, activation)
    return RLift(Conv(kernel_size, 1 => 1, activation, use_bias=false))
end

function RLift(kernel_size::Tuple)
    return RLift(Conv(kernel_size, 1 => 1, use_bias=false))
end

Lux.initialparameters(rng::AbstractRNG, m::RLift) = (conv=Lux.initialparameters(rng, m.conv),)
Lux.initialstates(rng::AbstractRNG, m::RLift) = (conv=Lux.initialstates(rng, m.conv),)

function (m::RLift)(x, ps, st)
    nx_in, _, B = size(x)
    K = ps.conv.weight
    n_filter = size(K, 1)
    nx_out = nx_in-n_filter+1
    y = zeros(Float32, nx_out, 1, B, 2)

    y[:,:,:,1] .+= m.conv(x, (weight=K,), st.conv)[1]
    y[:,:,:,2] .+= m.conv(x, (weight=-1*reverse(K, dims=1),), st.conv)[1]

    return y, st
end


struct RGroupConv{C} <: Lux.AbstractLuxLayer
    conv::C
end

function RGroupConv(kernel_size::Tuple, n_in_out::Pair, activation)
    n_in, n_out = n_in_out
    return RGroupConv(Conv(kernel_size, n_in => n_out, activation, use_bias=false))
end

function RGroupConv(kernel_size::Tuple, n_in_out::Pair)
    n_in, n_out = n_in_out
    return RGroupConv(Conv(kernel_size, n_in => n_out, use_bias=false))
end

Lux.initialparameters(rng::AbstractRNG, m::RGroupConv) = (conv=Lux.initialparameters(rng, m.conv),)
Lux.initialstates(rng::AbstractRNG, m::RGroupConv) = (conv=Lux.initialstates(rng, m.conv),)

function (m::RGroupConv)(x, ps, st)
    nx_in, c_in, B, g = size(x)

    K = ps.conv.weight
    c_out = size(K, 3)
    n_filter = size(K, 1)
    nx_out = nx_in-n_filter+1
    y = zeros(Float32, nx_out, c_out, B, g)

    y[:,:,:,1] .+= (m.conv(x[:,:,:,1], (weight=K,), st.conv)[1] .+ m.conv(x[:,:,:,2], (weight=-1*reverse(K, dims=1),), st.conv)[1])
    y[:,:,:,2] .+= (m.conv(x[:,:,:,1], (weight=-1*reverse(K, dims=1),), st.conv)[1] .+ m.conv(x[:,:,:,2], (weight=K,), st.conv)[1])

    return y, st
end


struct RDrop <: Lux.AbstractLuxLayer
end

Lux.initialparameters(::AbstractRNG, m::RDrop) = NamedTuple()
Lux.initialstates(::AbstractRNG, m::RDrop) = NamedTuple()

function (m::RDrop)(x, ps, st)
    return x[:,:,:,1], st
end


struct PadCircular <: Lux.AbstractLuxLayer
    n_pad::Int

    function PadCircular(n_filter::Int)
        return new(n_filter ÷ 2)
    end
end

Lux.initialparameters(::AbstractRNG, m::PadCircular) = NamedTuple()
Lux.initialstates(::AbstractRNG, m::PadCircular) = NamedTuple()

function (m::PadCircular)(x, ps, st)
    n_pad = m.n_pad
    nx = size(x, 1)
    
    y = vcat(
        x[end-n_pad+1:end, :, :],
        x,
        x[1:n_pad, :, :]
    )

    return y, st
end


struct RPadCircular <: Lux.AbstractLuxLayer
    n_pad::Int

    function RPadCircular(n_filter::Int)
        return new(n_filter ÷ 2)
    end
end

Lux.initialparameters(::AbstractRNG, m::RPadCircular) = NamedTuple()
Lux.initialstates(::AbstractRNG, m::RPadCircular) = NamedTuple()

function (m::RPadCircular)(x, ps, st)
    n_pad = m.n_pad
    nx = size(x, 1)
    
    y = vcat(
        x[end-n_pad+1:end, :, :, :],
        x,
        x[1:n_pad, :, :, :]
    )

    return y, st
end


struct EquiModel{M} <: Lux.AbstractLuxLayer
    main::M
end

function EquiModel(n_filter::Int, n_hidden::Int)
    main = Chain(
            PadCircular(n_filter),
            RLift((n_filter,), tanh),

            RPadCircular(n_filter),
            RGroupConv((n_filter,), 1 => n_hidden, tanh),

            # RPadCircular(n_filter),
            # RGroupConv((n_filter,), n_hidden => 1),

            RDrop()
        )

    return EquiModel(main)
end

Lux.initialparameters(rng::AbstractRNG, m::EquiModel) = (main=Lux.initialparameters(rng, m.main),)
Lux.initialstates(rng::AbstractRNG, m::EquiModel) = (main=Lux.initialstates(rng, m.main),)

(m::EquiModel)(x, ps, st) = m.main(x, ps.main, st.main)