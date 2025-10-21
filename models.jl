# The standard RNN model
struct BaseModel{C} <: Lux.AbstractLuxLayer
    main::C
end

function BaseModel(n_filter, n_hidden)
    main = Chain(
        PadCircular(n_filter),
        Conv((n_filter,), 1=>n_hidden, swish),

        PadCircular(n_filter),
        Conv((n_filter,), n_hidden=>n_hidden, swish),

        PadCircular(n_filter),
        Conv((n_filter,), n_hidden=>n_hidden, swish),
        
        PadCircular(n_filter),
        Conv((n_filter,), n_hidden=>1)
    )

    return BaseModel(SkipConnection(main, +))
end

Lux.initialparameters(rng::AbstractRNG, m::BaseModel) = (main=Lux.initialparameters(rng, m.main),)
Lux.initialstates(rng::AbstractRNG, m::BaseModel) = (main=Lux.initialstates(rng, m.main),)

function (m::BaseModel)(x, ps, st)
    y, newst = m.main(x, ps.main, st.main)
    return y, (main=newst,)
end


struct BaseModel2D{C} <: Lux.AbstractLuxLayer
    main::C
end

function BaseModel2D(n_filter, n_hidden)
    main = Chain(
        PadCircular2D(n_filter),
        Conv((n_filter, n_filter), 2=>n_hidden, swish),

        PadCircular2D(n_filter),
        Conv((n_filter, n_filter), n_hidden=>n_hidden, swish),

        PadCircular2D(n_filter),
        Conv((n_filter, n_filter), n_hidden=>n_hidden, swish),
        
        PadCircular2D(n_filter),
        Conv((n_filter, n_filter), n_hidden=>2)
    )

    return BaseModel2D(SkipConnection(main, +))
end

Lux.initialparameters(rng::AbstractRNG, m::BaseModel2D) = (main=Lux.initialparameters(rng, m.main),)
Lux.initialstates(rng::AbstractRNG, m::BaseModel2D) = (main=Lux.initialstates(rng, m.main),)

function (m::BaseModel2D)(x, ps, st)
    y, newst = m.main(x, ps.main, st.main)
    return y, (main=newst,)
end


struct BaseModelSmall{C} <: Lux.AbstractLuxLayer
    main::C
end

function BaseModelSmall(n_filter, n_hidden)
    div_kernel = reshape(Float32[0, -1, 1], :, 1, 1)

    main = Chain(
        PadCircular(n_filter),
        Conv((n_filter,), 1=>n_hidden, swish),
        
        PadCircular(n_filter),
        Conv((n_filter,), n_hidden=>1)
    )

    return BaseModelSmall(SkipConnection(main, +))
end

Lux.initialparameters(rng::AbstractRNG, m::BaseModelSmall) = (main=Lux.initialparameters(rng, m.main),)
Lux.initialstates(rng::AbstractRNG, m::BaseModelSmall) = (main=Lux.initialstates(rng, m.main),)

function (m::BaseModelSmall)(x, ps, st)
    y, newst = m.main(x, ps.main, st.main)
    return y, (main=newst,)
end


struct BaseModelSmall2D{C} <: Lux.AbstractLuxLayer
    main::C
end

function BaseModelSmall2D(n_filter, n_hidden)
    main = Chain(
        PadCircular2D(n_filter),
        Conv((n_filter, n_filter), 2=>n_hidden, swish),
        
        PadCircular2D(n_filter),
        Conv((n_filter, n_filter), n_hidden=>2)
    )

    return BaseModelSmall2D(SkipConnection(main, +))
end

Lux.initialparameters(rng::AbstractRNG, m::BaseModelSmall2D) = (main=Lux.initialparameters(rng, m.main),)
Lux.initialstates(rng::AbstractRNG, m::BaseModelSmall2D) = (main=Lux.initialstates(rng, m.main),)

function (m::BaseModelSmall2D)(x, ps, st)
    y, newst = m.main(x, ps.main, st.main)
    return y, (main=newst,)
end


# Custom convolution layer with static kernel
struct StaticConv <: Lux.AbstractLuxLayer
    kernel
end

function StaticConv(kernel::AbstractArray)
    return StaticConv(() -> copy(kernel))
end

Lux.initialparameters(::AbstractRNG, ::StaticConv) = NamedTuple()
Lux.initialstates(::AbstractRNG, layer::StaticConv) = (kernel = layer.kernel(),)

function (l::StaticConv)(x, ps, st)
    y = NNlib.conv(x, st.kernel)
    return y, st
end


# This model predicts fluxes as its next to last step, and then predicts the updated velocity field from those fluxes
struct FluxModel{C} <: Lux.AbstractLuxLayer
    main::C
end

function FluxModel(n_filter, n_hidden)
    div_kernel = reshape(Float32[0, -1, 1], :, 1, 1)

    main = Chain(
        PadCircular(n_filter),
        Conv((n_filter,), 1=>n_hidden, swish),

        PadCircular(n_filter),
        Conv((n_filter,), n_hidden=>n_hidden, swish),

        PadCircular(n_filter),
        Conv((n_filter,), n_hidden=>n_hidden, swish),
        
        PadCircular(n_filter),
        Conv((n_filter,), n_hidden=>1),

        PadCircular(size(div_kernel,1)),
        StaticConv(div_kernel)
    )

    return FluxModel(SkipConnection(main, +))
end

Lux.initialparameters(rng::AbstractRNG, m::FluxModel) = (main=Lux.initialparameters(rng, m.main),)
Lux.initialstates(rng::AbstractRNG, m::FluxModel) = (main=Lux.initialstates(rng, m.main),)

function (m::FluxModel)(x, ps, st)
    y, newst = m.main(x, ps.main, st.main)
    return y, (main=newst,)
end


struct FluxModel2D{C} <: Lux.AbstractLuxLayer
    main::C
end

function FluxModel2D(n_filter, n_hidden)
    div_kernel = reshape(Float32[0, -1, 1], :, 1, 1)

    main = Chain(
        PadCircular2D(n_filter),
        Conv((n_filter,), 2=>n_hidden, swish),

        PadCircular2D(n_filter),
        Conv((n_filter,), n_hidden=>n_hidden, swish),

        PadCircular2D(n_filter),
        Conv((n_filter,), n_hidden=>n_hidden, swish),
        
        PadCircular2D(n_filter),
        Conv((n_filter,), n_hidden=>2),

        PadCircular2D(size(div_kernel,1)),
        StaticConv(div_kernel)
    )

    return FluxModel2D(SkipConnection(main, +))
end

Lux.initialparameters(rng::AbstractRNG, m::FluxModel2D) = (main=Lux.initialparameters(rng, m.main),)
Lux.initialstates(rng::AbstractRNG, m::FluxModel2D) = (main=Lux.initialstates(rng, m.main),)

function (m::FluxModel2D)(x, ps, st)
    y, newst = m.main(x, ps.main, st.main)
    return y, (main=newst,)
end


struct FluxModelSmall{C} <: Lux.AbstractLuxLayer
    main::C
end

function FluxModelSmall(n_filter, n_hidden)
    div_kernel = reshape(Float32[0, -1, 1], :, 1, 1)

    main = Chain(
        PadCircular(n_filter),
        Conv((n_filter,), 1=>n_hidden, swish),
        
        PadCircular(n_filter),
        Conv((n_filter,), n_hidden=>1),

        PadCircular(size(div_kernel,1)),
        StaticConv(div_kernel)
    )

    return FluxModelSmall(SkipConnection(main, +))
end

Lux.initialparameters(rng::AbstractRNG, m::FluxModelSmall) = (main=Lux.initialparameters(rng, m.main),)
Lux.initialstates(rng::AbstractRNG, m::FluxModelSmall) = (main=Lux.initialstates(rng, m.main),)

function (m::FluxModelSmall)(x, ps, st)
    y, newst = m.main(x, ps.main, st.main)
    return y, (main=newst,)
end


struct RLift <: Lux.AbstractLuxLayer
end

Lux.initialparameters(rng::AbstractRNG, ::RLift) = NamedTuple()
Lux.initialstates(rng::AbstractRNG, ::RLift) = NamedTuple()

function (m::RLift)(x, ps, st)
    y = cat(x, -reverse(x, dims=1), dims=4)
    return y, st
end


struct RConv{C} <: Lux.AbstractLuxLayer
    conv::C
end

function RConv(kernel_size::Tuple, n_in_out::Pair, activation)
    n_in, n_out = n_in_out
    return RConv(Conv(kernel_size, n_in => n_out, activation))
end

function RConv(kernel_size::Tuple, n_in_out::Pair)
    n_in, n_out = n_in_out
    return RConv(Conv(kernel_size, n_in => n_out))
end

Lux.initialparameters(rng::AbstractRNG, m::RConv) = (conv=Lux.initialparameters(rng, m.conv),)
Lux.initialstates(rng::AbstractRNG, m::RConv) = (conv=Lux.initialstates(rng, m.conv),)

function (m::RConv)(x, ps, st)
    y_e, st_conv = m.conv(x[:,:,:,1], ps.conv, st.conv)
    y_r, st_conv = m.conv(x[:,:,:,2], ps.conv, st.conv)
    y = cat(y_e, y_r, dims=4)
    return y, (conv=st_conv,)
end


struct RSkipConnection{L} <: Lux.AbstractLuxLayer
    layer::L
end

Lux.initialparameters(rng::AbstractRNG, m::RSkipConnection) = (layer=Lux.initialparameters(rng, m.layer),)
Lux.initialstates(rng::AbstractRNG, m::RSkipConnection) = (layer=Lux.initialstates(rng, m.layer),)

function (m::RSkipConnection)(x, ps, st)
    y_e = m.layer(x[:,:,:,1], ps.layer, st.layer)[1] .+ x[:,:,:,1]
    y_r = m.layer(x[:,:,:,2], ps.layer, st.layer)[1] .+ x[:,:,:,2]
    y = cat(y_e, y_r, dims=4)
    return y, (layer=st,)
end


struct RDrop <: Lux.AbstractLuxLayer
end

Lux.initialparameters(::AbstractRNG, ::RDrop) = NamedTuple()
Lux.initialstates(::AbstractRNG, ::RDrop) = NamedTuple()

function (m::RDrop)(x, ps, st)
    y = 0.5f0(x[:,:,:, 1] .+ -reverse(x[:,:,:,2], dims=1))
    return y, st
end


struct RPick <: Lux.AbstractLuxLayer
    pick::Int
end

Lux.initialparameters(::AbstractRNG, ::RPick) = NamedTuple()
Lux.initialstates(::AbstractRNG, ::RPick) = NamedTuple()

function (m::RPick)(x, ps, st)
    y = x[:,:,:, m.pick]
    return y, st
end


struct PadCircular <: Lux.AbstractLuxLayer
    n_pad::Int

    function PadCircular(n_filter::Int)
        return new(n_filter ÷ 2)
    end
end

Lux.initialparameters(::AbstractRNG, ::PadCircular) = NamedTuple()
Lux.initialstates(::AbstractRNG, ::PadCircular) = NamedTuple()

function (m::PadCircular)(x, ps, st)
    n_pad = m.n_pad
    
    y = vcat(
        x[end-n_pad+1:end, :, :],
        x,
        x[1:n_pad, :, :]
    )

    return y, st
end


struct PadCircular2D <: Lux.AbstractLuxLayer
    n_pad::Int

    function PadCircular2D(n_filter::Int)
        return new(n_filter ÷ 2)
    end
end

Lux.initialparameters(::AbstractRNG, ::PadCircular2D) = NamedTuple()
Lux.initialstates(::AbstractRNG, ::PadCircular2D) = NamedTuple()

function (m::PadCircular2D)(x, ps, st)
    n_pad = m.n_pad

    y = vcat(
        x[end-n_pad+1:end, :, :, :],
        x,
        x[1:n_pad, :, :, :]
    )

    y = hcat(
        y[:, end-n_pad+1:end, :, :],
        y,
        y[:, 1:n_pad, :, :]
    )

    return y, st
end


struct RPadCircular <: Lux.AbstractLuxLayer
    n_pad::Int

    function RPadCircular(n_filter::Int)
        return new(n_filter ÷ 2)
    end
end

Lux.initialparameters(::AbstractRNG, ::RPadCircular) = NamedTuple()
Lux.initialstates(::AbstractRNG, ::RPadCircular) = NamedTuple()

function (m::RPadCircular)(x, ps, st)
    n_pad = m.n_pad
    
    y = vcat(
        x[end-n_pad+1:end, :, :, :],
        x,
        x[1:n_pad, :, :, :]
    )

    return y, st
end


struct RFluxConv{C} <: Lux.AbstractLuxLayer
    conv::C
end

function RFluxConv(kernel::AbstractArray)
    return RFluxConv(StaticConv(kernel))
end

Lux.initialparameters(rng::AbstractRNG, m::RFluxConv) = (conv=Lux.initialparameters(rng, m.conv),)
Lux.initialstates(rng::AbstractRNG, m::RFluxConv) = (conv=Lux.initialstates(rng, m.conv),)

function (m::RFluxConv)(x, ps, st)
    y_e, st_conv = m.conv(x[:,:,:,1], ps.conv, st.conv)
    y_r, st_conv = m.conv(x[:,:,:,2], ps.conv, st.conv)

    y = cat(y_e, y_r, dims=4)
    return y, (conv=st_conv,)
end


struct EquiModel{M} <: Lux.AbstractLuxLayer
    main::M
end

function EquiModel(n_filter::Int, n_hidden::Int)
    main = Chain(
            RLift(),

            RPadCircular(n_filter),
            RConv((n_filter,), 1 => n_hidden, swish),

            RPadCircular(n_filter),
            RConv((n_filter,), n_hidden => n_hidden, swish),

            RPadCircular(n_filter),
            RConv((n_filter,), n_hidden => n_hidden, swish),

            RPadCircular(n_filter),
            RConv((n_filter,), n_hidden => 1),

            RDrop()
        )

    return EquiModel(SkipConnection(main, +))
end

Lux.initialparameters(rng::AbstractRNG, m::EquiModel) = (main=Lux.initialparameters(rng, m.main),)
Lux.initialstates(rng::AbstractRNG, m::EquiModel) = (main=Lux.initialstates(rng, m.main),)

function (m::EquiModel)(x, ps, st)
    y, newst = m.main(x, ps.main, st.main)
    return y, (main=newst,)
end


struct EquiFluxModel{M} <: Lux.AbstractLuxLayer
    main::M
end

function EquiFluxModel(n_filter::Int, n_hidden::Int)
    div_kernel = reshape(Float32[0, -1, 1], :, 1, 1)

    main = Chain(
            RLift(),

            RPadCircular(n_filter),
            RConv((n_filter,), 1 => n_hidden, swish),

            RPadCircular(n_filter),
            RConv((n_filter,), n_hidden => n_hidden, swish),

            RPadCircular(n_filter),
            RConv((n_filter,), n_hidden => n_hidden, swish),

            RPadCircular(n_filter),
            RConv((n_filter,), n_hidden => 1),

            RPadCircular(size(div_kernel,1)),
            RFluxConv(div_kernel),

            RDrop()
        )

    return EquiFluxModel(SkipConnection(main, +))
end

Lux.initialparameters(rng::AbstractRNG, m::EquiFluxModel) = (main=Lux.initialparameters(rng, m.main),)
Lux.initialstates(rng::AbstractRNG, m::EquiFluxModel) = (main=Lux.initialstates(rng, m.main),)

function (m::EquiFluxModel)(x, ps, st)
    y, newst = m.main(x, ps.main, st.main)
    return y, (main=newst,)
end


struct EquiFluxModelSmall{M} <: Lux.AbstractLuxLayer
    main::M
end

function EquiFluxModelSmall(n_filter::Int, n_hidden::Int)
    div_kernel = reshape(Float32[0, -1, 1], :, 1, 1)

    main = Chain(
            RLift(),

            RPadCircular(n_filter),
            RConv((n_filter,), 1 => n_hidden, swish),

            RPadCircular(n_filter),
            RConv((n_filter,), n_hidden => 1),

            RPadCircular(size(div_kernel,1)),
            RFluxConv(div_kernel),

            RDrop()
        )

    return EquiFluxModelSmall(SkipConnection(main, +))
end

Lux.initialparameters(rng::AbstractRNG, m::EquiFluxModelSmall) = (main=Lux.initialparameters(rng, m.main),)
Lux.initialstates(rng::AbstractRNG, m::EquiFluxModelSmall) = (main=Lux.initialstates(rng, m.main),)

function (m::EquiFluxModelSmall)(x, ps, st)
    y, newst = m.main(x, ps.main, st.main)
    return y, (main=newst,)
end;