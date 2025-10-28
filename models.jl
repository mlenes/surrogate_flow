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


# Custom convolution layer with static kernel to conserve mass
struct FluxConv{K} <: Lux.AbstractLuxLayer
    kernel::K
end

function FluxConv()
    kernel = reshape(Float32[0, -1, 1], :, 1, 1)
    return FluxConv(() -> copy(kernel))
end

Lux.initialparameters(::AbstractRNG, ::FluxConv) = NamedTuple()
Lux.initialstates(::AbstractRNG, m::FluxConv) = (kernel = m.kernel(),)

function (::FluxConv)(x, ps, st)
    y = NNlib.conv(x, st.kernel)
    return y, st
end


struct FluxConv2D{K} <: Lux.AbstractLuxLayer
    kernel::K
end

function FluxConv2D()
    x_kernel = Float32[0  0 0;
                       0 -1 1;
                       0  0 0]

    y_kernel = Float32[0  1 0;
                       0 -1 0;
                       0  0 0]

    kernel = reshape(cat(x_kernel, y_kernel, dims=3), 3, 3, 2, 1)
    return FluxConv2D(() -> copy(kernel))
end

Lux.initialparameters(::AbstractRNG, ::FluxConv2D) = NamedTuple()
Lux.initialstates(::AbstractRNG, m::FluxConv2D) = (kernel = m.kernel(),)

function (::FluxConv2D)(x, ps, st)
    y_x = NNlib.conv(x[:,:,1:2,:], st.kernel)
    y_y = NNlib.conv(x[:,:,3:4,:], st.kernel)
    y = cat(y_x, y_y, dims=3)
    return y, st
end


# This model predicts fluxes as its next to last step, and then predicts the updated velocity field from those fluxes
struct FluxModel{C} <: Lux.AbstractLuxLayer
    main::C
end

function FluxModel(n_filter, n_hidden)

    main = Chain(
        PadCircular(n_filter),
        Conv((n_filter,), 1=>n_hidden, swish),

        PadCircular(n_filter),
        Conv((n_filter,), n_hidden=>n_hidden, swish),

        PadCircular(n_filter),
        Conv((n_filter,), n_hidden=>n_hidden, swish),
        
        PadCircular(n_filter),
        Conv((n_filter,), n_hidden=>1),

        PadCircular(3),
        FluxConv()
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

    main = Chain(
        PadCircular2D(n_filter),
        Conv((n_filter,n_filter), 2=>n_hidden, swish),

        PadCircular2D(n_filter),
        Conv((n_filter,n_filter), n_hidden=>n_hidden, swish),

        PadCircular2D(n_filter),
        Conv((n_filter,n_filter), n_hidden=>n_hidden, swish),
        
        PadCircular2D(n_filter),
        Conv((n_filter,n_filter), n_hidden=>4),

        PadCircular2D(3),
        FluxConv2D()
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
    main = Chain(
        PadCircular(n_filter),
        Conv((n_filter,), 1=>n_hidden, swish),
        
        PadCircular(n_filter),
        Conv((n_filter,), n_hidden=>1),

        PadCircular(3),
        FluxConv()
    )

    return FluxModelSmall(SkipConnection(main, +))
end

Lux.initialparameters(rng::AbstractRNG, m::FluxModelSmall) = (main=Lux.initialparameters(rng, m.main),)
Lux.initialstates(rng::AbstractRNG, m::FluxModelSmall) = (main=Lux.initialstates(rng, m.main),)

function (m::FluxModelSmall)(x, ps, st)
    y, newst = m.main(x, ps.main, st.main)
    return y, (main=newst,)
end


struct FluxModelSmall2D{C} <: Lux.AbstractLuxLayer
    main::C
end

function FluxModelSmall2D(n_filter, n_hidden)

    main = Chain(
        PadCircular2D(n_filter),
        Conv((n_filter,n_filter), 2=>n_hidden, swish),
        
        PadCircular2D(n_filter),
        Conv((n_filter,n_filter), n_hidden=>4),

        PadCircular2D(3),
        FluxConv2D()
    )

    return FluxModelSmall2D(SkipConnection(main, +))
end

Lux.initialparameters(rng::AbstractRNG, m::FluxModelSmall2D) = (main=Lux.initialparameters(rng, m.main),)
Lux.initialstates(rng::AbstractRNG, m::FluxModelSmall2D) = (main=Lux.initialstates(rng, m.main),)

function (m::FluxModelSmall2D)(x, ps, st)
    y, newst = m.main(x, ps.main, st.main)
    return y, (main=newst,)
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


struct RLift <: Lux.AbstractLuxLayer
end

Lux.initialparameters(::AbstractRNG, ::RLift) = NamedTuple()
Lux.initialstates(::AbstractRNG, ::RLift) = NamedTuple()

function (::RLift)(x, ps, st)
    y = cat(x, -reverse(x, dims=1), dims=4)
    return y, st
end


function rotl90_2D(x::AbstractArray, k::Int)
    k_mod = mod(k, 4)
    if k_mod == 0
        return x
    elseif k_mod == 1
        return reverse(permutedims(x, (2,1,3,4)), dims=1)
    elseif k_mod == 2
        return reverse(reverse(x, dims=1), dims=2)
    elseif k_mod == 3
        return reverse(permutedims(x, (2,1,3,4)), dims=2)
    end
end


function VecRot90_2D(x::AbstractArray, k::Int)
    k_mod = mod(k, 4)
    rotated = rotl90_2D(x, k_mod)
    if k_mod == 0
        return rotated
    elseif k_mod == 1
        return cat(-rotated[:,:,2:2,:], rotated[:,:,1:1,:], dims=3)
    elseif k_mod == 2
        return cat(-rotated[:,:,1:1,:], -rotated[:,:,2:2,:], dims=3)
    elseif k_mod == 3
        return cat(rotated[:,:,2:2,:], -rotated[:,:,1:1,:], dims=3)
    end
end


struct RLift2D <: Lux.AbstractLuxLayer
end

Lux.initialparameters(::AbstractRNG, ::RLift2D) = NamedTuple()
Lux.initialstates(::AbstractRNG, ::RLift2D) = NamedTuple()

function (::RLift2D)(x, ps, st)
    y = cat(x, VecRot90_2D(x, 1), VecRot90_2D(x, 2),  VecRot90_2D(x, 3), dims=5)
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


struct RConv2D{C} <: Lux.AbstractLuxLayer
    conv::C
end

function RConv2D(kernel_size::Tuple, n_in_out::Pair, activation)
    n_in, n_out = n_in_out
    return RConv2D(Conv(kernel_size, n_in => n_out, activation))
end

function RConv2D(kernel_size::Tuple, n_in_out::Pair)
    n_in, n_out = n_in_out
    return RConv2D(Conv(kernel_size, n_in => n_out))
end

Lux.initialparameters(rng::AbstractRNG, m::RConv2D) = (conv=Lux.initialparameters(rng, m.conv),)
Lux.initialstates(rng::AbstractRNG, m::RConv2D) = (conv=Lux.initialstates(rng, m.conv),)

function (m::RConv2D)(x, ps, st)
    y_0,   st_conv = m.conv(x[:,:,:,:,1], ps.conv, st.conv)
    y_90,  st_conv = m.conv(x[:,:,:,:,2], ps.conv, st.conv)
    y_180, st_conv = m.conv(x[:,:,:,:,3], ps.conv, st.conv)
    y_270, st_conv = m.conv(x[:,:,:,:,4], ps.conv, st.conv)
    y = cat(y_0, y_90, y_180, y_270, dims=5)
    return y, (conv=st_conv,)
end


struct RDrop <: Lux.AbstractLuxLayer
end

Lux.initialparameters(::AbstractRNG, ::RDrop) = NamedTuple()
Lux.initialstates(::AbstractRNG, ::RDrop) = NamedTuple()

function (::RDrop)(x, ps, st)
    y = 0.5f0(x[:,:,:, 1] .+ -reverse(x[:,:,:,2], dims=1))
    return y, st
end

struct RDrop2D <: Lux.AbstractLuxLayer
end

Lux.initialparameters(::AbstractRNG, ::RDrop2D) = NamedTuple()
Lux.initialstates(::AbstractRNG, ::RDrop2D) = NamedTuple()

function (::RDrop2D)(x, ps, st)
    y = 0.25f0*sum([x[:,:,:,:,1], VecRot90_2D(x[:,:,:,:,2], 3), VecRot90_2D(x[:,:,:,:,3], 2), VecRot90_2D(x[:,:,:,:,4], 1)])
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


struct RPadCircular2D <: Lux.AbstractLuxLayer
    n_pad::Int

    function RPadCircular2D(n_filter::Int)
        return new(n_filter ÷ 2)
    end
end

Lux.initialparameters(::AbstractRNG, ::RPadCircular2D) = NamedTuple()
Lux.initialstates(::AbstractRNG, ::RPadCircular2D) = NamedTuple()

function (m::RPadCircular2D)(x, ps, st)
    n_pad = m.n_pad

    y = vcat(
        x[end-n_pad+1:end, :, :, :, :],
        x,
        x[1:n_pad, :, :, :, :]
    )

    y = hcat(
        y[:, end-n_pad+1:end, :, :, :],
        y,
        y[:, 1:n_pad, :, :, :]
    )

    return y, st
end


struct RFluxConv{C} <: Lux.AbstractLuxLayer
    conv::C
end

function RFluxConv()
    return RFluxConv(FluxConv())
end

Lux.initialparameters(rng::AbstractRNG, m::RFluxConv) = (conv=Lux.initialparameters(rng, m.conv),)
Lux.initialstates(rng::AbstractRNG, m::RFluxConv) = (conv=Lux.initialstates(rng, m.conv),)

function (m::RFluxConv)(x, ps, st)
    y_e, st_conv = m.conv(x[:,:,:,1], ps.conv, st.conv)
    y_r, st_conv = m.conv(x[:,:,:,2], ps.conv, st.conv)

    y = cat(y_e, y_r, dims=4)
    return y, (conv=st_conv,)
end


struct RFluxConv2D{C} <: Lux.AbstractLuxLayer
    conv::C
end

function RFluxConv2D()
    return RFluxConv2D(FluxConv2D())
end

Lux.initialparameters(rng::AbstractRNG, m::RFluxConv2D) = (conv=Lux.initialparameters(rng, m.conv),)
Lux.initialstates(rng::AbstractRNG, m::RFluxConv2D) = (conv=Lux.initialstates(rng, m.conv),)

function (m::RFluxConv2D)(x, ps, st)
    y_0, st_conv   = m.conv(x[:,:,:,:,1], ps.conv, st.conv)
    y_90, st_conv  = m.conv(x[:,:,:,:,2], ps.conv, st.conv)
    y_180, st_conv = m.conv(x[:,:,:,:,3], ps.conv, st.conv)
    y_270, st_conv = m.conv(x[:,:,:,:,4], ps.conv, st.conv)

    y = cat(y_0, y_90, y_180, y_270, dims=5)
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


struct EquiModel2D{M} <: Lux.AbstractLuxLayer
    main::M
end

function EquiModel2D(n_filter::Int, n_hidden::Int)
    main = Chain(
            RLift2D(),

            RPadCircular2D(n_filter),
            RConv2D((n_filter,n_filter), 2 => n_hidden, swish),

            RPadCircular2D(n_filter),
            RConv2D((n_filter,n_filter), n_hidden => n_hidden, swish),

            RPadCircular2D(n_filter),
            RConv2D((n_filter,n_filter), n_hidden => n_hidden, swish),

            RPadCircular2D(n_filter),
            RConv2D((n_filter,n_filter), n_hidden => 2),

            RDrop2D()
        )

    return EquiModel2D(SkipConnection(main, +))
end

Lux.initialparameters(rng::AbstractRNG, m::EquiModel2D) = (main=Lux.initialparameters(rng, m.main),)
Lux.initialstates(rng::AbstractRNG, m::EquiModel2D) = (main=Lux.initialstates(rng, m.main),)

function (m::EquiModel2D)(x, ps, st)
    y, newst = m.main(x, ps.main, st.main)
    return y, (main=newst,)
end


struct EquiFluxModel{M} <: Lux.AbstractLuxLayer
    main::M
end

function EquiFluxModel(n_filter::Int, n_hidden::Int)

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

            RPadCircular(3),
            RFluxConv(),

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


struct EquiFluxModel2D{M} <: Lux.AbstractLuxLayer
    main::M
end

function EquiFluxModel2D(n_filter::Int, n_hidden::Int)

    main = Chain(
            RLift2D(),

            RPadCircular2D(n_filter),
            RConv2D((n_filter,n_filter), 2 => n_hidden, swish),

            RPadCircular2D(n_filter),
            RConv2D((n_filter,n_filter), n_hidden => n_hidden, swish),

            RPadCircular2D(n_filter),
            RConv2D((n_filter,n_filter), n_hidden => n_hidden, swish),

            RPadCircular2D(n_filter),
            RConv2D((n_filter,n_filter), n_hidden => 4),

            RPadCircular2D(3),
            RFluxConv2D(),

            RDrop2D()
        )

    return EquiFluxModel2D(SkipConnection(main, +))
end

Lux.initialparameters(rng::AbstractRNG, m::EquiFluxModel2D) = (main=Lux.initialparameters(rng, m.main),)
Lux.initialstates(rng::AbstractRNG, m::EquiFluxModel2D) = (main=Lux.initialstates(rng, m.main),)

function (m::EquiFluxModel2D)(x, ps, st)
    y, newst = m.main(x, ps.main, st.main)
    return y, (main=newst,)
end


struct EquiFluxModelSmall{M} <: Lux.AbstractLuxLayer
    main::M
end

function EquiFluxModelSmall(n_filter::Int, n_hidden::Int)

    main = Chain(
            RLift(),

            RPadCircular(n_filter),
            RConv((n_filter,), 1 => n_hidden, swish),

            RPadCircular(n_filter),
            RConv((n_filter,), n_hidden => 1),

            RPadCircular(3),
            RFluxConv(),

            RDrop()
        )

    return EquiFluxModelSmall(SkipConnection(main, +))
end

Lux.initialparameters(rng::AbstractRNG, m::EquiFluxModelSmall) = (main=Lux.initialparameters(rng, m.main),)
Lux.initialstates(rng::AbstractRNG, m::EquiFluxModelSmall) = (main=Lux.initialstates(rng, m.main),)

function (m::EquiFluxModelSmall)(x, ps, st)
    y, newst = m.main(x, ps.main, st.main)
    return y, (main=newst,)
end


struct EquiFluxModelSmall2D{M} <: Lux.AbstractLuxLayer
    main::M
end

function EquiFluxModelSmall2D(n_filter::Int, n_hidden::Int)

    main = Chain(
            RLift2D(),

            RPadCircular2D(n_filter),
            RConv2D((n_filter,n_filter), 2 => n_hidden, swish),

            RPadCircular2D(n_filter),
            RConv2D((n_filter,n_filter), n_hidden => 4),

            RPadCircular2D(3),
            RFluxConv2D(),

            RDrop2D()
        )

    return EquiFluxModelSmall2D(SkipConnection(main, +))
end

Lux.initialparameters(rng::AbstractRNG, m::EquiFluxModelSmall2D) = (main=Lux.initialparameters(rng, m.main),)
Lux.initialstates(rng::AbstractRNG, m::EquiFluxModelSmall2D) = (main=Lux.initialstates(rng, m.main),)

function (m::EquiFluxModelSmall2D)(x, ps, st)
    y, newst = m.main(x, ps.main, st.main)
    return y, (main=newst,)
end;