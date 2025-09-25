using JLD2, Plots, Statistics, Lux, Reactant, Enzyme, Optimisers, MLUtils, Random, NNlib, StatsPlots, LinearAlgebra
include("models.jl")

function show_unrolling(data_path, model, Xμ, Xσ, save_path)
	data = load(data_path)
	n_times = length(data["solution"])
	n_steps = n_times-1
	n_points = length(data["solution"][1])

	X = zeros(Float32, n_points, 1, n_steps)
	for t in 1:n_steps
	    X[:,:,t] .= data["solution"][t]
	end

	X_norm = (X .- Xμ) ./ Xσ

	output_times = data["times"][2:end]
	output_x = data["grid"]
	x0 = X_norm[:,:,1]

	x = reshape(x0, size(x0, 1), size(x0, 2), 1)
    y_unroll = zeros(Float32, size(x0, 1), size(x0, 2), n_times)
    y_unroll[:,:,1] .= x0
    
    for t in 1:(n_times-1)
      	x = model(x)
        y_unroll[:,:,t+1] .= x
    end

    y_unroll = (y_unroll .* Xσ) .+ Xμ

    begin
	    anim = @animate for i in 1:length(output_times)
	        p1 = plot(output_x, X[:,1,i], xlabel="X", ylabel="u", label="target u(t=$(round(output_times[i],digits=2)))")
	        plot!(output_x, y_unroll[:,1,i], label="Unrolled estimate", linestyle=:dash, legend=:topright, ylim=(minimum(X),maximum(X)))
	        plot(p1, size=(800,400))
	    end
	    gif(anim, save_path, fps=15)
	end 
end

function show_plots(data_path, model, Xμ, Xσ, save_path)
	data = load(data_path)
	n_times = length(data["solution"])
	n_steps = n_times-1
	n_points = length(data["solution"][1])

	X = zeros(Float32, n_points, 1, n_steps)
	for t in 1:n_steps
	    X[:,:,t] .= data["solution"][t]
	end

	X_norm = (X .- Xμ) ./ Xσ

	output_times = data["times"][2:end]
	x0 = X_norm[:,:,1]

	x = reshape(x0, size(x0, 1), size(x0, 2), 1)
    y_unroll = zeros(Float32, size(x0, 1), size(x0, 2), n_times)
    y_unroll[:,:,1] .= x0
    
    for t in 1:(n_times-1)
      	x = model(x)
        y_unroll[:,:,t+1] .= x
    end

    y_unroll = (y_unroll .* Xσ) .+ Xμ

    errors = zeros(n_steps)
	data_masses = zeros(n_steps)
	unrolled_masses = zeros(n_steps)
	for i in 1:n_steps
	    errors[i] = mean(abs2, y_unroll[:,1,i] .- X[:,1,i])
	    data_masses[i] = sum(X[:,1,i])
	    unrolled_masses[i] = sum(y_unroll[:,1,i])
	end

	p0 = plot(output_times, errors, xlabel="Time", ylabel="Error", label="MSE", title="Mean Squared Error in unrolled velocity")
	p1 = plot(output_times, data_masses, xlabel="Time", ylabel="Mass", label="Target")
	plot!(p1, output_times, unrolled_masses, label="Unrolled", title="Mass conservation of Target and Unrolled prediction")
	fig = plot(p0, p1, layout=(2,1), size=(800,800))
	savefig(fig, save_path)
	display(fig)
end

function full_jacobian_fd(model, u; ε=1e-6)
    N = length(u)
    J = zeros(Float32, N, N)
    for j in 1:N
        e = zeros(Float32, N); e[j] = 1.0f0
        J[:, j] = (model(u .+ Float32(ε)*e) .- model(u)) ./ ε
    end
    return J
end

function show_spectral_density(model, datapath, Xμ, Xσ, savepath)
	data = load(datapath)
	output_times = data["times"]

	begin
		anim = @animate for t in 1:length(data["solution"])
			u = (Float32.(reshape(data["solution"][t], :, 1, 1)) .- Xμ) ./ Xσ
			J = full_jacobian_fd(model, u)
			λ, v = eigen(J)
			λ_max = maximum(abs.(λ))
			λ_min = minimum(abs.(λ))
			d = density(abs.(λ), xlabel="Abs(λ)", xticks=20, legend=:topright, ylim=(0,3), ylabel="Density", title="Eigenvalue Density t=$(round(output_times[t],digits=2))", label="Base, |λ|_max = $(round(λ_max, digits=3))", xlim=(0, 2))
			fig = plot(d, size=(800,400))
		end
		gif(anim, savepath, fps=15)
	end
end

function show_max_perturbation(model, datapath, Xμ, Xσ, savepath)
    data=load(datapath)
    u0 = Float32.(reshape(data["solution"][1], :, 1, 1))
    u0_norm = (u0 .- Xμ) ./ Xσ

    J = full_jacobian_fd(model, u0_norm)
    all_λ, all_v = eigen(J)

    idx = argmax(abs.(all_λ))
    λ = all_λ[40]
    v = all_v[:, 40]

    v_real = real(v); v_imag = imag(v)
    v_pert = v_real .+ v_imag

    ϵ = 1e-3
    u_perturbed = Float32.(u0_norm .+ ϵ .* reshape(v_pert, :, 1 ,1 ))

    T = length(data["solution"])
    trajectory_ref = [u0_norm]
    trajectory_pert = [u_perturbed]

    u_ref = u0_norm
    u_pert = u_perturbed

    for t in 1:T
        u_ref = model(u_ref)
        u_pert = model(u_pert)

        push!(trajectory_ref, u_ref)
        push!(trajectory_pert, u_pert)
    end

    for i in 1:length(trajectory_ref)
    	trajectory_ref[i] = (trajectory_ref[i] .* Xσ) .+ Xμ
    	trajectory_pert[i] = (trajectory_pert[i] .* Xσ) .+ Xμ
    end

    begin
        anim = @animate for i in 1:T
            p1 = plot(data["grid"], vec(trajectory_ref[i]), xlabel="x", ylabel="u", label="Reference", legend=:topright, ylim=(minimum(u0), maximum(u0)), title="Perturbed by mode with |λ|=$(norm(λ)) at t=0")
            plot!(data["grid"], vec(trajectory_pert[i]), label="Perturbed")
            plot(p1, size=(800,400))
        end
        gif(anim, savepath, fps=15)
    end
end