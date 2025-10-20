using JLD2, Plots, Statistics, Lux, Reactant, Enzyme, Optimisers, MLUtils, Random, NNlib, StatsPlots, LinearAlgebra
include("models.jl")

function show_unrolling(data_path, model, ps, st, Xμ, Xσ, save_path)
	data=load(data_path)

	y_pred = zeros(length(data["solution"]), length(data["grid"]))
	y_true = zeros(length(data["solution"]), length(data["grid"]))
	y_pred[1,:] = (data["solution"][1] .- Xμ) ./ Xσ
	y_true[1,:] = data["solution"][1]

	for t in 2:length(data["solution"])
	    u_prev = Float32.(reshape(y_pred[t-1,:], :, 1, 1))
	    
	    y_pred[t,:] .= model(u_prev, ps, st)[1][:,1,1,1]
	    y_true[t,:] .= Float32.(data["solution"][t])
	end
	y_pred = y_pred .* Xσ .+ Xμ

	output_grid = data["grid"]
	output_times = data["times"]
	begin
	    anim = @animate for t in 1:length(output_times)
	        p = plot(output_grid, y_true[t,:], xlabel="x", ylabel="u", label="target u(t=$(round(output_times[t],digits=2)))")
	        plot!(p, output_grid, y_pred[t,:], label="Unrolled estimate", linestyle=:dash, legend=:topright, ylim=(minimum(y_true), maximum(y_true)))
	        plot(p, size=(800,400))
	    end
	    gif(anim, save_path, fps=15)
	end
end

function show_plots(data_path, model, ps, st, Xμ, Xσ, save_path)
	data=load(data_path)

	y_pred = zeros(length(data["solution"]), length(data["grid"]))
	y_true = zeros(length(data["solution"]), length(data["grid"]))
	y_pred[1,:] = (data["solution"][1] .- Xμ) ./ Xσ
	y_true[1,:] = data["solution"][1]

	for t in 2:length(data["solution"])
	    u_prev = Float32.(reshape(y_pred[t-1,:], :, 1, 1))
	    
	    y_pred[t,:] .= model(u_prev, ps, st)[1][:,1,1,1]
	    y_true[t,:] .= Float32.(data["solution"][t])
	end
	y_pred = y_pred .* Xσ .+ Xμ

	output_grid = data["grid"]
	output_times = data["times"]

    errors = zeros(length(output_times))
	true_masses = zeros(length(output_times))
	pred_masses = zeros(length(output_times))
	for i in 1:length(output_times)
	    errors[i] = mean(abs2, y_pred[i,:] .- y_true[i,:])
	    true_masses[i] = sum(y_true[i,:])
	    pred_masses[i] = sum(y_pred[i,:])
	end

	p0 = plot(output_times, errors, xlabel="Time", ylabel="Error", label="MSE", title="Mean Squared Error in unrolled velocity")
	p1 = plot(output_times, true_masses, xlabel="Time", ylabel="Mass", label="Target")
	plot!(p1, output_times, pred_masses, label="Unrolled", title="Mass conservation of Target and Unrolled prediction")
	fig = plot(p0, p1, layout=(2,1), size=(800,800))
	savefig(fig, save_path)
	display(fig)
end

function full_jacobian_fd(model, ps, st, u; ε=1e-6)
    N = length(u)
    J = zeros(Float32, N, N)
    for j in 1:N
        e = zeros(Float32, N); e[j] = 1.0f0
        J[:, j] = (model(u .+ Float32(ε)*e, ps, st)[1] .- model(u, ps, st)[1]) ./ ε
    end
    return J
end

function show_spectral_density(model, ps, st, datapath, Xμ, Xσ, savepath)
	data = load(datapath)
	output_times = data["times"]

	begin
		anim = @animate for t in 1:length(data["solution"])
			u = (Float32.(reshape(data["solution"][t], :, 1, 1)) .- Xμ) ./ Xσ
			J = full_jacobian_fd(model, ps, st, u)
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
    λ = all_λ[idx]
    v = all_v[:, idx]

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

function generate_datasets(trainpaths, truthpaths, pairs_per_set)
	n_data = pairs_per_set * length(trainpaths)
	n_points = length(load(trainpaths[1])["grid"])

	X = zeros(Float32, n_points, 1, n_data)
	y = zeros(Float32, n_points, 1, n_data)

	count = 1
	for i in 1:length(trainpaths)
		X_data = load(trainpaths[i])
		y_data = load(truthpaths[i])

		n_times = length(X_data["times"])
		pair_times = rand(1:n_times-1, pairs_per_set)
		
		for t in pair_times
			X[:,:,count] .= X_data["solution"][t]

			y[:,:,count] .= y_data["solution"][t+1]
			count+=1
		end
	end

	Xμ, Xσ = mean(X), std(X)
	Xμ = 0 # The equivariant model doesnt like mean subtracting.
	X = (X .- Xμ) ./ Xσ
	y = (y .- Xμ) ./ Xσ

	return X, y, Xμ, Xσ
end

function burgers_FV(nx, L, ν, k, u_mean, u_amplitude, noise_strength, t_end, cfl; nt=10000)
	Δx=L/nx
	x=range(Δx/2, L - Δx/2, length=nx)

	u0(x) = u_mean .+ u_amplitude.*cos.(x*k*pi/L)
	Δt=t_end/nt

	function convective_flux(u)
	    n = length(u)
	    flux = zeros(n)

	    u_prev = circshift(u, 1)
	    u_next = circshift(u, -1)

	    for i in 1:n
	        uL = u[i]
	        uR = u_next[i]

	        alpha = max(abs(uL), abs(uR))
	        flux_LR = 0.25*(uL^2 + uR^2) - 0.5*alpha*(uR-uL)

	        uL_prev = u_prev[i]
	        uR_prev = u[i]

	        alpha_prev = max(abs(uL_prev), abs(uR_prev))
	        flux_prev = 0.25*(uL_prev^2 + uR_prev^2) - 0.5*alpha_prev*(uR_prev - uL_prev)
	        
	        flux[i] = (-1/Δx) * (flux_LR - flux_prev)
	    end

	    return flux
	end

	function diffusive_flux(u)
	    n = length(u)
	    flux = zeros(n)

	    u_prev = circshift(u, 1)
	    u_next = circshift(u, -1)

	    for i in 1:n
	        flux[i] = (ν/Δx^2) * (u_prev[i] - 2u[i] + u_next[i])
	    end

	    return flux
	end

	R(u) = convective_flux(u) .+ diffusive_flux(u)

	function rk3_step(u, Δt)
	    k1 = R(u)
	    u1 = u .+ Δt .* k1

	    k2 = R(u1)
	    u2 = 0.75*u .+ 0.25*(u1 .+ Δt .* k2)

	    k3 = R(u2)
	    u_next = (1/3)*u .+ (2/3)*(u2 .+ Δt .* k3)

	    return u_next
	end

	sol = [u0(x)]
	current_u = u0(x)

	for n in 1:nt
	    current_u = rk3_step(current_u, Δt)
	    noised = ((rand(nx).-0.5)*2*noise_strength) .+ copy(current_u)
	    push!(sol, noised)
	end

	dt_out_target = cfl*Δx/abs(u_mean)
	t_step = Int(floor(dt_out_target / Δt))

	solution = []
	for t_idx in 1:t_step:length(sol)
	    push!(solution, sol[t_idx])
	end

	return solution
end
;