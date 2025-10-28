using JLD2, Plots, Statistics, Lux, Reactant, Enzyme, Optimisers, MLUtils, Random, NNlib, StatsPlots, LinearAlgebra
include("models.jl")

function show_unrolling(data_path, model, ps, st, Xμ, Xσ, save_path; do_norm=true)
	data=load(data_path)

	y_pred = zeros(length(data["solution"]), length(data["grid"]))
	y_true = zeros(length(data["solution"]), length(data["grid"]))

	if do_norm
		y_pred[1,:] = (data["solution"][1] .- Xμ) ./ Xσ
	else
		y_pred[1,:] = data["solution"][1]
	end
	
	y_true[1,:] = data["solution"][1]

	for t in 2:length(data["solution"])
	    u_prev = Float32.(reshape(y_pred[t-1,:], :, 1, 1))
	    
	    y_pred[t,:] .= model(u_prev, ps, st)[1][:,1,1]
	    y_true[t,:] .= Float32.(data["solution"][t])
	end
	if do_norm
		y_pred = y_pred .* Xσ .+ Xμ
	end

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

function show_unrolling_heatmap(data_path, model, ps, st, Xμ, Xσ, save_path; do_norm=true)
	data=load(data_path)

	y_pred = zeros(length(data["times"]), length(data["grid_x"]), length(data["grid_y"]), 2)
	y_true = zeros(length(data["times"]), length(data["grid_x"]), length(data["grid_y"]), 2)

	if do_norm
		y_pred[1,:,:,1] = (data["u_solution"][1] .- Xμ) ./ Xσ
		y_pred[1,:,:,2] = (data["v_solution"][1] .- Xμ) ./ Xσ
	else
		y_pred[1,:,:,1] = data["u_solution"][1]
		y_pred[1,:,:,2] = data["v_solution"][1]
	end
	
	y_true[1,:,:,1] = data["u_solution"][1]
	y_true[1,:,:,2] = data["v_solution"][1]

	for t in 2:length(data["times"])
	    sol_prev = Float32.(reshape(y_pred[t-1,:,:,:], size(y_pred,2), size(y_pred,3), 2, 1))
	    
	    y_pred[t,:,:,:] .= model(sol_prev, ps, st)[1][:,:,:,1]
	    y_true[t,:,:,1] .= Float32.(data["u_solution"][t])
	    y_true[t,:,:,2] .= Float32.(data["v_solution"][t])
	end

	if do_norm
		y_pred = y_pred .* Xσ .+ Xμ
	end

	grid_x = data["grid_x"]
	grid_y = data["grid_y"]
	times = data["times"]

	min_u=minimum(y_true[:,:,:,1])
	max_u=maximum(y_true[:,:,:,1])
	min_v=minimum(y_true[:,:,:,2])
	max_v=maximum(y_true[:,:,:,2])

	begin
	    anim = @animate for t in 1:length(times)
	        h_x_true = heatmap(grid_x, grid_y, y_true[t,:,:,1], title="u target t=$(round(times[t],digits=2))", clims=(min_u,max_u))
	        h_x_pred = heatmap(grid_x, grid_y, y_pred[t,:,:,1], title="u pred", clims=(min_u,max_u))
	        h_y_true = heatmap(grid_x, grid_y, y_true[t,:,:,2], title="v target", clims=(min_v,max_v))
	        h_y_pred = heatmap(grid_x, grid_y, y_pred[t,:,:,2], title="v pred", clims=(min_v,max_v))
	        plot(h_x_true, h_x_pred, h_y_true, h_y_pred, layout=(2,2), size=(1000,800), xlabel="x", ylabel="y")
	    end
	    gif(anim, save_path, fps=15)
	end
end

function show_plots(data_path, model, ps, st, Xμ, Xσ, save_path; do_norm=true)
	data=load(data_path)

	y_pred = zeros(length(data["solution"]), length(data["grid"]))
	y_true = zeros(length(data["solution"]), length(data["grid"]))
	
	if do_norm
		y_pred[1,:] = (data["solution"][1] .- Xμ) ./ Xσ
	else
		y_pred[1,:] = data["solution"][1]
	end

	y_true[1,:] = data["solution"][1]

	for t in 2:length(data["solution"])
	    u_prev = Float32.(reshape(y_pred[t-1,:], :, 1, 1))
	    
	    y_pred[t,:] .= model(u_prev, ps, st)[1][:,1,1]
	    y_true[t,:] .= Float32.(data["solution"][t])
	end
	if do_norm
		y_pred = y_pred .* Xσ .+ Xμ
	end

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

	p0 = plot(output_times, errors, xlabel="Time", ylabel="MSE", label="MSE", title="Mean Squared Error in unrolled velocity")
	p1 = plot(output_times, true_masses, xlabel="Time", ylabel="Mass", label="Target")
	plot!(p1, output_times, pred_masses, label="Unrolled", title="Mass conservation of Target and Unrolled prediction")
	fig = plot(p0, p1, layout=(2,1), size=(800,800))
	savefig(fig, save_path)
	display(fig)
end

function show_plots_2D(data_path, model, ps, st, Xμ, Xσ, save_path; do_norm=true)
	data=load(data_path)

	y_pred = zeros(length(data["times"]), length(data["grid_x"]), length(data["grid_y"]), 2)
	y_true = zeros(length(data["times"]), length(data["grid_x"]), length(data["grid_y"]), 2)

	if do_norm
		y_pred[1,:,:,1] = (data["u_solution"][1] .- Xμ) ./ Xσ
		y_pred[1,:,:,2] = (data["v_solution"][1] .- Xμ) ./ Xσ
	else
		y_pred[1,:,:,1] = data["u_solution"][1]
		y_pred[1,:,:,2] = data["v_solution"][1]
	end
	
	y_true[1,:,:,1] = data["u_solution"][1]
	y_true[1,:,:,2] = data["v_solution"][1]

	for t in 2:length(data["times"])
	    sol_prev = Float32.(reshape(y_pred[t-1,:,:,:], size(y_pred,2), size(y_pred,3), 2, 1))
	    
	    y_pred[t,:,:,:] .= model(sol_prev, ps, st)[1][:,:,:,1]
	    y_true[t,:,:,1] .= Float32.(data["u_solution"][t])
	    y_true[t,:,:,2] .= Float32.(data["v_solution"][t])
	end

	if do_norm
		y_pred = y_pred .* Xσ .+ Xμ
	end

	grid_x = data["grid_x"]
	grid_y = data["grid_y"]
	times = data["times"]

	errors_x = zeros(length(times))
	errors_y = zeros(length(times))
	true_masses_x = zeros(length(times))
	true_masses_y = zeros(length(times))
	pred_masses_x = zeros(length(times))
	pred_masses_y = zeros(length(times))

	for t in 1:length(times)
		errors_x[t] = mean(abs2, y_pred[t,:,:,1] .- y_true[t,:,:,1])
		errors_y[t] = mean(abs2, y_pred[t,:,:,2] .- y_true[t,:,:,2])
		true_masses_x[t] = sum(y_true[t,:,:,1])
		true_masses_y[t] = sum(y_true[t,:,:,2])
		pred_masses_x[t] = sum(y_pred[t,:,:,1])
		pred_masses_y[t] = sum(y_pred[t,:,:,2])
	end

	p0 = plot(times, errors_x, xlabel="Time", ylabel="MSE", label="Err u", title="Mean Squared Error in unrolled velocity")
	plot!(p0, times, errors_y, label="Err v")
	p1 = plot(times, true_masses_x, xlabel="Time", ylabel="Mass", label="Target u mass")
	plot!(p1, times, pred_masses_x, label="Unrolled u mass", title="Mass conservation of Target and Unrolled prediction", linestyle=:dash)
	p2 = plot(times, true_masses_y, label="Target v mass", xlabel="Time", ylabel="Mass")
	plot!(p2, times, pred_masses_y, label="Unrolled v mass", title="Mass conservation of Target and Unrolled prediction", linestyle=:dash)
	fig = plot(p0, p1, p2, layout=(3,1), size=(800,1200))
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

function show_spectral_density(model, ps, st, datapath, Xμ, Xσ, savepath; do_norm=true)
	data = load(datapath)
	output_times = data["times"]

	begin
		anim = @animate for t in 1:length(data["solution"])
			if do_norm
				u = (Float32.(reshape(data["solution"][t], :, 1, 1)) .- Xμ) ./ Xσ
			else
				u = Float32.(reshape(data["solution"][t], :, 1, 1))
			end
			
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

function show_max_perturbation(model, datapath, Xμ, Xσ, savepath; do_norm=true)
    data=load(datapath)
    u0 = Float32.(reshape(data["solution"][1], :, 1, 1))
    if do_norm
    	u0_norm = (u0 .- Xμ) ./ Xσ
    else
    	u0_norm = u0
    end

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
    	if do_norm
	    	trajectory_ref[i] = (trajectory_ref[i] .* Xσ) .+ Xμ
	    	trajectory_pert[i] = (trajectory_pert[i] .* Xσ) .+ Xμ
	    else
	    	trajectory_ref[i] = trajectory_ref[i]
	    	trajectory_pert[i] = trajectory_pert[i]
	    end
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

function generate_datasets(trainpaths, truthpaths, pairs_per_set; do_norm=true)
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
	if do_norm
		X = (X .- Xμ) ./ Xσ
		y = (y .- Xμ) ./ Xσ
	end

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


function burgers_FV_2D(nx, ny, Lx, Ly, ν, kx, ky, u_mean, u_amplitude, v_amplitude, noise_strength, t_end, cfl; nt=10000)
	Δx=Lx/nx; Δy=Ly/ny

	x = range(Δx/2, Lx - Δx/2, length=nx)
	y = range(Δy/2, Ly - Δy/2, length=ny)

	function initial_condition(x, y)
	    u0 = [u_mean + u_amplitude * cos(kx * π * xi / Lx) * cos(ky * π * yi / Ly) for yi in y, xi in x]
	    v0 = [u_amplitude * sin(kx * π * xi / Lx) * sin(ky * π * yi / Ly) for yi in y, xi in x]
	    return u0, v0
	end

	u0, v0 = initial_condition(x, y)

	Δt = t_end/nt

	circshift2(u, sx, sy) = circshift(u, (sy, sx))

	function convective_flux(u, v)
	    nx, ny = size(u)
	    dudt = zeros(nx, ny)
	    dvdt = zeros(nx, ny)

	    u_xp = circshift2(u, -1, 0)
	    u_xm = circshift2(u, 1, 0)
	    v_yp = circshift2(v, 0, -1)
	    v_ym = circshift2(v, 0, 1)

	    for j in 1:ny, i in 1:nx
	        # x direction
	        uL, uR = u[i,j], u_xp[i,j]
	        alpha_x = max(abs(uL), abs(uR))
	        flux_x_LR = 0.25(uL^2 + uR^2) - 0.5alpha_x*(uR - uL)

	        uL_prev, uR_prev = u_xm[i,j], u[i,j]
	        alpha_x_prev = max(abs(uL_prev), abs(uR_prev))
	        flux_x_prev = 0.25(uL_prev^2 + uR_prev^2) - 0.5alpha_x_prev*(uR_prev - uL_prev)

	        # y direction
	        vL, vR = v[i,j], v_yp[i,j]
	        alpha_y = max(abs(vL), abs(vR))
	        flux_y_LR = 0.25(vL^2 + vR^2) - 0.5alpha_y*(vR - vL)

	        vL_prev, vR_prev = v_ym[i,j], v[i,j]
	        alpha_y_prev = max(abs(vL_prev), abs(vR_prev))
	        flux_y_prev = 0.25(vL_prev^2 + vR_prev^2) - 0.5alpha_y_prev*(vR_prev - vL_prev)

	        dudt[i,j] = (-1/Δx)*(flux_x_LR - flux_x_prev) + (-1/Δy)*(flux_y_LR - flux_y_prev)
	        dvdt[i,j] = (-1/Δx)*(flux_x_LR - flux_x_prev) + (-1/Δy)*(flux_y_LR - flux_y_prev)
	    end

	    return dudt, dvdt
	end

	function diffusive_flux(u)
	    u_xp = circshift2(u, -1, 0)
	    u_xm = circshift2(u, 1, 0)
	    u_yp = circshift2(u, 0, -1)
	    u_ym = circshift2(u, 0, 1)
	    return (ν/Δx^2)*(u_xp - 2u + u_xm) + (ν/Δy^2)*(u_yp - 2u + u_ym)
	end

	function R(u, v)
	    du_conv, dv_conv = convective_flux(u, v)
	    du_diff = diffusive_flux(u)
	    dv_diff = diffusive_flux(v)
	    return du_conv .+ du_diff, dv_conv .+ dv_diff
	end

	function rk3_step(u, v, Δt)
	    k1u, k1v = R(u,v)
	    u1 = u .+ Δt .* k1u
	    v1 = v .+ Δt .* k1v

	    k2u, k2v = R(u1, v1)
	    u2 = 0.75u .+ 0.25(u1 .+ Δt .* k2u)
	    v2 = 0.75v .+ 0.25(v1 .+ Δt .* k2v)

	    k3u, k3v = R(u2, v2)
	    u_next = (1/3)*u .+ (2/3)*(u2 .+ Δt .* k3u)
	    v_next = (1/3)*v .+ (2/3)*(v2 .+ Δt .* k3v)

	    return u_next, v_next
	end

	u = copy(u0)
	v = copy(v0)
	sol_u = [copy(u)]
	sol_v = [copy(v)]

	for n in 1:nt
	    u, v = rk3_step(u, v, Δt)
	    u .+= (rand(nx, ny) .- 0.5)*2*noise_strength
	    v .+= (rand(nx, ny) .- 0.5)*2*noise_strength
	    push!(sol_u, copy(u))
	    push!(sol_v, copy(v))
	end

	dt_out_target = cfl * min(Δx, Δy) / abs(u_mean)
	t_step = Int(floor(dt_out_target / Δt))

	u_solution = []
	v_solution = []
	for t_idx in 1:t_step:length(sol_u)
	    push!(u_solution, sol_u[t_idx])
	    push!(v_solution, sol_v[t_idx])
	end

	return u_solution, v_solution
end
;