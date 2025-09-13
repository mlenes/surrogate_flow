using JLD2, Plots, Statistics

function show_unrolling(data_path, model, Xμ, Xσ, save_path)
	data = load(data_path)
	n_times = length(data["solution"])
	n_steps = n_times-1
	n_points = length(data["solution"][1])

	X = zeros(Float32, n_points, 1, n_steps)
	y = zeros(Float32, n_points, 1, n_steps)
	for t in 1:n_steps
	    X[:,:,t] .= data["solution"][t]
	    y[:,:,t] .= data["solution"][t+1]
	end

	X = (X .- Xμ) ./ Xσ
	y = (y .- Xμ) ./ Xσ

	output_times = data["times"][2:end]
	output_x = data["grid"]
	x0 = X[:,:,1]

	x = reshape(x0, size(x0, 1), size(x0, 2), 1)
    y_unroll = zeros(Float32, size(x0, 1), size(x0, 2), n_times)
    y_unroll[:,:,1] .= x0
    
    for t in 1:(n_times-1)
      	x = model(x)
        y_unroll[:,:,t+1] .= x
    end

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
	y = zeros(Float32, n_points, 1, n_steps)
	for t in 1:n_steps
	    X[:,:,t] .= data["solution"][t]
	    y[:,:,t] .= data["solution"][t+1]
	end

	X = (X .- Xμ) ./ Xσ
	y = (y .- Xμ) ./ Xσ

	output_times = data["times"][2:end]
	output_x = data["grid"]
	x0 = X[:,:,1]

	x = reshape(x0, size(x0, 1), size(x0, 2), 1)
    y_unroll = zeros(Float32, size(x0, 1), size(x0, 2), n_times)
    y_unroll[:,:,1] .= x0
    
    for t in 1:(n_times-1)
      	x = model(x)
        y_unroll[:,:,t+1] .= x
    end

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
	savefig(fig, save_path);
	display(fig)
end