println("Starting TUR analysis script (ensemble average for single particle)...")

# --- 0. 导入必要的包 ---
using JLD2
using DelimitedFiles
using Statistics
using Plots
using Glob
using DataFrames  # 可选，用于整理结果
ENV["GKSwstype"] = "100"

# --- 1. 加载必要的类型定义 (与主脚本一致) ---
try
    include("../juliaMD/src/Elastic.jl")
    using .Elastic.Model
    using .Elastic.Visualize
    println("Successfully included Elastic.jl and imported types.")
catch e
    println("="^40)
    @error "Failed to include Elastic.jl or import types." e
    println("="^40)
    exit(1)
end

# --- 2. 定义分析函数 ---
function perform_tur_analysis_ensemble(basepath, warm_up_steps, kB_value, Ts, N_tau_points_list, min_windows_for_stats)
    println("Analyzing results in: $basepath")
    println("Discarding first $warm_up_steps simulation steps for warm-up.")

    # --- 加载单粒子平均 Sigma ---
    sigma_filepath = joinpath(basepath, "sigma.txt")
    if !isfile(sigma_filepath)
        error("Sigma file not found: $sigma_filepath")
    end
    sigma = try
        val = parse(Float64, read(sigma_filepath, String))
        if isnan(val) || val < 0
            error("Invalid sigma value: $val")
        end
        val
    catch e
        error("Error reading sigma file $sigma_filepath: $e")
    end
    println("Successfully loaded single particle sigma = ", sigma)

    # --- 加载并处理轨迹数据 ---
    println("Loading trajectories from $(joinpath(basepath, "traj"))...")
    traj_files = glob("traj_*.jld2", joinpath(basepath, "traj"))
    sort!(traj_files, by = x -> parse(Float64, split(basename(x), '_')[2]))
    if isempty(traj_files)
        error("No trajectory files found.")
    end

    # 使用字典存储每个粒子的稳态 x 坐标列表
    particle_x_stable_data = Dict{Int, Vector{Float64}}()
    nparticle_loaded = -1
    dt_loaded = -1.0
    trajseq_loaded = -1
    n_files_processed = 0
    n_points_stable_total = 0  # 所有粒子稳态点总数

    for filepath in traj_files
        println("Processing file: $filepath")
        try
            jldopen(filepath, "r") do file
                TrajList_segment = read(file, "TrajList")
                start_step = haskey(file, "start_step") ? read(file, "start_step") : begin
                    @warn "File $filepath does not contain 'start_step', using 0 as default."
                    0
                end
                dt_file = read(file, "dt")
                trajseq_file = read(file, "trajseq")

                if nparticle_loaded == -1
                    nparticle_loaded = length(TrajList_segment)
                    dt_loaded = dt_file
                    trajseq_loaded = trajseq_file
                    for p_idx in 1:nparticle_loaded
                        particle_x_stable_data[p_idx] = Float64[]
                    end
                    println("  Detected nparticle=$nparticle_loaded, dt=$dt_loaded, trajseq=$trajseq_loaded")
                end

                num_samples = size(TrajList_segment[1].rl, 2)
                for t_idx in 1:num_samples
                    current_step = start_step + (t_idx - 1) * trajseq_loaded
                    if current_step > warm_up_steps
                        for p_idx in 1:nparticle_loaded
                            x_coord = TrajList_segment[p_idx].rl[1, t_idx]
                            push!(particle_x_stable_data[p_idx], x_coord)
                        end
                    end
                end
            end
            n_files_processed += 1
        catch e
            @warn "Failed processing $filepath: $e"
            continue
        end
    end

    println("Finished loading trajectory data.")
    println("Total trajectory files successfully processed: $n_files_processed")

    # 检查每个粒子是否有足够的数据点
    min_stable_points_per_particle = minimum(length(v) for v in values(particle_x_stable_data))
    println("Minimum stable data points found per particle: $min_stable_points_per_particle")
    if min_stable_points_per_particle < 2
        error("Not enough stable data points per particle.")
    end

    # --- 4. 计算系综平均的流统计量和 TUR 比率 ---
    filter!(N -> N < min_stable_points_per_particle ÷ min_windows_for_stats, N_tau_points_list)
    if isempty(N_tau_points_list)
        error("No valid tau windows possible.")
    end
    println("Analyzing tau windows (in sample points): ", N_tau_points_list)

    results = []
    println("\nCalculating statistics (ensemble average)...")
    time_interval = trajseq_loaded * dt_loaded

    for N_tau_points in N_tau_points_list
        tau_physical = N_tau_points * time_interval
        println("Processing N_tau_points = $N_tau_points (tau = $tau_physical)")

        all_displacements_ensemble = Float64[]

        for p_idx in 1:nparticle_loaded
            x_data = particle_x_stable_data[p_idx]
            n_particle_stable = length(x_data)

            if n_particle_stable < N_tau_points + 1
                continue
            end

            num_windows_particle = n_particle_stable - N_tau_points
            for i in 1:num_windows_particle
                start_idx = i
                end_idx = i + N_tau_points
                push!(all_displacements_ensemble, x_data[end_idx] - x_data[start_idx])
            end
        end

        total_windows_collected = length(all_displacements_ensemble)
        min_total_windows = min_windows_for_stats * nparticle_loaded
        if total_windows_collected < min_total_windows
            @warn "Skipping N_tau=$N_tau_points: collected total windows $total_windows_collected < $min_total_windows required."
            continue
        end
        println("  Using $total_windows_collected total displacement samples across all particles.")

        mean_J_x_ensemble = mean(all_displacements_ensemble)
        var_J_x_ensemble = var(all_displacements_ensemble, corrected=true)

        if abs(mean_J_x_ensemble) < 1e-12 || var_J_x_ensemble < 1e-18
            @warn "Skipping N_tau=$N_tau_points due to near-zero mean ($mean_J_x_ensemble) or variance ($var_J_x_ensemble)."
            continue
        end

        Sigma_tau = sigma * tau_physical
        if Sigma_tau <= 0
            @warn "Skipping N_tau=$N_tau_points due to non-positive Sigma_tau ($Sigma_tau)."
            continue
        end

        denominator = 2 * kB_value * Ts * mean_J_x_ensemble^2
        if denominator == 0
            @warn "Skipping N_tau=$N_tau_points due to zero denominator in Q_x."
            continue
        end

        Q_x = (var_J_x_ensemble * Sigma_tau) / denominator
        println("  tau=$tau_physical, <J_ens>=$mean_J_x_ensemble, Var(J_ens)=$var_J_x_ensemble, Στ=$Sigma_tau, Qx=$Q_x")
        push!(results, (tau_physical, mean_J_x_ensemble, var_J_x_ensemble, Sigma_tau, Q_x))
    end

    if isempty(results)
        error("Failed to calculate valid statistics for any tau window.")
    end
    println("Finished calculating statistics.")

    return results, sigma, time_interval
end

# --- 定义参数 ---
basepath = "output/brownianparticle_tiltedperiodic_bias_2"
using JSON
config=JSON.parsefile(joinpath(basepath, "config.json"))
maxstep =float(config["Simulation"]["maxstep"])
println("maxstep: $maxstep")
warm_up_steps = div(maxstep, 2) 
kB_value = 1.0
Ts = 1.0
N_tau_points_list = unique(round.(Int, exp10.(range(0, log10(div(maxstep,2)), length=1000))))  # 从10到500对数采样100个点
filter!(x -> 100 ≤ x ≤ maxstep, N_tau_points_list)  # 限制在图表实际范围
min_windows_for_stats = 10.0# 从 0.5 改为恒定阈值
gr()

# --- 调用分析函数并绘图 ---
try
    results, sigma_loaded, time_interval = perform_tur_analysis_ensemble(
        basepath, warm_up_steps, kB_value, Ts, N_tau_points_list, min_windows_for_stats
    )

    tau_plot = [r[1] for r in results]
    mean_J_plot = [r[2] for r in results]
    var_J_plot = [r[3] for r in results]
    Sigma_tau_plot = [r[4] for r in results]
    Qx_plot = [r[5] for r in results]

    plot_title_suffix = " (σ=$(round(sigma_loaded, digits=4)))"

    p1 = plot(tau_plot, Qx_plot,
        seriestype=:scatter, marker=:circle, markersize=4,
        xlabel="Time Window τ (physical units)",
        ylabel="TUR Ratio Qx = Var⋅Στ / (2kB⋅T⋅<J>²)",
        title="TUR Check for X-Current" * plot_title_suffix,
        legend=false, yaxis=:log10, framestyle=:box)
    hline!(p1, [1.0], linestyle=:dash, color=:red, linewidth=2)
    savefig(p1, joinpath(basepath, "TUR_Qx_vs_tau_ensemble.png"))
    println("Saved TUR_Qx_vs_tau_ensemble.png")

    precision = var_J_plot ./ (mean_J_plot.^2)
    cost = (2 * kB_value * Ts) ./ Sigma_tau_plot
    valid_indices = isfinite.(cost) .& isfinite.(precision) .& (cost .> 0) .& (precision .> 0)

    if sum(valid_indices) > 0
        p2 = plot(cost[valid_indices], precision[valid_indices],
            seriestype=:scatter, marker=:circle, markersize=4,
            xlabel="Thermodynamic Cost 2kB⋅T / Στ",
            ylabel="Relative Uncertainty Var(J) / <J>²",
            xscale=:log10, yscale=:log10,
            title="Precision vs Cost for X-Current" * plot_title_suffix,
            legend=false, framestyle=:box)
        savefig(p2, joinpath(basepath, "TUR_Precision_vs_Cost_ensemble.png"))
        println("Saved TUR_Precision_vs_Cost_ensemble.png")
    else
        @warn "No valid data points for Precision vs Cost plot."
    end

    # 在原有绘图代码后添加以下内容
    p3 = plot(tau_plot, mean_J_plot,
        seriestype=:scatter, marker=:circle, markersize=4,
        xlabel="Time Window τ (physical units)",
        ylabel="Mean Displacement <J>",
        title="Mean Displacement vs τ" * plot_title_suffix,
        legend=false, framestyle=:box)
    savefig(p3, joinpath(basepath, "Mean_vs_tau.png"))
    println("Saved Mean_vs_tau.png")

    p4 = plot(tau_plot, var_J_plot,
        seriestype=:scatter, marker=:circle, markersize=4,
        xlabel="Time Window τ (physical units)", 
        ylabel="Variance Var(J)",
        title="Variance vs τ" * plot_title_suffix,
        legend=false, framestyle=:box)  # 移除了 yaxis=:log10
    savefig(p4, joinpath(basepath, "Variance_vs_tau.png"))
    println("Saved Variance_vs_tau.png")

    precision = var_J_plot ./ (mean_J_plot.^2)
    cost = (2 * kB_value * Ts) ./ Sigma_tau_plot
    valid_indices = isfinite.(cost) .& isfinite.(precision) .& (cost .> 0) .& (precision .> 0)

    if sum(valid_indices) > 0
        p2 = plot(cost[valid_indices], precision[valid_indices],
            seriestype=:scatter, marker=:circle, markersize=4,
            xlabel="Thermodynamic Cost 2kB⋅T / Στ",
            ylabel="Relative Uncertainty Var(J) / <J>²",
            xscale=:log10, yscale=:log10,
            title="Precision vs Cost for X-Current" * plot_title_suffix,
            legend=false, framestyle=:box)
        savefig(p2, joinpath(basepath, "TUR_Precision_vs_Cost_ensemble.png"))
        println("Saved TUR_Precision_vs_Cost_ensemble.png")
    else
        @warn "No valid data points for Precision vs Cost plot."
    end

    println("\nAnalysis script (ensemble average) finished successfully.")
catch e
    println("\nAn error occurred during analysis:")
    showerror(stdout, e)
    println()
end