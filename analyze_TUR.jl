using JLD2             # 用于读取 .jld2 文件
using DelimitedFiles   # 用于读取 .txt 文件 (sigma.txt)
using Statistics       # 用于计算 mean 和 var
using Plots            # 用于绘图
using Glob             # 用于查找所有轨迹文件

# 加载 Elastic 模块的入口文件
include("src/Elastic.jl")

# 显式导入需要的模块和类型
using .Elastic.Model      # 导入基础类型 (Field, MutableField, UnitCell, Atom...)
using .Elastic.Visualize  # 导入包含 Trajectory 定义的模块

function perform_tur_analysis(basepath, warm_up_steps, kB_value, Ts, N_tau_points_list, min_windows_for_stats)
    # 函数体开始...
    println("Analyzing results in: $basepath") # <--- 确保这里使用的是函数参数 basepath

    # --- 2. 加载熵产生率 sigma ---
    sigma_filepath = joinpath(basepath, "sigma.txt")
    sigma = try
        val = readdlm(sigma_filepath)[1, 1]
        if !isa(val, Number) || isnan(val) || val <= 0
            error("Invalid sigma value in $sigma_filepath: $val. Must be a positive number.")
        end
        Float64(val)
    catch e
        error("Error reading sigma file $sigma_filepath: $e")
    end
    println("Successfully loaded sigma = ", sigma)
    if sigma < 1e-9
        @warn "Sigma value ($sigma) is very close to zero. TUR analysis might be problematic."
    end

    # --- 3. 加载并合并稳态轨迹数据 ---
    println("Loading and merging trajectory files from $(joinpath(basepath, "traj"))...")
    traj_files = glob("traj_*.jld2", joinpath(basepath, "traj"))
    sort!(traj_files, by = x -> parse(Int, split(basename(x), '_')[2]))
    if isempty(traj_files)
        error("No trajectory files found in $(joinpath(basepath, "traj"))")
    end

    X_CM_stable_list = Float64[]
    nparticle_loaded = -1
    dt_loaded = -1.0
    trajseq_loaded = -1
    n_files_processed = 0
    n_points_total = 0
    n_points_stable = 0

    for filepath in traj_files
        println("Processing file: $filepath")
        skip_file = false  # 初始化跳过标志

        try
            jldopen(filepath, "r") do file
                # --- 元数据检查 ---
                if !all(haskey(file, k) for k in ["TrajList", "start_step", "end_step", "dt", "trajseq"])
                    @warn "Skipping $filepath: Missing required metadata keys."
                    skip_file = true  # 标记为跳过
                    return  # 退出 do 块
                end

                # --- 加载数据 ---
                TrajList_segment = read(file, "TrajList")
                start_step = read(file, "start_step")
                end_step = read(file, "end_step")
                dt_file = read(file, "dt")
                trajseq_file = read(file, "trajseq")

                # --- 元数据一致性检查 ---
                if nparticle_loaded == -1
                    # 首次加载校验
                    if isempty(TrajList_segment) || !isa(TrajList_segment[1], Trajectory) || !hasproperty(TrajList_segment[1], :rl)
                        @warn "Empty or invalid TrajList structure in first file $filepath. Skipping."
                        skip_file = true  # 标记为跳过
                        return  # 退出 do 块
                    end
                    nparticle_loaded = length(TrajList_segment)
                    dt_loaded = dt_file
                    trajseq_loaded = trajseq_file
                    println("  Detected nparticle=$nparticle_loaded, dt=$dt_loaded, trajseq=$trajseq_loaded")
                elseif nparticle_loaded != length(TrajList_segment) || dt_loaded != dt_file || trajseq_loaded != trajseq_file
                    @warn "Inconsistent metadata in $filepath. Skipping."
                    skip_file = true  # 标记为跳过
                    return  # 退出 do 块
                end

                # --- 数据有效性检查 ---
                if size(TrajList_segment[1].rl, 2) < 1
                    @warn "Trajectory data array (.rl) seems empty in $filepath. Skipping."
                    skip_file = true  # 标记为跳过
                    return  # 退出 do 块
                end

                # --- 处理有效数据 ---
                num_samples_in_segment = size(TrajList_segment[1].rl, 2)
                n_points_total += num_samples_in_segment

                points_added_this_file = 0
                for t_idx in 1:num_samples_in_segment
                    current_step = start_step + (t_idx - 1) * trajseq_loaded
                    if current_step > warm_up_steps
                        sum_x = 0.0
                        try
                            for p_idx in 1:nparticle_loaded
                                sum_x += TrajList_segment[p_idx].rl[1, t_idx]
                            end
                            push!(X_CM_stable_list, sum_x / nparticle_loaded)
                            points_added_this_file += 1
                        catch e
                            @warn "Error accessing trajectory data (t_idx=$t_idx) in $filepath. Skipping point. Error: $e"
                            continue
                        end
                    end
                end
                println("  Added $points_added_this_file stable points from this file (steps $start_step to $end_step).")
                n_points_stable += points_added_this_file
            end  # jldopen do 块结束

            # --- 根据标志跳过文件 ---
            if skip_file
                continue  # ✅ 正确：位于外层循环
            end

            n_files_processed += 1  # 记录成功处理文件数

        catch e
            @warn "Failed to load or process file $filepath: $e"
            continue  # ✅ 正确：位于外层循环
        end
    end

    println("Finished loading trajectory data.")
    println("Total trajectory files successfully processed: $n_files_processed")
    println("Total stable data points collected: $n_points_stable")
    if n_points_stable < 2
        error("Not enough stable data points ($n_points_stable) found after warm-up ($warm_up_steps). Cannot perform analysis.")
    end

    # --- 4. 计算流统计量和 TUR 比率 ---
    filter!(N -> N < n_points_stable ÷ min_windows_for_stats, N_tau_points_list)
    if isempty(N_tau_points_list)
        error("No valid tau windows possible with $n_points_stable stable points and min_windows=$min_windows_for_stats.")
    end
    println("Analyzing tau windows (in sample points): ", N_tau_points_list)

    results = []
    println("\nCalculating statistics...")
    time_interval_between_points = trajseq_loaded * dt_loaded

    for N_tau_points in N_tau_points_list
        tau_physical = N_tau_points * time_interval_between_points
        println("Processing N_tau_points = $N_tau_points (tau = $tau_physical)")

        currents_J_x_samples = Float64[]
        num_windows = div(n_points_stable - N_tau_points, 1)

        if num_windows < min_windows_for_stats
            @warn "Skipping N_tau=$N_tau_points: $num_windows windows < $min_windows_for_stats required."
            continue
        end
        println("  Using $num_windows non-overlapping windows.")

        for i in 0:(num_windows - 1)
            start_idx = i * N_tau_points + 1
            end_idx = start_idx + N_tau_points
            if end_idx > n_points_stable
                break  # 确保这段代码在循环内
            end
            delta_X_CM = X_CM_stable_list[end_idx] - X_CM_stable_list[start_idx]
            push!(currents_J_x_samples, delta_X_CM)
        end

        if length(currents_J_x_samples) < min_windows_for_stats
            @warn "Skipping N_tau=$N_tau_points: collected windows $(length(currents_J_x_samples)) < $min_windows_for_stats."
            continue
        end

        mean_J_x = mean(currents_J_x_samples)
        var_J_x = var(currents_J_x_samples, corrected=true)

        if abs(mean_J_x) < 1e-12 || var_J_x < 1e-18
            @warn "Skipping N_tau=$N_tau_points due to near-zero mean ($mean_J_x) or variance ($var_J_x)."
            continue
        end

        Sigma_tau = sigma * tau_physical
        if Sigma_tau <= 0
            @warn "Skipping N_tau=$N_tau_points due to non-positive Sigma_tau ($Sigma_tau)."
            continue
        end
        denominator = 2 * kB_value * Ts * mean_J_x^2
        if denominator == 0
            @warn "Skipping N_tau=$N_tau_points due to zero denominator in Q_x."
            continue
        end
        Q_x = (var_J_x * Sigma_tau) / denominator

        println("  tau=$tau_physical, <J>=$mean_J_x, Var(J)=$var_J_x, Στ=$Sigma_tau, Qx=$Q_x")
        push!(results, (tau_physical, mean_J_x, var_J_x, Sigma_tau, Q_x))

        if Q_x < 1.0
            println("  WARNING: Qx < 1.0 for N_tau=$N_tau_points. This may indicate a violation of the TUR.")
        end
    end

    if isempty(results)
        error("Failed to calculate valid statistics for any tau window.")
    end
    println("Finished calculating statistics.")

    return results, sigma, time_interval_between_points
end # 确保函数定义有对应的 end

# --- 在这里定义参数 ---
# 1. 用户配置参数
# ==============================================================================
# !!! 修改以下参数以匹配你的模拟设置 !!!

# 指定包含模拟输出的目录
basepath = "output/brownianparticle_3d_bias_deltax_50.0" # <--- 修改为你的路径

# 模拟中使用的预热步数
warm_up_steps = 200000 # <--- 保持或修改为你需要的

# 模拟中使用的系综参数
kB_value = 1.0
Ts = 1.0

# 分析用的时间窗口大小 (以 *采样点* 数定义)
N_tau_points_list = unique(round.(Int, exp10.(range(1, log10(5000000), length=100000))))  
# 计算方差所需的最少独立窗口数
min_windows_for_stats = 50

# (可选) 设置绘图后端
gr()
# ==============================================================================

# --- 调用分析函数 ---
try
    # 现在调用函数时，这些变量都已经定义好了
    results, sigma_loaded, time_interval = perform_tur_analysis(
        basepath, warm_up_steps, kB_value, Ts, N_tau_points_list, min_windows_for_stats
    )

    # --- 5. 可视化结果 ---
    println("\nGenerating plots...")

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
    savefig(p1, joinpath(basepath, "TUR_Qx_vs_tau.png"))
    println("Saved TUR_Qx_vs_tau.png")

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
        savefig(p2, joinpath(basepath, "TUR_Precision_vs_Cost.png"))
        println("Saved TUR_Precision_vs_Cost.png")
    else
        @warn "No valid data points for Precision vs Cost plot."
    end

    println("\nAnalysis script finished successfully.")
catch e
    println("\nAn error occurred during analysis:")
    showerror(stdout, e)
    println()
end