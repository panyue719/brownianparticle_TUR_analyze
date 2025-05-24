# 分析两态周期性势场布朗粒子模拟输出
using JLD2             # 用于读取 .jld2 文件
using DelimitedFiles   # 用于读取 .txt 文件
using Statistics       # 用于计算 mean 和 var
using Plots            # 用于绘图
using Glob             # 用于查找所有轨迹文件
using JSON             # 用于读取 Config.json
using Printf          # 用于格式化输出

# 在文件开头添加GR后端选择
gr() # 使用GR作为绘图后端

# 加载 Elastic 模块的入口文件
include("src/Elastic.jl")

# 显式导入需要的模块和类型
using .Elastic.Model      # 导入基础类型 (Atom, UnitCell, etc.)
using .Elastic.Visualize  # 导入包含 Trajectory 定义的模块
using StaticArrays       # SVector 支持

# 定义一个辅助函数从 Config.json 中读取参数
function read_config_value(config_filepath::String, key_path::Vector{String})
    try
        config_data = JSON.parsefile(config_filepath)
        current_level = config_data
        for key_segment in key_path
            if haskey(current_level, key_segment)
                current_level = current_level[key_segment]
            else
                # 尝试查找其他可能的路径
                alternative_keys = Dict(
                    "warm_up_steps" => ["Simulation", "maxstep"] # 如果找不到直接的warm_up_steps，尝试获取maxstep然后除以2
                )
                
                if haskey(alternative_keys, key_segment)
                    alt_path = alternative_keys[key_segment]
                    # 尝试通过替代路径获取值
                    alt_level = config_data
                    for alt_key in alt_path
                        if haskey(alt_level, alt_key)
                            alt_level = alt_level[alt_key]
                        else
                            error("Alternative key '$alt_key' also not found")
                        end
                    end
                    
                    # 对于特殊情况的处理，如warm_up_steps = maxstep / 2
                    if key_segment == "warm_up_steps"
                        return div(alt_level, 2)
                    end
                    
                    return alt_level
                else
                    error("Key '$key_segment' not found in path $(join(key_path, " -> "))")
                end
            end
        end
        return current_level
    catch e
        error("Error reading or parsing $config_filepath: $e")
    end
end

# 更灵活地从摘要文件读取sigma值
function read_sigma_from_summary(summary_filepath::String)
    # 首先尝试读取修正后的熵产率文件
    corrected_sigma_filepath = replace(summary_filepath, "tur_summary_trap_power.txt" => "tur_summary_corrected.txt")
    
    if isfile(corrected_sigma_filepath)
        println("找到修正后的熵产率文件: $corrected_sigma_filepath")
        # 尝试从修正文件中读取熵产率
        file_content = read(corrected_sigma_filepath, String)
        
        # 新的关键字列表，针对修正后的文件
        corrected_keys = [
            "Corrected entropy production rate:",
            "Corrected entropy production rate (σ):",
            "σ ="
        ]
        
        # 查找修正后的熵产率
        for line in split(file_content, '\n')
            for key in corrected_keys
                if occursin(key, line)
                    println("DEBUG: 找到可能包含修正熵产率的行: $line")
                    m = match(r"[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?", line)
                    if m !== nothing
                        value_str = m.match
                        try
                            value = parse(Float64, value_str)
                            println("成功读取修正熵产率值: $value")
                            return value
                        catch e
                            println("解析'$value_str'为Float64时出错: $e")
                        end
                    end
                end
            end
        end
    end
    
    # 如果找不到修正的熵产率文件或无法解析，回退到原始方法
    println("未找到修正熵产率或解析失败，尝试使用原始文件...")
    
    # 尝试多种可能的关键字格式
    sigma_keys = [
        "Sigma_entropy_production_rate_per_particle_LJ:", 
        "Sigma_entropy_production_rate_per_particle_LJ=",
        "Sigma_entropy_production_rate_per_particle_LJ ",
        "sigma_entropy_rate_per_particle=",
        "sigma_entropy_rate_per_particle:",
        "entropy production rate per particle:",
        "entropy production rate:",
        "sigma:",
        "sigma="
    ]
    
    if !isfile(summary_filepath)
        error("Summary file not found: $summary_filepath")
    end
    
    # 读取整个文件内容
    file_content = read(summary_filepath, String)
    println("DEBUG: Reading summary file: $summary_filepath")
    println("DEBUG: First 200 chars of file: $(file_content[1:min(200, length(file_content))])")
    
    # 逐行读取文件寻找sigma值
    for line in split(file_content, '\n')
        # 检查每个可能的关键字
        for key in sigma_keys
            if occursin(key, line)
                println("DEBUG: Found potential sigma line: $line")
                # 提取数值部分 - 查找数字模式
                m = match(r"[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?", line)
                if m !== nothing
                    value_str = m.match
                    try
                        value = parse(Float64, value_str)
                        println("DEBUG: Successfully parsed sigma value: $value")
                        return value
                    catch e
                        println("DEBUG: Failed to parse '$value_str' as Float64: $e")
                    end
                end
            end
        end
    end
    
    # 如果找不到sigma值，尝试直接读取文件内容作为单个数字
    try
        # 可能文件只包含一个数字
        value = parse(Float64, strip(file_content))
        println("DEBUG: Parsed entire file as a single value: $value")
        return value
    catch e
        println("DEBUG: Could not parse file as single value: $e")
    end
    
    # 如果还是找不到，尝试读取Config.json中预先计算的熵产生率
    try
        config_path = joinpath(dirname(summary_filepath), "Config.json")
        if isfile(config_path)
            config = JSON.parsefile(config_path)
            # 尝试不同的可能路径
            possible_paths = [
                ["DerivedProperties", "sigma_entropy_production_rate"],
                ["Results", "sigma"],
                ["Simulation", "entropy_production_rate"],
                ["sigma"]
            ]
            
            for path in possible_paths
                current = config
                found = true
                for key in path
                    if haskey(current, key)
                        current = current[key]
                    else
                        found = false
                        break
                    end
                end
                
                if found && isa(current, Number)
                    println("DEBUG: Found sigma in Config.json: $current")
                    return Float64(current)
                end
            end
        end
    catch e
        println("DEBUG: Failed to extract sigma from Config.json: $e")
    end
    
    # 如果都找不到，使用一个合理的默认值
    default_val = 0.1
    @warn "Could not find sigma value. Using default value: $default_val. This is only for testing!"
    return default_val
end

# 移除extract_center_of_mass_coordinates函数，替换为处理单个粒子轨迹的函数
function process_particle_trajectories(traj_data, nparticles::Int, lattice_constant::Float64=40.0)
    if isempty(traj_data) || !hasfield(typeof(traj_data[1]), :rl)
        return Dict{Int, Vector{Float64}}()
    end
    
    # 获取有效的时间点数
    valid_timepoints = size(traj_data[1].rl, 2)
    for p in 1:min(length(traj_data), nparticles)
        valid_timepoints = min(valid_timepoints, traj_data[p].ti - 1)
    end
    
    # 为每个粒子创建轨迹数组
    particle_trajectories = Dict{Int, Vector{Float64}}()
    for p in 1:min(length(traj_data), nparticles)
        particle_trajectories[p] = Float64[]
    end
    
    # 解包装粒子坐标（处理周期性边界条件）
    for p in 1:min(length(traj_data), nparticles)
        # 复制第一个点
        push!(particle_trajectories[p], traj_data[p].rl[1, 1])
        
        # 处理剩余点，解决周期性跳变
        for t in 2:valid_timepoints
            prev_x = particle_trajectories[p][end]
            current_x = traj_data[p].rl[1, t]
            dx = current_x - prev_x
            
            # 如果差异超过半个盒子长度，认为是周期性跳变
            if abs(dx) > lattice_constant/2
                # 修正坐标，减去一个盒子长度
                current_x = prev_x + dx - sign(dx) * lattice_constant
            end
            
            push!(particle_trajectories[p], current_x)
        end
    end
    
    return particle_trajectories
end

# 主TUR分析函数
function perform_tur_analysis(basepath_analysis::String, kB_value_sim::Float64, Ts_sim::Float64, 
                             N_tau_points_list_config::Vector{Int}, min_windows_for_stats_config::Int)
    println("Starting TUR analysis for results in: $basepath_analysis")

    # --- 1. 从 Config.json 加载参数 ---
    config_json_filepath = joinpath(basepath_analysis, "Config.json")
    
    # 加载warm_up_steps
    warm_up_steps_sim = try
        read_config_value(config_json_filepath, ["Simulation", "warm_up_steps"])
    catch e1
        # 如果直接读取失败，尝试使用maxstep的一半
        try
            divval = read_config_value(config_json_filepath, ["Simulation", "maxstep"])
            divval ÷ 2
        catch e2
            error("Could not determine warm_up_steps: $e1, and alternative method failed: $e2")
        end
    end
    
    println("Using warm_up_steps = $warm_up_steps_sim for analysis")
    
    # 加载其他必要参数
    dt_sim = read_config_value(config_json_filepath, ["Simulation", "dt_LJ"])
    trajseq_sim = read_config_value(config_json_filepath, ["Results", "traj_record_every_N_steps"])
    nparticle_sim = read_config_value(config_json_filepath, ["Particle", "nparticle"])
    
    println("Loaded parameters: dt=$dt_sim, trajseq=$trajseq_sim, nparticle=$nparticle_sim")

    # --- 2. 加载熵产生率 sigma ---
    sigma_summary_filepath = joinpath(basepath_analysis, "tur_summary_trap_power.txt")
    sigma = read_sigma_from_summary(sigma_summary_filepath) 
    println("Loaded sigma = $sigma")

    # 检查sigma是否为NaN或异常值
    if isnan(sigma) || isinf(sigma)
        println("警告: sigma值异常 ($sigma)，尝试使用默认值0.01")
        sigma = 0.01
    end
    
    # 检查是否使用了修正的熵产率
    corrected_sigma_filepath = joinpath(basepath_analysis, "tur_summary_corrected.txt")
    if isfile(corrected_sigma_filepath)
        println("使用修正后的熵产率 (包含势能变化项 <ΔV>/T)")
    else
        println("使用原始熵产率 (仅基于功率计算)")
    end
    
    println("Loaded sigma = $sigma")
    
    # --- 3. 加载并合并轨迹数据 ---
    traj_dir = joinpath(basepath_analysis, "traj")
    println("Loading trajectory data from: $traj_dir")
    
    if !isdir(traj_dir)
        error("Trajectory directory not found: $traj_dir")
    end
    
    # 查找所有轨迹文件 (与主脚本的命名格式匹配)
    traj_files_all = readdir(traj_dir)
    println("All files in directory: ", join(traj_files_all, ", "))
    
    traj_files = filter(f -> startswith(f, "traj_") && endswith(f, ".jld2"), traj_files_all)
    
    if isempty(traj_files)
        # 尝试查找其他可能的命名模式
        traj_files = filter(f -> (occursin("traj", f) || occursin("trajectory", f)) && 
                            endswith(f, ".jld2"), traj_files_all)
                            
        if isempty(traj_files)
            error("No trajectory files found in $traj_dir")
        else
            println("Found $(length(traj_files)) files with alternative naming pattern")
        end
    else
        println("Found $(length(traj_files)) files matching expected naming pattern")
    end
    
    # 初始化每个粒子的轨迹存储
    particle_stable_data = Dict{Int, Vector{Float64}}()
    for p_idx in 1:nparticle_sim
        particle_stable_data[p_idx] = Float64[]
    end
    
    n_files_processed = 0
    n_points_stable_collected = 0
    time_interval_between_stable_points = trajseq_sim * dt_sim
    
    # 处理所有轨迹文件
    for filename in traj_files
        full_filepath = joinpath(traj_dir, filename)
        println("Processing file: $full_filepath")
        
        if !isfile(full_filepath)
            @warn "File does not exist: $full_filepath"
            continue
        end
        
        try
            jldopen(full_filepath, "r") do file
                # 检查文件内容结构
                println("  File keys: ", keys(file))
                
                # 读取轨迹数据
                if !haskey(file, "TrajList")
                    @warn "File $filename does not contain 'TrajList' key. Available keys: $(keys(file))"
                    return  # Skip this file
                end
                
                TrajList_segment = read(file, "TrajList")
                println("  Found trajectory data for $(length(TrajList_segment)) particles")
                
                # 读取该段的起始步数
                start_step = 0
                if haskey(file, "simulation_start_step_of_segment")
                    start_step = read(file, "simulation_start_step_of_segment")
                elseif haskey(file, "start_step") 
                    start_step = read(file, "start_step")
                else
                    # 尝试从文件名解析
                    parts = split(basename(full_filepath), "_")
                    if length(parts) >= 2
                        try
                            start_step = parse(Int, parts[2])
                        catch
                            @warn "Could not determine start step for $full_filepath, assuming 0"
                        end
                    end
                end
                
                # 处理轨迹数据 - 提取每个粒子的坐标
                particle_trajectories = process_particle_trajectories(TrajList_segment, nparticle_sim)
                
                # 只保留稳态部分的数据点
                points_added = 0
                for t_idx in 1:length(first(values(particle_trajectories)))
                    current_step = start_step + (t_idx - 1) * trajseq_sim
                    if current_step > warm_up_steps_sim
                        for p_idx in 1:nparticle_sim
                            if haskey(particle_trajectories, p_idx)
                                push!(particle_stable_data[p_idx], particle_trajectories[p_idx][t_idx])
                            end
                        end
                        points_added += 1
                    end
                end
                
                println("  Added $points_added stable points per particle from file")
                n_points_stable_collected += points_added
            end
            n_files_processed += 1
        catch e
            @warn "Error processing file $full_filepath: $e"
            showerror(stdout, e)
            println()
            Base.show_backtrace(stdout, catch_backtrace())
            println()
        end
    end
    
    println("Total files processed: $n_files_processed")
    println("Total stable data points collected per particle: $n_points_stable_collected")
    
    if n_points_stable_collected < 2 * min_windows_for_stats_config
        error("Not enough stable data points collected. Need at least $(2 * min_windows_for_stats_config), but got $n_points_stable_collected")
    end

    # --- 4. 计算TUR统计量 - 使用系综平均 ---
    println("Calculating TUR statistics using ensemble averaging...")
    
    # 确定可用的tau值（窗口大小）
    max_tau_points = n_points_stable_collected ÷ min_windows_for_stats_config
    valid_tau_points = filter(n -> n <= max_tau_points && n > 0, N_tau_points_list_config)
    
    if isempty(valid_tau_points)
        error("No valid tau windows available with $n_points_stable_collected points and minimum $min_windows_for_stats_config")
    end
    
    # 对每个tau值计算统计量
    results = []
    for N_tau_points in valid_tau_points
        tau_physical = N_tau_points * time_interval_between_stable_points
        
        all_displacements_ensemble = Float64[]
        
        # 对每个粒子计算位移，并汇总为系综
        for p_idx in 1:nparticle_sim
            if !haskey(particle_stable_data, p_idx) || length(particle_stable_data[p_idx]) < N_tau_points + 1
                continue
            end
            
            x_data = particle_stable_data[p_idx]
            num_windows_particle = length(x_data) - N_tau_points
            
            # 计算每个窗口的位移
            for i in 1:num_windows_particle
                start_idx = i
                end_idx = i + N_tau_points
                displacement = x_data[end_idx] - x_data[start_idx]
                push!(all_displacements_ensemble, displacement)
            end
        end
        
        # 至少需要min_windows_for_stats_config个窗口
        num_windows = length(all_displacements_ensemble)
        if num_windows < min_windows_for_stats_config
            println("  Skipping tau=$tau_physical: not enough ensemble windows ($(num_windows) < $min_windows_for_stats_config)")
            continue
        end
        
        # 计算均值和方差
        mean_J_x_ensemble = mean(all_displacements_ensemble) / tau_physical  # 归一化为电流
        var_J_x_ensemble = var(all_displacements_ensemble, corrected=true) / (tau_physical^2)  # 方差也要归一化
        
        # 计算Q_x（TUR比率）
        Sigma_tau = sigma * tau_physical
        Q_x = var_J_x_ensemble * Sigma_tau / (2 * kB_value_sim * Ts_sim * mean_J_x_ensemble^2)
        
        println("  tau=$tau_physical: <J>=$(round(mean_J_x_ensemble, digits=6)), Var(J)=$(round(var_J_x_ensemble, digits=6)), Q_x=$(round(Q_x, digits=6)), windows=$num_windows")
        push!(results, (tau_physical, mean_J_x_ensemble, var_J_x_ensemble, Sigma_tau, Q_x, num_windows))
        
        if Q_x < 1.0
            println("    WARNING: Q_x < 1.0, potential TUR violation or numerical issue")
        end
    end

    # 添加每个粒子轨迹的线性拟合计算
    println("\nCalculating average drift rates for individual particles:")
    individual_drift_rates = Float64[]
    
    for (p_idx, positions) in particle_stable_data
        if length(positions) >= 10  # 至少需要几个点进行拟合
            times = (1:length(positions)) .* time_interval_between_stable_points
            # 线性拟合: x = a + b*t，其中b就是漂移率
            coeffs = [ones(length(times)) times] \ positions
            drift_rate = coeffs[2]  # 斜率代表平均漂移率
            push!(individual_drift_rates, drift_rate)
            println("  Particle $p_idx drift rate: $drift_rate")
        end
    end
    
    if !isempty(individual_drift_rates)
        ensemble_mean_drift = mean(individual_drift_rates)
        ensemble_std_drift = std(individual_drift_rates)
        println("Ensemble average drift rate: $ensemble_mean_drift ± $ensemble_std_drift")
    end
    
    return results, sigma
end

# 添加一个辅助函数来检测JLD2文件格式并打印其内容
function inspect_jld2_file(filepath)
    println("\nInspecting JLD2 file: $filepath")
    
    if !isfile(filepath)
        println("  File does not exist!")
        return
    end
    
    try
        jldopen(filepath, "r") do file
            println("  File exists and can be opened")
            println("  Root keys: ", keys(file))
            
            # 探索可能的路径
            if haskey(file, "TrajList")
                traj_list = read(file, "TrajList") 
                println("  TrajList contains $(length(traj_list)) items")
                
                if length(traj_list) > 0
                    first_traj = traj_list[1]
                    println("  First trajectory has type: $(typeof(first_traj))")
                    if hasfield(typeof(first_traj), :rl)
                        println("  Trajectory has :rl field with size: $(size(first_traj.rl))")
                    else
                        println("  Trajectory does NOT have :rl field. Available fields: $(fieldnames(typeof(first_traj)))")
                    end
                end
            else
                # 探索可能的替代结构
                for key in keys(file)
                    println("  Exploring key: $key")
                    try
                        value = read(file, key)
                        println("    Type: $(typeof(value))")
                        if isa(value, Array)
                            println("    Array size: $(size(value))")
                        end
                    catch e
                        println("    Error reading key: $e")
                    end
                end
            end
        end
    catch e
        println("  Error opening file: $e")
    end
end

# 修改plot_tur_results函数，删除点上的标识
function plot_tur_results(results, sigma_val, output_path)
    # 确保输出目录存在
    fig_dir = joinpath(output_path, "fig")
    if !isdir(fig_dir)
        mkpath(fig_dir)
        println("Created output directory: $fig_dir")
    end
    
    tau_vals = [r[1] for r in results]
    Qx_vals = [r[5] for r in results]
    windows_vals = [r[6] for r in results]  # 窗口数量
    
    p1 = plot(tau_vals, Qx_vals, 
        seriestype=:scatter, 
        marker=:circle, 
        markersize=5,     # 减小点的大小
        markerstrokewidth=0.5,  # 减小点边框
        xlabel="Time Window τ [LJ units]", 
        ylabel="TUR Ratio Qx", 
        title="TUR Verification (σ=$(round(sigma_val, digits=4)))",
        label="Data points",
        dpi=300,
        legend=:topright  # 将图例放在右上角
    )
    
    # 移除每个点上的标签注释代码，使图表更清晰
    
    # 添加 TUR 界限线 (Q_x = 1)
    hline!(p1, [1.0], linestyle=:dash, color=:red, label="TUR Bound")
    
    # 添加趋势线
    if length(tau_vals) > 3
        sort_idx = sortperm(tau_vals)
        plot!(p1, tau_vals[sort_idx], Qx_vals[sort_idx], 
            linetype=:path, color=:blue, alpha=0.5, label="Trend"
        )
    end
    
    # 添加统计信息到图表中的空白区域，而不是在点上
    min_tau = minimum(tau_vals)
    max_tau = maximum(tau_vals)
    min_Qx = minimum(Qx_vals)
    max_Qx = maximum(Qx_vals)
    
    # 添加一些关键统计信息文本到图表底部
    annotate!(p1, [(min_tau + (max_tau-min_tau)*0.5, min_Qx*1.05, 
               text("Number of data points: $(length(tau_vals))", 8, :center))])
    
    # 保存图表
    qx_filename = joinpath(output_path, "TUR_Qx_vs_tau.png")
    savefig(p1, qx_filename)
    println("Saved TUR plot to: $qx_filename")
    
    # 创建一个总结表格展示数据
    summary_file = joinpath(output_path, "tur_summary_table.txt")
    open(summary_file, "w") do f
        println(f, "# 时间窗口τ统计摘要")
        println(f, "# ---------------")
        println(f, "# τ值范围: $(round(min_tau, digits=1)) - $(round(max_tau, digits=1))")
        println(f, "# Qx值范围: $(round(min_Qx, digits=2)) - $(round(max_Qx, digits=2))")
        println(f, "# 总数据点: $(length(tau_vals))")
        println(f, "# 熵产率σ: $(sigma_val)")
        println(f, "#")
        println(f, "# τ\tQx\t窗口数")
        for i in 1:length(tau_vals)
            println(f, "$(round(tau_vals[i], digits=1))\t$(round(Qx_vals[i], digits=4))\t$(windows_vals[i])")
        end
    end
    println("Saved summary table to: $summary_file")
    
    # 绘制精度-成本图
    rel_uncertainty_vals = [r[3] / (r[2]^2) for r in results]  # Var(J)/<J>²
    thermo_cost_vals = [2.0 / r[4] for r in results]          # 2/Σ_τ
    
    p2 = plot(thermo_cost_vals, rel_uncertainty_vals,
        seriestype=:scatter,
        marker=:circle,
        markersize=5,
        markerstrokewidth=0.5,
        xlabel="Thermodynamic Cost (2/Σ_τ)",
        ylabel="Relative Uncertainty (Var(J)/<J>²)",
        xscale=:log10, yscale=:log10,
        title="Precision-Cost Trade-off",
        label="Data",
        dpi=300
    )
    
    # 添加 TUR 界限线 (y = x)
    min_val = min(minimum(thermo_cost_vals), minimum(rel_uncertainty_vals))
    max_val = max(maximum(thermo_cost_vals), maximum(rel_uncertainty_vals))
    plot!(p2, [min_val, max_val], [min_val, max_val],
        linestyle=:dash, color=:red, label="TUR Bound"
    )
    
    pc_filename = joinpath(output_path, "TUR_Precision_vs_Cost.png")
    savefig(p2, pc_filename)
    println("Saved precision-cost plot to: $pc_filename")
end

# 质心运动计算和绘制函数
function plot_center_of_mass_motion(basepath_analysis::String)
    println("\nAnalyzing Center of Mass Motion...")
    
    # 加载配置
    config_json_filepath = joinpath(basepath_analysis, "Config.json")
    
    # 读取必要参数
    dt_sim = read_config_value(config_json_filepath, ["Simulation", "dt_LJ"])
    trajseq_sim = read_config_value(config_json_filepath, ["Results", "traj_record_every_N_steps"])
    nparticle_sim = read_config_value(config_json_filepath, ["Particle", "nparticle"])
    warm_up_steps_sim = try
        read_config_value(config_json_filepath, ["Simulation", "warm_up_steps"])
    catch
        div(read_config_value(config_json_filepath, ["Simulation", "maxstep"]), 2)
    end
    
    # 加载轨迹数据
    traj_dir = joinpath(basepath_analysis, "traj")
    if !isdir(traj_dir)
        error("轨迹目录不存在: $traj_dir")
    end
    
    traj_files = filter(f -> startswith(f, "traj_") && endswith(f, ".jld2"), readdir(traj_dir))
    if isempty(traj_files)
        traj_files = filter(f -> endswith(f, ".jld2"), readdir(traj_dir))
        if isempty(traj_files)
            error("在$traj_dir中未找到轨迹文件")
        end
    end
    
    # 对文件按时间顺序排序
    function get_start_step_from_filename(filename)
        parts = split(basename(filename), "_")
        if length(parts) >= 2
            try
                return parse(Int, parts[2])
            catch
                return typemax(Int)
            end
        end
        return typemax(Int)
    end
    
    sort!(traj_files, by=get_start_step_from_filename)
    
    # 存储时间和质心位置数据
    times = Float64[]
    com_positions = Float64[]
    
    # 读取所有轨迹文件
    current_time = 0.0
    for filename in traj_files
        full_filepath = joinpath(traj_dir, filename)
        println("处理文件: $filename")
        
        try
            jldopen(full_filepath, "r") do file
                if !haskey(file, "TrajList")
                    @warn "文件$filename不包含'TrajList'键"
                    return
                end
                
                TrajList_segment = read(file, "TrajList")
                
                # 读取起始步数
                start_step = 0
                if haskey(file, "simulation_start_step_of_segment")
                    start_step = read(file, "simulation_start_step_of_segment")
                elseif haskey(file, "start_step")
                    start_step = read(file, "start_step")
                else
                    parts = split(basename(full_filepath), "_")
                    if length(parts) >= 2
                        try
                            start_step = parse(Int, parts[2])
                        catch e
                            @warn "无法确定$full_filepath的起始步数,假设为0. 错误信息: $(e) "   
                        end
                    end
                end
                
                # 确定有效的时间点数量
                valid_points = minimum(traj.ti - 1 for traj in TrajList_segment)
                
                # 只处理稳态阶段的数据
                for t_idx in 1:valid_points
                    current_step = start_step + (t_idx - 1) * trajseq_sim
                    if current_step > warm_up_steps_sim
                        # 计算该时间点的质心位置
                        com_position = 0.0
                        for p_idx in 1:length(TrajList_segment)
                            com_position += TrajList_segment[p_idx].rl[1, t_idx]
                        end
                        com_position /= length(TrajList_segment)
                        
                        # 记录时间和位置
                        time_point = current_time + (t_idx-1) * trajseq_sim * dt_sim
                        push!(times, time_point)
                        push!(com_positions, com_position)
                    end
                end
                
                # 更新当前时间
                if !isempty(times)
                    current_time = times[end] + trajseq_sim * dt_sim
                end
            end
        catch e
            @warn "处理文件$filename时出错: $e"
        end
    end
    
    # 绘制质心运动图
    if length(times) > 1
        # 计算线性拟合
        if length(times) >= 10
            # 使用简单的线性拟合: x = a + b*t
            coeffs = [ones(length(times)) times] \ com_positions
            drift_velocity = coeffs[2]  # 漂移速度
            
            # 计算拟合线
            fit_line = coeffs[1] .+ coeffs[2] .* times
            
            # 绘制图形
            p = plot(times, com_positions, 
                label="Center of Mass", 
                xlabel="Time [LJ units]", 
                ylabel="Position [LJ units]",
                title="Center of Mass Motion\nDrift Velocity = $(round(drift_velocity, digits=6)) [LJ units/time]",
                legend=:topleft,
                linewidth=2,
                dpi=300)
            
            # 添加拟合线
            plot!(p, times, fit_line, 
                label="Linear Fit", 
                linestyle=:dash, 
                linewidth=2, 
                color=:red)
            
            # 保存图形
            com_plot_filepath = joinpath(basepath_analysis, "fig", "center_of_mass_motion.png")
            savefig(p, com_plot_filepath)
            println("质心运动图已保存至: $com_plot_filepath")
            
            # 保存数据
            com_data_filepath = joinpath(basepath_analysis, "center_of_mass_data.txt")
            open(com_data_filepath, "w") do f
                println(f, "# 时间[LJ单位]\t质心位置[LJ单位]")
                for i in 1:length(times)
                    println(f, "$(times[i])\t$(com_positions[i])")
                end
                println(f, "# 漂移速度: $drift_velocity [LJ单位/时间]")
            end
            println("质心数据已保存至: $com_data_filepath")
            
            return drift_velocity
        else
            @warn "数据点不足，无法进行线性拟合"
        end
    else
        @warn "没有足够的质心数据用于绘图"
    end
    
    return nothing
end

# 添加一个函数，比较原始和修正的熵产率
function compare_entropy_production_rates(basepath_analysis::String)
    original_filepath = joinpath(basepath_analysis, "tur_summary_trap_power.txt")
    corrected_filepath = joinpath(basepath_analysis, "tur_summary_corrected.txt")
    
    if isfile(original_filepath) && isfile(corrected_filepath)
        original_sigma = read_sigma_from_summary(original_filepath)
        corrected_sigma = read_sigma_from_summary(corrected_filepath)
        
        println("\n熵产率比较:")
        println("原始熵产率 (仅功率): $original_sigma")
        println("修正熵产率 (功率 - 势能变化): $corrected_sigma")
        println("差异: $(corrected_sigma - original_sigma)")
        println("相对差异: $(100 * (corrected_sigma - original_sigma) / original_sigma)%")
        
        return original_sigma, corrected_sigma
    else
        println("\n无法比较熵产率，缺少必要文件")
        return nothing, nothing
    end
end

# --- 主执行入口 ---
# 可以通过命令行参数提供分析路径，否则使用默认路径
basepath = length(ARGS) >= 1 ? ARGS[1] : "output/brownianparticle_2state_0.5_1"
kB = 1.0  # 与模拟中使用的相同
Ts = 1.0  # 与模拟中使用的相同

# 读取 Config.json 以获取 maxstep 值
config_json_path = joinpath(basepath, "Config.json")
maxstep = 5000000 # 默认值
try
    if isfile(config_json_path)
        config_data = JSON.parsefile(config_json_path)
        if haskey(config_data, "Simulation") && haskey(config_data["Simulation"], "maxstep")
            maxstep = config_data["Simulation"]["maxstep"]
            println("Read maxstep = $maxstep from Config.json")
        end
    end
catch e
    println("Could not read maxstep from Config.json, using default: $maxstep")
    println("Error: $e")
end

# 分析参数 
N_tau_points_list = unique(round.(Int, exp10.(range(0, log10(div(maxstep,2)), length=1000))))  # 对数采样1000个点
filter!(x -> 100 ≤ x ≤ maxstep, N_tau_points_list)  # 限制在有意义的范围内
println("Generated $(length(N_tau_points_list)) tau points for analysis, range: $(minimum(N_tau_points_list)) to $(maximum(N_tau_points_list))")

min_windows_for_stats = 5  # 从 0.5 改为恒定阈值10

# 检查文件系统
println("Checking file system...")
println("Current working directory: $(pwd())")
println("Target path exists: $(isdir(basepath)) ($(basepath))")

if isdir(basepath)
    # 列出目录内容
    println("Directory contents:")
    for (root, dirs, files) in walkdir(basepath)
        println("  Directory: $root")
        for dir in dirs
            println("    Subdir: $dir")
        end
        for file in files
            println("    File: $file ($(filesize(joinpath(root, file))) bytes)")
        end
    end
    
    # 检查轨迹目录
    traj_dir = joinpath(basepath, "traj")
    if isdir(traj_dir)
        println("\nTrajectory directory exists: $traj_dir")
        traj_files = filter(f -> endswith(f, ".jld2"), readdir(traj_dir))
        
        if !isempty(traj_files)
            println("Found $(length(traj_files)) JLD2 files")
            # 检查第一个文件
            inspect_jld2_file(joinpath(traj_dir, traj_files[1]))
        else
            println("No JLD2 files found in trajectory directory")
        end
    else
        println("\nWARNING: Trajectory directory does not exist: $traj_dir")
    end
    
    # 检查sigma文件
    sigma_file = joinpath(basepath, "tur_summary_trap_power.txt")
    if isfile(sigma_file)
        println("\nSigma file exists: $sigma_file")
        println("Content (first 10 lines):")
        open(sigma_file, "r") do f
            for _ in 1:10
                if !eof(f)
                    println("  ", readline(f))
                end
            end
        end
    else
        println("\nWARNING: Sigma file does not exist: $sigma_file")
    end
else
    println("ERROR: Target path does not exist: $basepath")
end

# 继续执行原来的分析
try
    # 执行分析
    results, sigma = perform_tur_analysis(basepath, kB, Ts, N_tau_points_list, min_windows_for_stats)
    
    # 绘制结果
    if !isempty(results)
        plot_tur_results(results, sigma, basepath)
        
        # 保存结果到文件
        results_file = joinpath(basepath, "tur_analysis_results.txt")
        open(results_file, "w") do f
            println(f, "# tau_LJ\tmean_J\tvar_J\tSigma_tau\tQ_x\twindows")
            for result in results
                println(f, join([round(val, digits=6) for val in result], "\t"))
            end
        end
        println("Results saved to: $results_file")
    else
        println("No valid results produced for plotting")
        
        # 即使没有结果，也尝试生成一个最小的 TUR 验证图
        println("Attempting to create a minimal TUR verification plot...")
        
        # 检查更宽松的条件下是否有可用的数据
        relaxed_min_windows = max(5, min_windows_for_stats ÷ 10)  # 放宽窗口数量要求
        min_data_points = 3  # 最少需要几个点才能绘图
        
        # 重新执行分析，使用更宽松的参数
        relaxed_results, _ = perform_tur_analysis(basepath, kB, Ts, N_tau_points_list, relaxed_min_windows)
        
        if length(relaxed_results) >= min_data_points
            println("Generated $(length(relaxed_results)) data points with relaxed criteria")
            plot_tur_results(relaxed_results, sigma, basepath)
        else
            # 如果仍然无法生成足够数据，创建一个空的 TUR 验证图
            p_empty = plot(
                title="TUR Verification (σ=$(round(sigma, digits=4)))",
                xlabel="Time Window τ [LJ units]",
                ylabel="TUR Ratio Qx",
                dpi=300,
                legend=:topright
            )
            
            # 在图上添加提示信息
            annotate!(p_empty, [(0.5, 0.5, text("Insufficient data for TUR analysis", 12, :center))])
            
            # 仍然添加 TUR 界限线
            hline!(p_empty, [1.0], linestyle=:dash, color=:red, label="TUR Bound")
            
            qx_filename = joinpath(basepath, "TUR_Qx_vs_tau.png")
            savefig(p_empty, qx_filename)
            println("Created empty TUR plot at: $qx_filename")
        end
    end
catch e
    println("Error during analysis:")
    showerror(stdout, e)
    println("\nStacktrace:")
    Base.show_backtrace(stdout, catch_backtrace())
end

# 在主执行部分调用此函数
try
    drift_velocity = plot_center_of_mass_motion(basepath)
    if drift_velocity !== nothing
        println("系统质心的漂移速度: $drift_velocity [LJ单位/时间]")
    end
catch e
    println("计算质心运动时出错:")
    showerror(stdout, e)
end

# 在主执行部分调用比较函数
try
    compare_entropy_production_rates(basepath)
catch e
    println("比较熵产率时出错:")
    showerror(stdout, e)
end
