using StaticArrays
# using Plots
using LinearAlgebra
using LsqFit
include("src/Elastic.jl")
using .Elastic
using FFMPEG
using DelimitedFiles
using Distributions
using JLD2
using Base.Threads
using Plots
using Random
using JSON
# rdseed=255515
ENV["GKSwstype"] = "100"
rdseed=rand(10^(4-1):10^(8)-1)
Random.seed!(rdseed)
println("Number of threads: ", Threads.nthreads())

# 获取系统路径
output_root = "output" # 与 brownianparticle_gauss_3d_bias.jl 保持一致
basepath = joinpath(output_root, "brownianparticle_tiltedperiodic_bias_5")
println("Basepath: $basepath")

unit="lj"
set_unit(unit)
tt=0.5052622965408168 #ms/[t]
projectname="brownianparticle_tiltedperiodic_bias"

basepath=joinpath("output",projectname)
if !isdir(basepath)
    mkpath(basepath)
    println("Directory $basepath created.\n")
else
    local counter = 1
    local newpath = basepath * "_$counter"
    while isdir(newpath)
        counter += 1
        newpath = basepath * "_$counter"
    end
    mkpath(newpath)
    println("Directory exists,new Directories $newpath created.\n")
    basepath=newpath
end

Comments=""
#cell
lattice_constant=40.0

#particle
nparticle=32
xmax=lattice_constant
mass=4.2 #pg

#trap
ksin=5.0
N=6 #周期数
phi0=0
bias=0#单位g=pN/pg=N/kg
w=pi/lattice_constant*N

#ensemble
Ts = 1.0
t0 = 0.000222 / tt  # ms

#simulation
fixdim = [2, 3]
maxstep =5000000
warm_up_steps = div(maxstep, 2)  # 热化步数50%
savetrajseq =maxstep  # 每 5% 保存一次
dt = 0.01
kB_value = 1.0  # 玻尔兹曼常数 (模拟单位)
trajseq = 100
ifsavetraj = true
PBC = false
ifsavefinalstate = true
pltlist = 1:nparticle
logsequence = 1000  # 日志记录间隔

lattice_vectors = collect((Matrix([
    lattice_constant 0.0 0.0; #a1
    0.0 lattice_constant 0.0; #a2
    0.0 0.0 lattice_constant] #a3
))')

function biasE(r::SVector{3,Float64})
    x=r[1]
    return -x*bias+ksin*sin(phi0+w*x)
end

function biasF(r::SVector{3,Float64})
    x=r[1]
    return SVector{3,Float64}([bias-ksin*w*cos(phi0+w*x),0.0,0.0])
end

invlt=inv(lattice_vectors)

atom_positions = [[0.0,0.0,0.0] for xi in (range(-xmax, xmax, length=nparticle+1)[1:end-1])] ./ lattice_constant
atoms = [Atom(pos,mass) for pos in atom_positions];
println(atom_positions)
inicell=UnitCell(lattice_vectors, atoms)

trap=Field(biasE,biasF)
inter=Vector{AbstractInteraction}([trap])
nb=Vector{Neighbor}([Neighbor()])
interactions=Interactions(inter,nb)

@show interactions

function ensure_writable_path(path::String)
    try
        mkpath(path)
        println("Directory $path is writable and created.")
    catch e
        error("Cannot create directory $path. Please check permissions or choose a different path.")
    end
end

ensure_writable_path(basepath)

# 假设所有变量都已经定义并赋值
config_dict = Dict(
    "projectname" => projectname,
    "IntergrateMethod" => "Langevin",
    "Interaction" => "Trap(gauss)/Yukawa/WCA",
    "RandomSeed" => rdseed,
    "Unit"=>"lj",
    "Cell" => Dict(
        "lattice_vectors" => lattice_vectors
    ),
    "Particle" => Dict(
        "nparticle" => nparticle,
        "mass" => mass,
        "generate_xmax" => xmax,
        "avgdistance" => nparticle > 1 ? getrij(inicell, 1, 2) : "None"
    ),
    "Trap" => Dict(
        "ktrap" => ksin,
        "N" => N,
        "phi0" => phi0,
        "bias" => bias
    ),
    "Ensemble" => Dict(
        "Ts" => Ts,
        "t0" => t0
    ),
    "Simulation" => Dict(
        "maxstep" => maxstep,
        "dt" => dt,
        "fixdim" => fixdim,
        "simulation_time" => dt * maxstep * tt * 10^(-6)
    ),
    "Results" => Dict(
        "trajseq" => trajseq,
        "savetrajseq" => savetrajseq,
        "PBC" => PBC,
        "pltlist" => pltlist
    ),
    "Comments" => Comments
)

open(joinpath(basepath,"Config.json"), "w") do file
    write(file, JSON.json(config_dict))
end

write_config_file(basepath,config_dict)
mkpath(joinpath(basepath,"traj"))
mkpath(joinpath(basepath,"fig"))

x=-lattice_constant:0.01:lattice_constant
y1=[biasE(SVector{3,Float64}([xi,0.0,0.0])) for xi in x]
y2=[biasF(SVector{3,Float64}([xi,0.0,0.0]))[1] for xi in x]
p=plot(
    plot(x, y1, title="Energy"),
    plot(x, y2, title="Force"),
    layout = (1, 2),
    size = (800, 400)
)
savefig(joinpath(basepath,"fig","trap.png"))

cell=deepcopy(inicell)
fix_dim!(cell,fixdim)
update_rmat!(cell)
update_fmat!(cell,interactions)
trajsavecount=0
TrajList=[Trajectory(0,min(savetrajseq,maxstep),trajseq,dt) for _ in 1:nparticle]

# 在定义 mass 和 t0 的地方计算 gamma
if t0 <= 0.0
    error("Relaxation time t0 must be positive!")
end
const gamma = mass / t0  # 计算阻尼系数 gamma = m / t0
println("Calculated friction coefficient gamma (m/t0): ", gamma)

open(joinpath(basepath, "log.txt"), "w") do logfile
    # Initialization for sigma calculation
    total_power_diss_stable = 0.0  # 累积所有粒子的总耗散功率
    steady_state_steps_count = 0   # 稳态步数计数

    last_saved_step = 0  # 在主循环前初始化

    for step in 1:maxstep
        # --- 执行 Langevin 动力学步骤 ---
        LangevinDump_step!(dt, cell, interactions, Ts, t0, fixdim=fixdim)

        # --- 更新力矩阵 (如果 LangevinDump_step! 没有做) ---
        # update_fmat!(cell, interactions) # <-- 如果需要，取消这行的注释

        # +++ 修改后的耗散功率计算 START +++
        # 1. 获取步结束时的物理时间
        step_end_time = step * dt

        # 2. 初始化 *当前步* 所有粒子耗散功率的总和
        power_sum_this_step = 0.0

        # 3. 遍历每个粒子计算其耗散功率并累加
        for i in 1:nparticle
            # --- 获取第 i 个粒子的速度 ---
            momentum_i = SVector{3,Float64}(cell.atoms[i].momentum)
            mass_i = cell.atoms[i].mass
            if mass_i <= 0
                continue # 跳过质量非正的粒子
            end
            velocity_i = momentum_i / mass_i
            vx_i = velocity_i[1]

            # --- 获取作用在粒子 i 上的 *总* 确定性力 ---
            F_total_i = cell.fmat[i] # 这是 SVector{3,Float64}

            # --- 从总力中分离出 *非保守力* (即恒定力 bias) ---
            # 在这个模型中，非保守力就是 bias
            force_non_conservative_x = F_total_i = cell.fmat[i] 

            # --- 计算第 i 个粒子的瞬时耗散功率 ---
            # power_diss_i = bias * vx_i # P_i = F_nc_x * vx_i
            power_diss_i = force_non_conservative_x[1] * vx_i # P_i = F_nc_x * vx_i

            # --- 累加到当前步的总和 ---
            power_sum_this_step += power_diss_i
        end
        # +++ 修改后的耗散功率计算 END +++

        # --- 累加稳态下的总功率 ---
        if step > warm_up_steps
            total_power_diss_stable += power_sum_this_step
            steady_state_steps_count += 1
        end

        # --- 日志记录 ---
        if mod(step, logsequence) == 0
            T = cell_temp(cell)
            Ep = cell_energy(cell, interactions)
            Ek = cell_Ek(cell)
            println("step=$step, T=$T, Ep=$Ep, Ek=$Ek, E=$(Ep + Ek), Pdiss_sum_inst=$power_sum_this_step")
            writedlm(logfile, [step T Ep Ek Ep + Ek power_sum_this_step])
            flush(logfile)
        end

        # 轨迹记录
        if mod(step, trajseq) == 0
            fix_traj!(TrajList, cell, PBC=PBC)
        end

        # 保存轨迹的判断
        is_save_interval = mod(step, savetrajseq) == 0
        is_last_step = step == maxstep

        if (is_save_interval || is_last_step) && ifsavetraj
    



            # +++ 精确计算保存区间 (使用整数) +++
            current_save_start = last_saved_step
            current_save_end = step

            #savefig
            p = Plots.plot(dpi=800, xlabel="step", ylabel="y/um")  # 设置x轴和y轴标签
            for i in pltlist
            Plots.plot!(p,[TrajList[i].rl[1,1:end-1]],label="P$i")
            end 
            global trajsavecount
                savefig(joinpath(basepath,"fig","trajfig_$(current_save_start)_$(current_save_end).png"))

            # --- 构造文件名 (使用整数) ---
            filepath = joinpath(basepath, "traj", "traj_$(current_save_start)_$(current_save_end).jld2")
            println("Saving trajectory segment $(current_save_start + 1) to $current_save_end to $filepath...")

            try
                # --- 写入 JLD2 文件 ---
                jldopen(filepath, "w") do file
                    write(file, "TrajList", TrajList)
                    write(file, "start_step", current_save_start + 1) # 从 1 开始计数
                    write(file, "end_step", current_save_end)
                    write(file, "dt", dt)
                    write(file, "trajseq", trajseq)
                end
                println("  Successfully saved trajectory data and metadata.")

                # --- 添加验证读取 ---
                try
                    jldopen(filepath, "r") do f
                        println("  Verification read: Found keys = ", keys(f))
                        required_keys = ["TrajList", "start_step", "end_step", "dt", "trajseq"]
                        if all(haskey(f, k) for k in required_keys)
                            println("  Verification read: start_step = ", read(f, "start_step"))
                            println("  Verification read: trajseq = ", read(f, "trajseq"))
                            println("  Verification read: OK.")
                        else
                            missing_keys = filter(k -> !haskey(f, k), required_keys)
                            @error "  Verification read FAILED: Missing keys in saved file! Missing: $missing_keys"
                        end
                    end
                catch read_err
                    @error "  Error during verification read: $read_err"
                end

                # --- 更新最后保存的步数 ---
                last_saved_step = current_save_end

                # --- 清空 Trajectory 内存 ---
                if !is_last_step
                    println("  Clearing Trajectory data in memory...")
                    try
                        clear_traj!(TrajList)
                    catch e
                        @warn "Failed to clear TrajList. Error: $e"
                    end
                end



            catch e
                println("ERROR saving trajectory file $filepath: $e")
                # error("Failed to save JLD2 file, stopping.") # 如需强制终止可取消注释
            end
        end
    end

    # --- Simulation loop结束 ---
    println("Simulation loop finished.")

    # --- 计算并保存 sigma ---
    sigma_entropy_rate = NaN
    if steady_state_steps_count > 0
        # 1. 计算平均总耗散功率 (所有粒子的总和的平均值)
        avg_total_power_diss = total_power_diss_stable / steady_state_steps_count

        # 2. 计算平均单粒子耗散功率
        avg_single_particle_power_diss = avg_total_power_diss / nparticle  # 除以粒子数

        # 3. 计算反映单粒子行为的 sigma
        sigma_entropy_rate = avg_single_particle_power_diss / (Ts * kB_value)

        println("--- Steady State Summary ---")
        println("Total accumulated power sum in steady state: ", total_power_diss_stable)
        println("Number of steady state steps counted: ", steady_state_steps_count)
        println("Average TOTAL power dissipation (all particles): ", avg_total_power_diss)
        println("Average SINGLE particle power dissipation: ", avg_single_particle_power_diss)
        println("Entropy Production Rate per particle (sigma): ", sigma_entropy_rate)

        # 保存 sigma 到文件
        sigma_filepath = joinpath(basepath, "sigma.txt")
        open(sigma_filepath, "w") do file
            write(file, string(sigma_entropy_rate))
        end
        println("Sigma saved to $sigma_filepath")
    else
        println("Warning: No steady-state steps recorded. Sigma calculation skipped.")
    end
end

