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
ENV["GKSwstype"] = "100"
# rdseed=255515
rdseed=rand(10^(4-1):10^(8)-1)
Random.seed!(rdseed)
println("Number of threads: ", Threads.nthreads())
# 获取系统路径
data_root = ""
# try
#     global data_root
#     home_parent = dirname(dirname(ENV["HOME"]))
#     data_root = joinpath(home_parent, "data/xieyuncheng")
# catch
#     global data_root
#     data_root = ""
# end
println("$data_root") 
tt=0.5052622965408168 #ms/[t]
g=10.0 #um/ms^2
projectname="brownianparticle_3d_bias_deltax_100.0"
Comments = ""  # 定义一个空字符串作为默认值

#cell
lattice_constant=40.0

#particle
nparticle=32
xmax=lattice_constant
mass=4.2 #pg

#trap
ktrap=1000.0
trapsigma=3.0
trapcut=10*trapsigma
ntrap=64
deltax=2*lattice_constant/ntrap
deltat=1.0/tt #ms

#interaction
#yukawa
yukawak=2.0
yukawaF0=1.0
yukawacut=2*yukawak
#wca
epsilon=1.0
sigma=2.0

#bias
biasF0=0.01*g

#ensemble
Ts=1.0
t0=0.000222/tt #ms 

#simulation
fixdim=Vector{Int}([])
maxstep=1000000
dt=0.01 #[t]
warm_up_steps = maxstep ÷ 5 # 定义预热步数 (例如10%)，之后可以调整
kB_value = 1.0              # 定义玻尔兹曼常数(模拟单位)
println("Warm-up steps for sigma calculation: $warm_up_steps") # 打印信息

#results
trajseq=1
savetrajseq= 1000000
ifsavetraj=true
PBC=false
ifsavefinalstate=true
pltlist=1:nparticle
logsequence=1000

#property
phi=sigma*nparticle/2/lattice_constant

lattice_vectors = collect((Matrix([
    lattice_constant 0.0 0.0; #a1
    0.0 lattice_constant 0.0; #a2
    0.0 0.0 lattice_constant] #a3
))')

function lj(r::Float64)
    return 4*epsilon*((sigma/r)^12-(sigma/r)^6+1/4)
end
function Flj(r::SVector{3,Float64})
    rn=norm(r)
    return 24*epsilon*(2*(sigma/rn)^14-(sigma/rn)^8)*r/sigma^2
end
invlt=inv(lattice_vectors)


function trapenergy(r::SVector{3,Float64},t::Float64)
    x0=SVector{3,Float64}([deltax*mod(div(t,deltat),ntrap)-lattice_constant,0.0,0.0])
    r0=r-x0
    invr0=invlt*r0
    cp=cell.copy
    for k=1:3
        if invr0[k]>cp[k]
            invr0[k]-=2*cp[k]
        elseif invr0[k]<-cp[k]
            invr0[k]+=2*cp[k]
        end
    end
    r0=lattice_vectors*invr0
    nr0=norm(r0)
    if nr0 < trapcut
        return -ktrap*(exp(-0.5 * nr0^2 / trapsigma^2)-exp(-0.5 * trapcut^2 / trapsigma^2))
    else
        return 0.0
    end
end

function trapforce(r::SVector{3,Float64},t::Float64)
    x0=SVector{3,Float64}([deltax*mod(div(t,deltat),ntrap)-lattice_constant,0.0,0.0])
    r0=r-x0
    cp=cell.copy
    invr0=invlt*r0
    for k=1:3
        if invr0[k]>cp[k]
            invr0[k]-=2*cp[k]
        elseif invr0[k]<-cp[k]
            invr0[k]+=2*cp[k]
        end
    end
    r0=lattice_vectors*invr0
    nr0=norm(r0)

    if  nr0< trapcut
        return r0/trapsigma^2*(-ktrap*exp(-0.5 * nr0^2 / trapsigma^2))
    else
        return zeros(3)
    end
end

function YukawaE(r::Float64)
    return yukawaF0*exp(-yukawak*norm(r))/norm(r)
end

function YukawaF(r::SVector{3,Float64})
    nr=norm(r)
    return (yukawaF0*exp(-yukawak*nr)*(1.0+yukawak*nr)/nr^3 ).* r
end



function biasE(r::SVector{3,Float64})
    return -r[3]*biasF0
end

function biasF(r::SVector{3,Float64})
    return SVector{3,Float64}([0.0,0.0,biasF0])
end

atom_positions = [[xi,0.0,0.0]+[lattice_constant/nparticle,0.0,0.0] for xi in (range(-xmax, xmax, length=nparticle+1)[1:end-1])] ./ lattice_constant
atoms = [Atom(pos,mass) for pos in atom_positions];
println(atom_positions)
inicell=UnitCell(lattice_vectors, atoms)

trap=MutableField(trapenergy,trapforce,0.0)
yukawa=Interaction(YukawaE,YukawaF,yukawacut,0.1)
interlj=Interaction(lj,Flj,sigma*2^(1/6),0.1)
bias=Field(biasE,biasF)
inter=Vector{AbstractInteraction}([trap,yukawa,interlj,bias])
nb=Vector{Neighbor}([Neighbor(),Neighbor(inicell),Neighbor(inicell),Neighbor()])
interactions=Interactions(inter,nb)

basepath=joinpath(data_root,"output",projectname)
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

##logfile
open(joinpath(basepath,"Config.txt"), "w") do logfile
    write(logfile, "projectname=$projectname\n")
    write(logfile,"IntergrateMethod:Langevin,Interaction:Trap(gauss)/Yukawa/WCA\n")
    write(logfile,"RandomSeed=$rdseed\n")
    write(logfile, "\nCell:\n")
    write(logfile, "lattice_vectors=$lattice_vectors\n")
    write(logfile, "\nParticle:\n")
    write(logfile, "nparticle=$nparticle\n")
    write(logfile, "mass=$mass\n")
    write(logfile, "generate_xmax=$xmax\n")
    if nparticle>1
        write(logfile, "avgdistance=$(getrij(inicell,1,2))\n")
    else
        write(logfile, "avgdistance=None\n")
    end
    write(logfile, "\nTrap:\n")
    write(logfile, "ktrap=$ktrap\n")
    write(logfile, "trapsigma=$trapsigma\n")
    write(logfile, "trapcut=$trapcut\n")
    write(logfile, "ntrap=$ntrap\n")
    write(logfile, "deltax=$deltax\n")
    write(logfile, "deltat=$deltat\n")
    write(logfile, "\nInteraction:\n")
    write(logfile, "Yukawak=$yukawak\n")
    write(logfile, "yukawaF0=$yukawaF0\n")
    write(logfile, "yukawacut=$yukawacut\n")
    write(logfile, "\nWCA:\n")
    write(logfile, "epsilon=$epsilon\n")
    write(logfile, "sigma=$sigma\n")
    write(logfile, "biasF0=$biasF0\n")
    write(logfile, "\nEnsemble:\n")
    write(logfile, "Ts=$Ts\n")
    write(logfile, "t0=$t0\n")
    write(logfile, "\nSimulation:\n")
    write(logfile, "maxstep=$maxstep\n")
    write(logfile, "dt=$dt\n")
    write(logfile, "fixdim=$fixdim\n")
    write(logfile, "simulation time:$(dt*maxstep*tt*10^(-6))s\n")
    write(logfile, "\nResults:\n")
    write(logfile, "trajseq=$trajseq\n")
    write(logfile, "savetrajseq=$savetrajseq\n")
    write(logfile, "PBC=$PBC\n")
    write(logfile, "pltlist=$pltlist\n")
    write(logfile, "\nProperty:\n")
    write(logfile, "phi=$phi\n")
    write(logfile, "\nComments\n$Comments\n")
end

# 假设所有变量都已经定义并赋值
config_dict = Dict(
    "projectname" => projectname,
    "IntergrateMethod" => "Langevin",
    "Interaction" => "Trap(gauss)/Yukawa/WCA",
    "RandomSeed" => rdseed,
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
        "ktrap" => ktrap,
        "trapsigma" => trapsigma,
        "trapcut" => trapcut,
        "ntrap" => ntrap,
        "deltax" => deltax,
        "deltat" => deltat
    ),
    "Interaction" => Dict(
        "Yukawak" => yukawak,
        "yukawaF0" => yukawaF0,
        "yukawacut" => yukawacut
    ),
    "WCA" => Dict(
        "epsilon" => epsilon,
        "sigma" => sigma
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
    "Property" => Dict(
        "phi" => phi
    ),
    "Comments" => Comments
)

open(joinpath(basepath,"Config.json"), "w") do file
    write(file, JSON.json(config_dict))
end

mkpath(joinpath(basepath,"traj"))
mkpath(joinpath(basepath,"fig"))
mkpath(joinpath(basepath,"fig","x"))
mkpath(joinpath(basepath,"fig","y"))
mkpath(joinpath(basepath,"fig","z"))
# mkpath("$basepath/figm")

cell=deepcopy(inicell)
fix_dim!(cell,fixdim)
update_rmat!(cell)
update_fmat!(cell,interactions)
trajsavecount=0
# 初始化 TrajList
sampling_dt = dt * trajseq # <--- 计算采样间隔
initial_step_or_time = 0   # 假设第一个参数是初始时间或步数
# 分配足够存储一个片段的大小 (+1 用于边界)
samples_per_segment = ceil(Int, savetrajseq / trajseq) + 1
TrajList = [Trajectory(initial_step_or_time, samples_per_segment, trajseq, sampling_dt) for _ in 1:nparticle]

# Initialization for sigma calculation
total_power_diss_stable = Ref(0.0)  # 使用 Ref 初始化为 0.0
steady_state_steps_count = Ref(0)   # 使用 Ref 初始化为 0

# --- 获取非保守力函数的引用 (推荐做法) ---

trap_force_function = trap.force      # 使用正确的字段名 .force
bias_force_function = bias.force      # 使用正确的字段名 .force
# ---------------------------------------

open("$basepath/log.txt", "w") do logfile
    for step in 1:maxstep
        interactions.interactions[1].t += dt # 更新时间

        # --- 执行 Langevin 动力学步骤 ---
        LangevinDump_step!(dt, cell, interactions, Ts, t0, fixdim=fixdim)
        # ---------------------------------

        # +++ 添加计算耗散功率的逻辑 START +++
        # 1. 获取步结束时的物理时间
        step_end_time = step * dt # 时间点对应于 v(t+dt) 和 r(t+dt)

        # 2. 计算作用在新位置 r(t+dt) 上的非保守力
        ltv_mat = SMatrix{3,3}(cell.lattice_vectors) # 静态矩阵提高性能
        positions_cartesian = [ltv_mat * SVector{3,Float64}(cell.atoms[i].position) for i in 1:nparticle]

        current_forces_trap = [trap_force_function(positions_cartesian[i], step_end_time) for i in 1:nparticle]
        current_forces_bias = [bias_force_function(positions_cartesian[i]) for i in 1:nparticle]

        # 3. 计算瞬时耗散功率
        current_power_diss_step = 0.0
        for i in 1:nparticle
            # --- 在这里计算第 i 个粒子的速度 ---
            momentum_i = SVector{3,Float64}(cell.atoms[i].momentum) # 获取更新后的动量
            mass_i = cell.atoms[i].mass
            if mass_i <= 0
                println("Warning: Particle $i has non-positive mass ($mass_i). Skipping.")
                continue
            end
            velocity_i = momentum_i / mass_i # 计算更新后的速度
            # ------------------------------------

            # 点积计算每个粒子上的耗散功率贡献
            power_trap = dot(current_forces_trap[i], velocity_i)
            power_bias = dot(current_forces_bias[i], velocity_i)
            current_power_diss_step += (power_trap + power_bias)
        end
        # +++ 添加计算耗散功率的逻辑 END +++

        # --- Accumulate power dissipation after warm-up ---
        if step > warm_up_steps
            total_power_diss_stable[] += current_power_diss_step  # 使用 [] 访问 Ref 内容
            steady_state_steps_count[] += 1                       # 使用 [] 访问 Ref 内容
        end
        # -------------------------------------------------

        # --- Record trajectory data ---
        if mod(step, trajseq) == 0 # Record every trajseq steps
            fix_traj!(TrajList, cell, PBC=PBC)
        end
        # ---------------------------

        # --- 日志记录 ---
        if mod(step, logsequence) == 0
            T = cell_temp(cell)
            Ep = cell_energy(cell, interactions) # Assuming no time arg needed
            Ek = cell_Ek(cell)
            println("step=$step, T=$T, Ep=$Ep, Ek=$Ek, E=$(Ep + Ek), Pdiss_inst=$current_power_diss_step")
            # Add instantaneous power dissipation to log
            writedlm(logfile, [step T Ep Ek Ep + Ek current_power_diss_step])
            flush(logfile)
        end

        # --- 保存 Trajectory 数据并重建 TrajList ---
        if (mod(step, savetrajseq) == 0) || (step == maxstep)
            current_save_start_step = trajsavecount * savetrajseq # Approximation
            current_save_end_step = step

            println("Saving trajectory segment ending at step $current_save_end_step...")
            global trajsavecount # Declare global to modify

            # --- Save trajectory to JLD2 ---
            if ifsavetraj
                filepath = joinpath(basepath, "traj", "traj_$(current_save_start_step)_$(current_save_end_step).jld2")
                try
                    jldopen(filepath, "w") do file
                        write(file, "TrajList", TrajList)
                        # Add metadata
                        write(file, "start_step", current_save_start_step + 1)
                        write(file, "end_step", current_save_end_step)
                        write(file, "trajseq", trajseq)
                        write(file, "dt", dt)
                    end
                    println("Saved trajectory to $filepath")
                catch e
                    println("Error saving trajectory file $filepath: $e")
                end
            end
            # -----------------------------

            # --- Save figures ---
            should_save_figs = (step == maxstep) || mod(trajsavecount, 5) == 0 # Example: save every 5 saves or at end
            if should_save_figs && ifsavetraj
                println("Saving figures...")
               
                println("Figures saved.")
            end
            # ----------------------------------------------

            trajsavecount += 1 # Increment counter *after* using it for filename

            # --- Clear Trajectory data using clear_traj! ---
            if step < maxstep
                println("Clearing Trajectory data in memory...")
                try
                    clear_traj!(TrajList) # <--- 使用找到的清空函数
                catch e
                    println("Warning: Failed to clear TrajList. Error: $e")
                end
            end
            # ---------------------------------------------

        end # 结束 if (mod(step, savetrajseq) == 0) || (step == maxstep)
    end
end

# --- Simulation loop结束 ---
println("Simulation loop finished.")

# --- 计算并保存 sigma ---
sigma_entropy_rate = NaN
if steady_state_steps_count[] > 0  # 使用 [] 访问 Ref 内容
    avg_power_diss = total_power_diss_stable[] / steady_state_steps_count[]  # 使用 [] 访问 Ref 内容
    sigma_entropy_rate = avg_power_diss / (Ts * kB_value)
    println("--- Steady State Summary ---")
    println("Total accumulated power in steady state: ", total_power_diss_stable[])  # 使用 [] 访问 Ref 内容
    println("Number of steady state steps counted: ", steady_state_steps_count[])  # 使用 [] 访问 Ref 内容
    println("Average Power Dissipation: ", avg_power_diss)
    println("Entropy Production Rate (sigma): ", sigma_entropy_rate)

    sigma_filepath = joinpath(basepath, "sigma.txt")
    try
        open(sigma_filepath, "w") do sigfile
            write(sigfile, "$sigma_entropy_rate")
        end
        println("Saved sigma to $sigma_filepath")
    catch e
        println("Error saving sigma.txt: $e")
    end
else
    println("Warning: No steady-state steps recorded. Sigma calculation skipped.")
end

println("Script finished.")

