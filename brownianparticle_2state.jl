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
basepath = joinpath(output_root, "brownianparticle_2state_test")
println("Basepath: $basepath")

unit="lj"
set_unit(unit)
tt=0.5052622965408168 #ms/[t]
projectname="brownianparticle_2state_test"



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
ktrap=1.0
e=0.5 #势场的偏心率，>0 #E=A sin(w x)/(1+e+cos(w x))
deltat_on=100.0/tt #ms
deltat_off=100.0/tt #ms
N=2 #势场周期数
phi0=0

#interaction
ifinteraction=false #是否使用相互作用,false 为自由粒子
#yukawa
yukawak=2.0
yukawaF0=1.0
yukawacut=2*yukawak
#wca
epsilon=1.0
sigma=2.0

#ensemble
Ts=1.0
t0=0.000222/tt #ms 

#simulation
fixdim=[2,3]
maxstep=1000000
dt=0.1 #[t]
println("simulation time:$(dt*maxstep*tt*10^(-3))s")

#results
trajseq=10
savetrajseq= maxstep
ifsavetraj=false
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

function A(e::Float64)
    return sqrt(e)*sqrt(2+e)
end

function trapenergy(r0::SVector{3,Float64},t::Float64)  
    nr0=r0[1]
    deltat=deltat_on+deltat_off
    tn=mod(t,deltat)
    w=pi/lattice_constant*N
    if tn<deltat_on
        return ktrap*A(e)*sin(w*nr0+phi0)/(1+e+cos(w*nr0+phi0))
    else
        return 0.0
    end
    
end

function trapforce(r0::SVector{3,Float64},t::Float64)
    nr0=r0[1]
    deltat=deltat_on+deltat_off
    tn=mod(t,deltat)
    w=pi/lattice_constant*N
    if tn<deltat_on
        return ktrap*A(e)*w*(cos(nr0*w+phi0)+sin(w*nr0+phi0)^2/(1+e+cos(w*nr0+phi0)))/(1+e+cos(w*nr0+phi0))*(-r0/nr0)
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


atom_positions = [[xi,0.0,0.0]+[lattice_constant/nparticle,0.0,0.0] for xi in (range(-xmax, xmax, length=nparticle+1)[1:end-1])] ./ lattice_constant
atoms = [Atom(pos,mass) for pos in atom_positions];
println(atom_positions)
inicell=UnitCell(lattice_vectors, atoms)
if ifinteraction
    trap=MutableField(trapenergy,trapforce,0.0)
    yukawa=Interaction(YukawaE,YukawaF,yukawacut,0.1)
    interlj=Interaction(lj,Flj,sigma*2^(1/6),0.1)
    inter=Vector{AbstractInteraction}([trap,yukawa,interlj])
    nb=Vector{Neighbor}([Neighbor(),Neighbor(inicell),Neighbor(inicell)])
    interactions=Interactions(inter,nb)   
else
    trap=MutableField(trapenergy,trapforce,0.0)
    inter=Vector{AbstractInteraction}([trap])
    nb=Vector{Neighbor}([Neighbor()])
    interactions=Interactions(inter,nb)
end
@show interactions

# 移除原系统路径获取逻辑，改为固定相对路径
output_root = "output"  # 与 222brownianparticle_tiltedperiodic.jl 保持一致
projectname = "brownianparticle_2state"
Comments = ""

# 基于 output_root 构建 basepath
basepath = joinpath(output_root, projectname)
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
    println("Directory exists, new Directories $newpath created.\n")
    basepath = newpath
end



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
        "ktrap" => ktrap,
        "N" => N,
        "deltat_on" => deltat_on,
        "deltat_off" => deltat_off,
        "phi0" => phi0
    ),
    "Interaction" => Dict(
        "ifinteraction" => ifinteraction,
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

write_config_file(basepath,config_dict)
mkpath(joinpath(basepath,"traj"))
mkpath(joinpath(basepath,"fig"))
# mkpath("$basepath/figm")
x=-lattice_constant:0.01:lattice_constant
y1=[trapenergy(SVector{3,Float64}([xi,0.0,0.0]),0.0) for xi in x]
y2=[trapforce(SVector{3,Float64}([xi,0.0,0.0]),0.0)[1] for xi in x]
p=plot(
    plot(x, y1, title="Energy"),  
    plot(x, y2, title="Force"),  
    layout = (1, 2),            
    size = (800, 400)            
)
savefig(joinpath(basepath,"fig","trap.png"))


x=0:1.0:5000.0
y1=[trapenergy(SVector{3,Float64}([1.0,0.0,0.0]),xi) for xi in x]
y2=[trapforce(SVector{3,Float64}([1.0,0.0,0.0]),xi)[1] for xi in x]
p=plot(
    plot(x, y1, title="Energy"),  
    plot(x, y2, title="Force"),  
    layout = (1, 2),            
    size = (800, 400)            
)
savefig(joinpath(basepath,"fig","trap_t.png"))

cell=deepcopy(inicell)
fix_dim!(cell,fixdim)
update_rmat!(cell)
update_fmat!(cell,interactions)
trajsavecount=0
TrajList=[Trajectory(0,min(savetrajseq,maxstep),trajseq,dt) for _ in 1:nparticle]
open("$basepath/log.txt", "w") do logfile
# 定义预热步骤（前50%步骤作为预热期）
warm_up_steps = maxstep ÷ 2  

# 初始化耗散功率计算相关变量（移除const声明）
gamma = mass / t0  # 计算阻尼系数 gamma = m / t0（移除const）
println("Calculated friction coefficient gamma (m/t0): ", gamma)
total_power_diss_stable = 0.0
steady_state_steps_count = 0

# 主模拟循环
for step in 1:maxstep
    interactions.interactions[1].t += dt
    LangevinDump_step!(dt, cell, interactions, Ts, t0, fixdim=fixdim)  # 关键步骤执行后立即计算功率

    # 日志记录
    if mod(step, logsequence) == 0
        T = cell_temp(cell)
        Ep = cell_energy(cell, interactions)
        Ek = cell_Ek(cell)
        println("step=$step,T=$T,Ep=$Ep,Ek=$Ek,E=$(Ep+Ek)")
        writedlm(logfile, [step, T, Ep, Ek, Ep+Ek])
    end

    # 轨迹记录
    if mod(step, trajseq) == 0
        fix_traj!(TrajList, cell, PBC=PBC)
    end

    provide_cell(cell, dt)

    # 结果保存
    if (mod(step, savetrajseq) == 0) || (step == maxstep)
        #savefig
        p = Plots.plot(dpi=800, xlabel="step", ylabel="y/um")  # 设置x轴和y轴标签
        for i in pltlist
        Plots.plot!(p,[TrajList[i].rl[1,1:end-1]],label="P$i")
        end 
        global trajsavecount
        if ifsavetraj
            jldopen(joinpath(basepath,"traj","traj_$(trajsavecount*savetrajseq)_$((trajsavecount+1)*savetrajseq).jld2"), "w") do file
                write(file, "TrajList", TrajList)
            end
        end

        # if ifsavetraj
        #     jldopen("$basepath/trajm/trajm_$(trajsavecount*savetrajseq)_$((trajsavecount+1)*savetrajseq).jld2", "w") do file
        #         write(file, "TrajList", TrajList)
        #     end
        # end

        savefig(joinpath(basepath,"fig","trajfig_$(trajsavecount*savetrajseq)_$((trajsavecount+1)*savetrajseq).png"))
        println("save traj_$(trajsavecount*savetrajseq)_$((trajsavecount+1)*savetrajseq).")
        trajsavecount+=1
        clear_traj!(TrajList)

        if ifsavefinalstate
            jldopen(joinpath(basepath,"finalcell.jld2"), "w") do file
                write(file, "cell", cell)
            end
        end

    end
# +++ 修正计算耗散功率的逻辑 START +++
# 获取笛卡尔坐标（需要 lattice_vectors 转换）
ltv_mat = SMatrix{3,3}(cell.lattice_vectors)  #晶格向量矩阵
positions_cartesian = [ltv_mat * SVector{3,Float64}(cell.atoms[i].position) for i in 1:nparticle]
step_end_time = step * dt  # 当前步骤结束时间

#计算当前步骤的trap力（非保守力）
current_forces_trap = [trapforce(positions_cartesian[i], step_end_time) for i in 1:nparticle] # 使用现有trapforce函数

# 计算当前步骤的总耗散功率
current_power_diss_step = 0.0
for i in 1:nparticle
momentum_i = SVector{3,Float64}(cell.atoms[i].momentum)
mass_i = cell.atoms[i].mass
if mass_i <= 0
continue
end
velocity_i = momentum_i / mass_i  #速度计算保持不变 仅使用trap力（非保守力）点乘速度计算功率
power_diss_i = dot(current_forces_trap[i], velocity_i)  # 三维点乘（Y/Z方向速度为0时可简化为x分量）
current_power_diss_step += power_diss_i
end
#+++ 修正计算耗散功率的逻辑 END +++

# --- 稳态分析（移至功率计算后） ---
if step > warm_up_steps
total_power_diss_stable += current_power_diss_step  # 直接累加总功率
steady_state_steps_count += 1
end
end  # 主循环闭合
end

# 计算并保存 sigma（修正公式） ---
if steady_state_steps_count > 0
    avg_total_power_diss = total_power_diss_stable / steady_state_steps_count  # 总平均功率
    sigma_entropy_rate = avg_total_power_diss / (Ts * kB_value)  #不再除以nparticle
    sigma_filepath = joinpath(basepath, "sigma.txt")
    open(sigma_filepath, "w") do file
        write(file, string(sigma_entropy_rate))
    end
    println("Sigma entropy rate saved to: ", sigma_filepath)
else
    println("Not enough data for steady-state analysis.")
end

