using StaticArrays
# using Plots
using LinearAlgebra
using LsqFit
# Ensure Elastic.jl is in the correct path or in your Julia LOAD_PATH
# For example, if Elastic.jl is in "src/Elastic.jl" relative to the script:
include("src/Elastic.jl") # This should contain your Model module
using .Elastic # Assuming your Model module is accessible via Elastic
using FFMPEG # For saving plots if Plots.jl uses it as a backend for animations/movies
using DelimitedFiles
using Distributions
using JLD2
using Base.Threads
using Plots
using Random
using JSON
using Printf # For @sprintf

# Ensure GKSwstype is set for headless environments if needed
ENV["GKSwstype"] = "100"
rdseed = rand(10^(4-1):10^(8)-1)
Random.seed!(rdseed)
println("Number of threads: ", Threads.nthreads())

# --- Path Configuration (User's Preferred Logic) ---
data_root = "" # As per user's snippet, ensures local output
println("Data root (as per user preference): '$data_root'") 
tt = 0.5052622965408168 #ms/[t]
g_gravity_example = 10.0 #um/ms^2 (from user snippet, currently unused in script logic)
projectname = "brownianparticle_2state_on=50_off=50" # From user snippet, for output directory AND config
Comments = ""  # From user snippet

# --- Unit System ---
unit = "lj"
set_unit(unit) 

# --- Cell Parameters ---
lattice_constant = 40.0 

# --- Particle Parameters ---
nparticle = 32
xmax = lattice_constant 
mass = 4.2 

# --- Trap Parameters (Time-Dependent) ---
ktrap = 10.0
e = 2.0  # E = A sin(w x)/(1 + e + cos(w x))   
deltat_on = 50.0 / tt 
deltat_off = 50.0 / tt 
N = 5      
phi0 = 0.0  

# --- Interaction Parameters ---
ifinteraction = false 
yukawak = 2.0
yukawaF0 = 1.0
yukawacut = 2 * yukawak
epsilon = 1.0 
sigma_interaction = 2.0   # User's name for LJ sigma

# --- Ensemble Parameters ---
Ts = 1.0  
t0 = 0.1 / tt 
if t0 <= 0.0 error("Relaxation time t0 must be positive!") end
gamma_friction = mass / t0 
println("Calculated friction coefficient gamma (m/t0): ", gamma_friction)
kB_value = 1.0 

# --- Simulation Parameters ---
fixdim = [2, 3] 
maxstep = 5000000 
dt = 0.1 
println("Simulation time: $(dt*maxstep*tt*10^(-3)) s")
warm_up_steps = div(maxstep, 1.5) 

# --- Results Saving Parameters ---
trajseq = 100 
savetrajseq = div(maxstep, 20) 
ifsavetraj = true
PBC = false 
ifsavefinalstate = true
pltlist = 1:nparticle 
logsequence = 1000 
debug_print_sequence = logsequence * 10 

# --- Derived Property ---
phi = sigma_interaction * nparticle / (2 * lattice_constant) 

# --- Lattice Definition ---
lattice_vectors = collect((Matrix([ 
    lattice_constant 0.0 0.0; 
    0.0 lattice_constant 0.0; 
    0.0 0.0 lattice_constant] 
))') 
inv_lattice_vectors = inv(Matrix(lattice_vectors)) 

# --- Potential and Force Definitions ---
function lj(r::Float64) 
    if r < sigma_interaction * 2^(1/6)
        sr6 = (sigma_interaction / r)^6
        return 4 * epsilon * (sr6^2 - sr6) + epsilon 
    else
        return 0.0
    end
end

function Flj(r_vec::SVector{3,Float64}) 
    r = norm(r_vec)
    if r == 0.0 return zeros(SVector{3,Float64}) end
    if r < sigma_interaction * 2^(1/6) && r > 0.0
        # User's original Flj form:
        return 24*epsilon*(2*(sigma_interaction/r)^14-(sigma_interaction/r)^8)*r_vec/sigma_interaction^2
    else
        return zeros(SVector{3,Float64})
    end
end
wca_cutoff = sigma_interaction * 2^(1/6)

# 将A_factor函数改回为A函数，保持与brownianparticle_2state_xyc.jl一致
function A(e::Float64)
    return sqrt(e) * sqrt(2 + e)
end

function trapenergy(r0::SVector{3,Float64}, t::Float64)  
    nr0 = r0[1]  # 只使用x坐标，与brownianparticle_2state_xyc.jl一致
    deltat = deltat_on + deltat_off
    tn = mod(t, deltat)
    w = pi / lattice_constant * N
    if tn < deltat_on
        return ktrap * A(e) * sin(w * nr0 + phi0) / (1 + e + cos(w * nr0 + phi0))
    else
        return 0.0
    end
end

function trapforce(r0::SVector{3,Float64}, t::Float64)
    nr0 = r0[1]  # 只使用x坐标，与brownianparticle_2state_xyc.jl一致
    deltat = deltat_on + deltat_off
    tn = mod(t, deltat)
    w = pi / lattice_constant * N
    if tn < deltat_on
        # 根据brownianparticle_2state_xyc.jl的公式
        force_vector = ktrap * A(e) * w * (cos(nr0 * w + phi0) + sin(w * nr0 + phi0)^2 / (1 + e + cos(w * nr0 + phi0))) / (1 + e + cos(w * nr0 + phi0)) * (-r0 / nr0)
        # 确保返回正确类型的向量
        return SVector{3,Float64}(force_vector[1], force_vector[2], force_vector[3])
    else
        return zeros(SVector{3,Float64})
    end
end

function YukawaE(r_norm::Float64)
    if r_norm == 0; return Inf; end
    return yukawaF0 * exp(-yukawak * r_norm) / r_norm
end

function YukawaF(r_vec::SVector{3,Float64})
    r_norm = norm(r_vec)
    if r_norm == 0; return zeros(SVector{3,Float64}); end
    return (yukawaF0*exp(-yukawak*r_norm)*(1.0+yukawak*r_norm)/r_norm^3 ).* r_vec
end

# --- User's Energy Verification Functions (Moved Up) ---
function correct_trap_energy(current_cell::UnitCell, current_interactions::Interactions)
    total_trap_energy = 0.0
    trap_interaction_local = current_interactions.interactions[1] 
    trap_time_local = trap_interaction_local.t
    for i_atom_E in 1:length(current_cell.atoms)
        pos_fractional_E = SVector{3,Float64}(current_cell.atoms[i_atom_E].position)
        pos_cartesian_E_vec = Matrix(current_cell.lattice_vectors) * pos_fractional_E 
        pos_cartesian_E_svec = SVector{3,Float64}(pos_cartesian_E_vec)
        energy_i_E = trapenergy(pos_cartesian_E_svec, trap_time_local) 
        total_trap_energy += energy_i_E
    end
    return total_trap_energy
end

function debug_energy_calculation(current_cell::UnitCell, current_interactions::Interactions, current_step::Int)
    calculated_trap_energy = correct_trap_energy(current_cell, current_interactions)
    if mod(current_step, debug_print_sequence) == 0 
        original_elastic_energy = cell_energy(current_cell, current_interactions) 
        println("DEBUG_ENERGY @ step $(current_step): Elastic.cell_energy=$(@sprintf("%.6e", original_elastic_energy)), " *
               "Recalculated_trap_energy_only=$(@sprintf("%.6e", calculated_trap_energy)), " *
               "Difference (if only trap exists)=$(@sprintf("%.6e", abs(original_elastic_energy - calculated_trap_energy)))")
    end
    if !ifinteraction
        return calculated_trap_energy 
    else
        @warn "debug_energy_calculation returns only TRAP energy. If other interactions active, logged Ep is underestimate." maxlog=1
        return calculated_trap_energy 
    end
end

function verify_energy_calculation(current_cell::UnitCell, current_interactions::Interactions) 
    println("\n=== Energy Calculation Verification ===")
    orig_energy = cell_energy(current_cell, current_interactions) 
    println("Original energy from Elastic.jl: $orig_energy")
    corrected_energy = correct_trap_energy(current_cell, current_interactions) 
    println("Corrected trap energy (using Cartesian coords for trap): $corrected_energy")
    energy_diff = abs(orig_energy - corrected_energy)
    if !ifinteraction && abs(orig_energy)>1e-10
        diff_percent = energy_diff / abs(orig_energy) * 100.0
        println("Energy difference (trap only case): $energy_diff ($(round(diff_percent, digits=2))%)")
        if diff_percent > 1.0
            @warn "Trap energy calculation difference exceeds 1%. Check Elastic.jl's trap energy part."
        else
            println("  Trap energy difference is small.")
        end
    elseif ifinteraction
         println("Energy difference (trap part vs total): $energy_diff. Cannot compute meaningful percentage.")
    end
    println("=== End of Energy Verification ===\n")
end

# --- Initial Particle Configuration (User's method) ---
initial_real_positions_x_range = range(-xmax, xmax, length=nparticle+1)[1:end-1]
atom_positions_frac_temp = [[xi_val + lattice_constant/nparticle, 0.0, 0.0] ./ lattice_constant for xi_val in initial_real_positions_x_range]
atoms = [Atom(Vector(pos_frac),mass) for pos_frac in atom_positions_frac_temp];

println("Initial fractional atom positions (first 3):")
for i in 1:min(3,nparticle)
    println("  Particle $i fractional_pos: ", atoms[i].position) 
end
inicell = UnitCell(lattice_vectors, atoms)

# --- Interactions Setup ---
trap_interaction = MutableField(trapenergy, trapforce, 0.0)
interactions_list = Vector{AbstractInteraction}([trap_interaction])
neighbor_lists = Vector{Neighbor}([Neighbor()]) 
if ifinteraction
    yukawa_interaction = Interaction(YukawaE, YukawaF, yukawacut, 0.1)
    push!(interactions_list, yukawa_interaction)
    push!(neighbor_lists, Neighbor(inicell))
    wca_interaction = Interaction(lj, Flj, wca_cutoff, 0.1)
    push!(interactions_list, wca_interaction)
    push!(neighbor_lists, Neighbor(inicell))
end
interactions = Interactions(interactions_list, neighbor_lists)
@show interactions

# --- Output Directory Setup (User's Preferred Logic) ---
global basepath_actual_run = joinpath(data_root, "output", projectname) 
if !isdir(basepath_actual_run)
    mkpath(basepath_actual_run)
    println("Directory $basepath_actual_run created.\n")
else
    local counter_suffix = 1
    local original_basepath_for_suffix = basepath_actual_run 
    local newpath_with_suffix = original_basepath_for_suffix * "_$counter_suffix" 
    while isdir(newpath_with_suffix)
        counter_suffix += 1
        newpath_with_suffix = original_basepath_for_suffix * "_$counter_suffix"
    end
    mkpath(newpath_with_suffix)
    println("Base directory $original_basepath_for_suffix already exists. Created new directory for this run: $newpath_with_suffix.\n")
    basepath_actual_run = newpath_with_suffix 
end
println("Final effective basepath for this run: $basepath_actual_run")

# --- Configuration Saving ---
config_dict = Dict(
    "projectname" => projectname, 
    "ActualOutputBasepath" => basepath_actual_run, 
    "Comments" => Comments, 
    "RandomSeed" => rdseed,
    "UnitSystem" => unit,
    "TimeConversion_ms_per_tLJ" => tt,
    "Gravity_example_g" => g_gravity_example, 
    "Cell" => Dict( "lattice_vectors" => lattice_vectors, "PBC_setting_for_traj" => PBC ),
    "Particle" => Dict( "nparticle" => nparticle, "mass" => mass, "initial_xmax_distrib_real_space" => xmax,
        "avg_initial_distance_1_2_real_space" => nparticle > 1 ? norm(Matrix(lattice_vectors)*(atoms[2].position - atoms[1].position)) : "N/A"),
    "Trap" => Dict( "type" => "TimeDependentSinusoidalUserForce", "ktrap" => ktrap, "e_eccentricity" => e, "N_periods" => N,
        "phi0_phase" => phi0, "deltat_on_LJ" => deltat_on, "deltat_off_LJ" => deltat_off, "deltat_on_ms" => deltat_on * tt, "deltat_off_ms" => deltat_off * tt ),
    "InteractionsOnOff" => Dict( "ifinteraction_particle_particle" => ifinteraction, "Yukawa_k" => ifinteraction ? yukawak : "N/A",
        "Yukawa_F0" => ifinteraction ? yukawaF0 : "N/A", "Yukawa_cutoff" => ifinteraction ? yukawacut : "N/A",
        "WCA_epsilon" => ifinteraction ? epsilon : "N/A", "WCA_sigma_interaction" => ifinteraction ? sigma_interaction : "N/A", "WCA_cutoff" => ifinteraction ? wca_cutoff : "N/A" ),
    "Ensemble" => Dict( "Ts_temperature_LJ" => Ts, "t0_thermostat_relax_time_LJ" => t0, "gamma_friction_coeff" => gamma_friction, "kB_value" => kB_value ),
    "Simulation" => Dict( "integrator" => "LangevinDump", "maxstep" => maxstep, "dt_LJ" => dt, "total_time_LJ" => dt * maxstep,
        "total_time_s" => dt * maxstep * tt * 10^(-3), "fixdim" => fixdim, "warm_up_steps" => warm_up_steps ),
    "Results" => Dict( "logsequence" => logsequence, "traj_record_every_N_steps" => trajseq, "traj_save_to_file_every_N_steps" => savetrajseq,
        "ifsavetraj_to_jld2" => ifsavetraj, "ifsavefinalstate" => ifsavefinalstate, "plot_particle_indices" => string(pltlist) ),
    "DerivedProperties" => Dict( "phi_packing_fraction_approx" => phi )
)
config_json_path = joinpath(basepath_actual_run, "Config.json")
open(config_json_path, "w") do file 
    write(file, JSON.json(config_dict)) 
end
println("Configuration saved to $config_json_path")

mkpath(joinpath(basepath_actual_run,"traj"))
mkpath(joinpath(basepath_actual_run,"fig"))

# --- Plot Initial Trap Potential and Force ---
x_plot_config = -lattice_constant:0.01:lattice_constant 
y1_trap_E_config = [trapenergy(SVector{3,Float64}([xi,0.0,0.0]),0.0) for xi in x_plot_config]
y2_trap_F_config = [trapforce(SVector{3,Float64}([xi,0.0,0.0]),0.0)[1] for xi in x_plot_config]
p_trap_config = plot(
    plot(x_plot_config, y1_trap_E_config, title="Energy"),  
    plot(x_plot_config, y2_trap_F_config, title="Force"),  
    layout = (1, 2), size = (800, 400)            
)
savefig(joinpath(basepath_actual_run,"fig","trap.png")) 

x_time_plot_config = 0:1.0:5000.0 
y1_trap_E_vs_t_config = [trapenergy(SVector{3,Float64}([1.0,0.0,0.0]),xi) for xi in x_time_plot_config]
y2_trap_F_vs_t_config = [trapforce(SVector{3,Float64}([1.0,0.0,0.0]),xi)[1] for xi in x_time_plot_config]
p_trap_t_config = plot(
    plot(x_time_plot_config, y1_trap_E_vs_t_config, title="Energy"),  
    plot(x_time_plot_config, y2_trap_F_vs_t_config, title="Force"),  
    layout = (1, 2), size = (800, 400)            
)
savefig(joinpath(basepath_actual_run,"fig","trap_t.png")) 

# --- Simulation ---
cell = deepcopy(inicell) 
fix_dim!(cell, fixdim) 
update_rmat!(cell) 
interactions.interactions[1].t = 0.0 
update_fmat!(cell, interactions) 

println("Actual initial fractional positions in cell after setup:") 
for i_atom in 1: nparticle
    println("Particle $i_atom: ", cell.atoms[i_atom].position)
end
println("\nCoordinate system verification (after cell setup):")
println("lattice_vectors (first row): ", lattice_vectors[1, :])
println("cell.atoms[1].position (should be fractional): ", cell.atoms[1].position)
fractional_pos_example_cell = cell.atoms[1].position
cartesian_pos_example_cell = Matrix(lattice_vectors) * SVector{3,Float64}(fractional_pos_example_cell) 
println("Converted to cartesian from cell: ", cartesian_pos_example_cell)

verify_energy_calculation(cell, interactions) 

max_points_per_segment = ceil(Int, savetrajseq / trajseq)
dt_for_trajectory_object = trajseq * dt 
TrajList = [Trajectory(max_points_per_segment, dt_for_trajectory_object) for _ in 1:nparticle]

total_power_diss_stable = Ref(0.0)
steady_state_steps_count = Ref(0)
trajsavecount = 0 

# 在文件开头添加以下变量声明（在其他global变量之后）
power_recording_interval = 100  # 每隔多少步记录一次功率
power_history = Vector{Tuple{Float64, Float64, Bool}}()  # (时间, 平均功率, 陷阱是否开启)

println("\nStarting simulation loop...")
log_file_path = joinpath(basepath_actual_run, "log.txt")
open(log_file_path, "w") do logfile
    header = "Step\tTemperature\tPotentialEnergy\tKineticEnergy\tTotalEnergy\tInstantaneousTrapPowerSum"
    println(logfile, header)
    println(header) 
    flush(logfile)

    for step_sim in 1:maxstep 
        interactions.interactions[1].t += dt 
        LangevinDump_step!(dt, cell, interactions, Ts, t0, fixdim=fixdim)

        current_trap_time = interactions.interactions[1].t
        power_sum_this_step = 0.0
        local F_trap_p1 = zeros(SVector{3,Float64})
        local V_p1 = zeros(SVector{3,Float64})
        local power_p1 = 0.0

        for i in 1:nparticle
            momentum_i = SVector{3,Float64}(cell.atoms[i].momentum) 
            if mass <= 0.0; continue; end
            velocity_i = momentum_i / mass 
            
            position_fractional = SVector{3,Float64}(cell.atoms[i].position)
            position_cartesian_vec = Matrix(lattice_vectors) * position_fractional # Result is Vector
            position_cartesian_svec = SVector{3,Float64}(position_cartesian_vec) # Convert to SVector
            
            F_trap_on_i = trapforce(position_cartesian_svec, current_trap_time) # Pass SVector
            power_i_val = LinearAlgebra.dot(F_trap_on_i, velocity_i) 
            power_sum_this_step += power_i_val

            if i == 1 
                F_trap_p1 = F_trap_on_i; V_p1 = velocity_i; power_p1 = power_i_val
                if mod(step_sim, debug_print_sequence) == 0 
                    println("DEBUG_COORDS @ step $(step_sim): P1_frac=$(position_fractional), P1_cart=$(position_cartesian_svec)") # Print SVector
                end
            end
        end

        # 计算所有粒子的平均功率
        avg_power_this_step = power_sum_this_step / nparticle
        
        # 记录功率历史
        if mod(step_sim, power_recording_interval) == 0
            current_time = step_sim * dt
            tn_cycle = mod(current_trap_time, (deltat_on + deltat_off))
            trap_is_on = tn_cycle < deltat_on
            push!(power_history, (current_time, avg_power_this_step, trap_is_on))
        end

        if step_sim > warm_up_steps
            total_power_diss_stable[] += power_sum_this_step
            steady_state_steps_count[] += 1
            if mod(step_sim, debug_print_sequence) == 0 && nparticle > 0
                avg_power_so_far = steady_state_steps_count[] > 0 ? total_power_diss_stable[] / steady_state_steps_count[] : 0.0
                println("DEBUG_POWER @ step $(step_sim): P_inst=$(@sprintf("%.4e", power_sum_this_step)), P_avg_accum=$(@sprintf("%.4e", avg_power_so_far))")
                println("DEBUG_PARTICLE_1 @ step $(step_sim): F_trap=$(F_trap_p1), V=$(V_p1), Power_p1=$(@sprintf("%.4e", power_p1))")
                tn_cycle_debug = mod(current_trap_time, (deltat_on + deltat_off))
                trap_is_on_debug = tn_cycle_debug < deltat_on
                println("DEBUG_TRAP_STATE @ step $(step_sim): t_trap=$(@sprintf("%.2f", current_trap_time)), tn_cycle=$(@sprintf("%.2f", tn_cycle_debug)), trap_on=$trap_is_on_debug")
            end
        end

        if mod(step_sim, logsequence) == 0
            T_current = cell_temp(cell)
            Ep_current = debug_energy_calculation(cell, interactions, step_sim)
            Ek_current = cell_Ek(cell)
            E_total_current = Ep_current + Ek_current
            
            println("step=$step_sim,T=$T_current,Ep=$Ep_current,Ek=$Ek_current,E=$E_total_current,P_inst=$power_sum_this_step")
            writedlm(logfile,[step_sim T_current Ep_current Ek_current E_total_current power_sum_this_step]) 
            flush(logfile)
        end

        if mod(step_sim, trajseq) == 0
            fix_traj!(TrajList, cell, PBC=PBC) 
        end
        
        provide_cell(cell,dt) # User script

        if (mod(step_sim,savetrajseq)==0) || (step_sim==maxstep)
            p_traj_segment = Plots.plot(dpi=300, xlabel="Time [LJ units]", ylabel="x-position (real space) [LJ units]", title="Trajectory Segment")
            segment_start_time_offset = trajsavecount * savetrajseq * dt 
            for particle_idx_plot in pltlist 
                if particle_idx_plot > nparticle continue end
                num_valid_points = TrajList[particle_idx_plot].ti - 1 
                if num_valid_points > 0
                    segment_times = (0:(num_valid_points-1)) .* TrajList[particle_idx_plot].dt
                    absolute_times_for_plot = segment_start_time_offset .+ segment_times
                    positions_x_for_plot = TrajList[particle_idx_plot].rl[1, 1:num_valid_points]
                    Plots.plot!(p_traj_segment, absolute_times_for_plot, positions_x_for_plot, label="P$particle_idx_plot")
                end
            end
            
            filename_label_start_user = trajsavecount * savetrajseq 
            filename_label_end_user = (step_sim == maxstep) ? maxstep : (trajsavecount + 1) * savetrajseq
            
            plot_filename = joinpath(basepath_actual_run, "fig", "trajfig_$(filename_label_start_user)_to_$(filename_label_end_user).png")
            savefig(p_traj_segment, plot_filename)
            println("Saved trajectory plot: $plot_filename")

            if ifsavetraj 
                jld2_filename = joinpath(basepath_actual_run, "traj", "traj_$(filename_label_start_user)_to_$(filename_label_end_user).jld2") 
                println("Saving trajectory data segment to $jld2_filename...")
                try
                    jldopen(jld2_filename, "w") do file
                        write(file, "TrajList", TrajList) 
                        write(file, "simulation_start_step_of_segment", filename_label_start_user + 1) 
                        write(file, "simulation_end_step_of_segment", filename_label_end_user)
                        write(file, "dt_simulation_step", dt) 
                        write(file, "traj_recording_interval_steps", trajseq) 
                    end
                    println("  Successfully saved trajectory data and metadata.")
                catch e_save
                    println("ERROR saving JLD2 trajectory file $jld2_filename: $e_save")
                end
            end
            
            global trajsavecount
            
            trajsavecount += 1 
            
            if step_sim < maxstep
                clear_traj!(TrajList) 
            end
        end
        
        if step_sim == maxstep && ifsavefinalstate 
            finalstate_filename = joinpath(basepath_actual_run, "finalcell.jld2") 
            println("Saving final cell state to $finalstate_filename...")
            try
                jldopen(finalstate_filename, "w") do file
                    write(file, "cell", cell) 
                    write(file, "simulation_time_at_end_LJ", interactions.interactions[1].t) 
                end
                println("  Successfully saved final cell state.")
            catch e_final
                println("ERROR saving final cell state: $e_final")
            end
        end
    end 
end 

println("\nSimulation loop finished.")

# --- Post-simulation: Calculate and Save TUR-related Quantities (Sigma) ---
sigma_entropy_rate_per_particle = NaN 
avg_total_trap_power_input = NaN
avg_single_particle_trap_power_input = NaN

if steady_state_steps_count[] > 0
    avg_total_trap_power_input = total_power_diss_stable[] / steady_state_steps_count[]
    avg_single_particle_trap_power_input = avg_total_trap_power_input / nparticle
    sigma_entropy_rate_per_particle = avg_single_particle_trap_power_input / (kB_value * Ts)

    println("\n--- Steady State Summary (based on trap power input) ---")
    println("Total accumulated trap power input sum in steady state: ", total_power_diss_stable[])
    println("Number of steady state steps counted: ", steady_state_steps_count[])
    println("Warm-up steps: ", warm_up_steps)
    println("Average TOTAL power input by trap (all particles): ", avg_total_trap_power_input)
    println("Average SINGLE particle power input by trap: ", avg_single_particle_trap_power_input)
    println("Calculated Sigma_entropy_production_rate_per_particle_LJ (before writing to file): ", sigma_entropy_rate_per_particle)

    sigma_filepath = joinpath(basepath_actual_run, "tur_summary_trap_power.txt")
    open(sigma_filepath, "w") do file
        write(file, "TUR Analysis Summary (based on power input by time-dependent trap)\n")
        write(file, "-----------------------------------------------------------------\n")
        write(file, "Actual_output_basepath: $basepath_actual_run\n")
        write(file, "Warm_up_steps: $warm_up_steps\n")
        write(file, "Steady_state_steps_counted: $(steady_state_steps_count[])\n") 
        write(file, "Total_accumulated_trap_power_input_steady_state_LJ: $(total_power_diss_stable[])\n") 
        write(file, "Average_total_trap_power_input_all_particles_LJ: $avg_total_trap_power_input\n")
        write(file, "Average_single_particle_trap_power_input_LJ: $avg_single_particle_trap_power_input\n")
        write(file, "Bath_temperature_Ts_LJ: $Ts\n")
        write(file, "Boltzmann_constant_kB_LJ: $kB_value\n")
        write(file, "Number_of_particles: $nparticle\n")
        write(file, "Sigma_entropy_production_rate_per_particle_LJ: $sigma_entropy_rate_per_particle\n")
    end
    println("TUR related quantities (based on trap power) saved to $sigma_filepath")
else
    println("Warning: No steady-state steps recorded (steady_state_steps_count = $(steady_state_steps_count[])).") 
    println("         Sigma calculation (based on trap power) skipped. Check warm_up_steps ($warm_up_steps) vs maxstep ($maxstep).")
end

println("\nScript execution finished.")

# --- (可选) 添加绘制完整轨迹的函数 (来自用户脚本) ---
function plot_complete_trajectory(output_basepath_func::String) 
    println("\nGenerating complete trajectory plot...")
    traj_dir = joinpath(output_basepath_func, "traj")
    
    traj_files_found = filter(f -> startswith(f, "traj_") && endswith(f, ".jld2") && !occursin("steps",f), readdir(traj_dir)) 
    
    function get_start_step_from_filename_user_format(fname_str)
        parts = split(basename(fname_str), '_') 
        if length(parts) >= 2 && occursin(r"^\d+$", parts[2])
            try return parse(Int, parts[2]) catch; end
        end
        @warn "Could not parse start step from user filename format: $fname_str. Using typemax(Int)."
        return typemax(Int) 
    end
    sort!(traj_files_found, by = get_start_step_from_filename_user_format)
    
    if isempty(traj_files_found)
        println("No trajectory segment files (user format: traj_START_END.jld2) found in $traj_dir.")
        return
    end
    
    full_trajectory_time = Dict(i => Float64[] for i in 1:nparticle) 
    full_trajectory_x_pos = Dict(i => Float64[] for i in 1:nparticle)
    
    println("Found $(length(traj_files_found)) trajectory segments (user format).")

    last_segment_end_time = 0.0 

    for (file_idx, fname) in enumerate(traj_files_found)
        file_path = joinpath(traj_dir, fname)
        println("  Loading segment $(file_idx)/$(length(traj_files_found)): $fname")
        try
            jldopen(file_path, "r") do f
                if !haskey(f, "TrajList")
                    @warn "Skipping $fname: Missing 'TrajList' data."
                    return 
                end
                traj_list_segment_f = f["TrajList"]
                
                for p_idx_f in 1:nparticle 
                    if p_idx_f > length(traj_list_segment_f) continue end 
                    tr_obj_f = traj_list_segment_f[p_idx_f]
                    num_points_in_tr_obj_f = tr_obj_f.ti - 1 
                    
                    if num_points_in_tr_obj_f > 0
                        times_this_segment_f = last_segment_end_time .+ (0:(num_points_in_tr_obj_f-1)) .* tr_obj_f.dt
                        positions_this_segment_f = tr_obj_f.rl[1, 1:num_points_in_tr_obj_f]
                        
                        append!(full_trajectory_time[p_idx_f], times_this_segment_f)
                        append!(full_trajectory_x_pos[p_idx_f], positions_this_segment_f)
                        
                        if p_idx_f == 1 && !isempty(times_this_segment_f) 
                            last_segment_end_time = times_this_segment_f[end] + tr_obj_f.dt 
                        end
                    end
                end
            end
        catch e_f
            println("Error reading or processing trajectory file $fname: $e_f")
        end
    end
    
    means_offset = Dict()
    for p_idx_offset in 1:nparticle
        if haskey(full_trajectory_x_pos, p_idx_offset) && !isempty(full_trajectory_x_pos[p_idx_offset])
            means_offset[p_idx_offset] = mean(full_trajectory_x_pos[p_idx_offset])
        end
    end
    all_means_offset = filter(!isnan, collect(values(means_offset))) 
    if !isempty(all_means_offset)
        global_mean_offset = mean(all_means_offset)
        println("DEBUG plot_complete_trajectory: Detected trajectory mean x-coordinate: $global_mean_offset")
        if abs(global_mean_offset) > lattice_constant 
            offset_guess_val = sign(global_mean_offset) * floor(abs(global_mean_offset) / lattice_constant) * lattice_constant
            println("DEBUG plot_complete_trajectory: Detected potential offset, attempting correction: $offset_guess_val")
            for p_idx_offset2 in 1:nparticle
                if haskey(full_trajectory_x_pos, p_idx_offset2) && !isempty(full_trajectory_x_pos[p_idx_offset2])
                    full_trajectory_x_pos[p_idx_offset2] .-= offset_guess_val
                end
            end
            println("DEBUG plot_complete_trajectory: Applied coordinate correction: $(-offset_guess_val)")
        end
    end
    
    p_complete_user = plot(dpi=300, size=(1200, 800), title="Complete Particle Trajectories", 
                      xlabel="Time [LJ units]", ylabel="x-position [LJ units]",
                      legend=:outerright)  # 将图例放在图表右侧外部
    
    for p_idx_plot_user in 1:nparticle  # 遍历所有粒子
        if haskey(full_trajectory_time, p_idx_plot_user) && !isempty(full_trajectory_time[p_idx_plot_user]) && 
           haskey(full_trajectory_x_pos, p_idx_plot_user) && !isempty(full_trajectory_x_pos[p_idx_plot_user])
            plot!(p_complete_user, full_trajectory_time[p_idx_plot_user], full_trajectory_x_pos[p_idx_plot_user], 
                  label="P$p_idx_plot_user", linewidth=1.0, alpha=0.8)
        else
            println("No data to plot for Particle $p_idx_plot_user in user's plot_complete_trajectory")
        end
    end
    
    warm_up_time_point_user = warm_up_steps * dt 
    all_times_flat_check_user = vcat([arr for arr in values(full_trajectory_time) if !isempty(arr)]...) 
    if !isempty(all_times_flat_check_user) && minimum(all_times_flat_check_user) <= warm_up_time_point_user <= maximum(all_times_flat_check_user)
        vline!(p_complete_user, [warm_up_time_point_user], label="End of warm-up", linestyle=:dash, linewidth=2, color=:black)
    end
    
    plot_filename_user_png = joinpath(output_basepath_func, "fig", "complete_trajectory_x_vs_TIME_merged.png") 
    savefig(p_complete_user, plot_filename_user_png)
    println("Saved complete trajectory plot (PNG from user func) to: $plot_filename_user_png")
    
    plot_filename_user_pdf = joinpath(output_basepath_func, "fig", "complete_trajectory_x_vs_TIME_merged.pdf") 
    savefig(p_complete_user, plot_filename_user_pdf)
    println("Saved complete trajectory plot (PDF from user func) to: $plot_filename_user_pdf")
end

if ifsavetraj 
    try
        plot_complete_trajectory(basepath_actual_run) 
    catch e
        println("Error generating complete trajectory plot using user's function: $e")
        showerror(stdout, e) 
        Base.show_backtrace(stdout, backtrace()) 
    end
end

# 在模拟结束后添加（在计算sigma之后）
# 分析并可视化功率时间序列
if !isempty(power_history)
    # 提取时间和功率数据
    power_times = [p[1] for p in power_history]
    power_values = [p[2] for p in power_history]
    trap_state = [p[3] for p in power_history]
    
    # 分离陷阱打开和关闭状态的功率
    on_times = power_times[trap_state]
    on_powers = power_values[trap_state]
    off_times = power_times[.!trap_state]
    off_powers = power_values[.!trap_state]
    
    # 计算基本统计量
    mean_power = mean(power_values)
    mean_on_power = isempty(on_powers) ? NaN : mean(on_powers)
    mean_off_power = isempty(off_powers) ? NaN : mean(off_powers)
    
    # 统计正负值比例
    positive_powers = filter(p -> p > 0, power_values)
    negative_powers = filter(p -> p < 0, power_values)
    percent_positive = 100 * length(positive_powers) / length(power_values)
    percent_negative = 100 * length(negative_powers) / length(power_values)
    
    # 输出统计信息
    println("\n--- 粒子平均功率(F*v)时间序列分析 ---")
    println("记录的功率数据点数: $(length(power_history))")
    println("总平均功率: $(mean_power)")
    println("陷阱打开时平均功率: $(mean_on_power)")
    println("陷阱关闭时平均功率: $(mean_off_power)")
    println("正功率比例: $(percent_positive)%")
    println("负功率比例: $(percent_negative)%")
    println("最大正功率: $(isempty(positive_powers) ? "N/A" : maximum(positive_powers))")
    println("最小负功率: $(isempty(negative_powers) ? "N/A" : minimum(negative_powers))")
    
    # 绘制功率随时间的变化图
    power_plot = plot(dpi=300, title="Average Power per Particle (F·v) Time Series",
                      xlabel="Time [LJ units]", ylabel="Power [LJ units]",
                      size=(1200, 800))
    
    # 绘制陷阱打开时的功率（使用不同颜色）
    if !isempty(on_times)
        scatter!(power_plot, on_times, on_powers, 
                 label="Trap ON", markersize=3, alpha=0.6, color=:blue)
    end
    
    # 绘制陷阱关闭时的功率
    if !isempty(off_times)
        scatter!(power_plot, off_times, off_powers, 
                 label="Trap OFF", markersize=3, alpha=0.6, color=:red)
    end
    
    # 添加零线和均值线
    hline!(power_plot, [0.0], linestyle=:dash, color=:black, label="Zero power")
    hline!(power_plot, [mean_power], linestyle=:solid, color=:green, 
           linewidth=2, label="Mean power (all)")
    
    # 标记warm-up结束时间
    vline!(power_plot, [warm_up_steps * dt], linestyle=:dot, color=:purple, 
           linewidth=2, label="End of warm-up")
    
    # 保存图表
    power_plot_path = joinpath(basepath_actual_run, "fig", "power_time_series.png")
    savefig(power_plot, power_plot_path)
    println("功率时间序列图已保存至: $power_plot_path")
    
    # 保存原始数据
    power_data_path = joinpath(basepath_actual_run, "power_time_series.txt")
    open(power_data_path, "w") do f
        println(f, "# 时间[LJ units]\t平均功率(F·v)[LJ units]\t陷阱状态(1=ON,0=OFF)")
        for (t, p, s) in power_history
            println(f, "$(t)\t$(p)\t$(Int(s))")
        end
    end
    println("功率时间序列数据已保存至: $power_data_path")
    
    # 绘制功率分布直方图（分别绘制陷阱ON和OFF的情况）
    p_hist = plot(layout=(2,1), dpi=300, size=(900, 800))
    
    # ON状态直方图
    if !isempty(on_powers)
        histogram!(p_hist[1], on_powers, 
                   bins=50, 
                   title="Power Distribution (Trap ON)",
                   xlabel="Power [LJ units]",
                   ylabel="Frequency",
                   legend=false,
                   color=:blue,
                   alpha=0.7)
        vline!(p_hist[1], [mean_on_power], color=:green, linewidth=2, label="Mean")
        vline!(p_hist[1], [0.0], color=:black, linewidth=2, linestyle=:dash, label="Zero")
    end
    
    # OFF状态直方图
    if !isempty(off_powers)
        histogram!(p_hist[2], off_powers, 
                   bins=50, 
                   title="Power Distribution (Trap OFF)",
                   xlabel="Power [LJ units]",
                   ylabel="Frequency",
                   legend=false,
                   color=:red,
                   alpha=0.7)
        vline!(p_hist[2], [mean_off_power], color=:green, linewidth=2, label="Mean")
        vline!(p_hist[2], [0.0], color=:black, linewidth=2, linestyle=:dash, label="Zero")
    end
    
    # 保存直方图
    power_hist_path = joinpath(basepath_actual_run, "fig", "power_distribution.png")
    savefig(p_hist, power_hist_path)
    println("功率分布直方图已保存至: $power_hist_path")
    
    # 分析热平衡后的功率数据
    warm_up_idx = findfirst(t -> t >= warm_up_steps * dt, power_times)
    if !isnothing(warm_up_idx)
        stable_times = power_times[warm_up_idx:end]
        stable_powers = power_values[warm_up_idx:end]
        stable_trap = trap_state[warm_up_idx:end]
        
        stable_on_powers = stable_powers[stable_trap]
        stable_off_powers = stable_powers[.!stable_trap]
        
        # 热平衡后的统计
        mean_stable_power = mean(stable_powers)
        mean_stable_on = isempty(stable_on_powers) ? NaN : mean(stable_on_powers)
        mean_stable_off = isempty(stable_off_powers) ? NaN : mean(stable_off_powers)
        
        println("\n--- 热平衡后的功率统计 ---")
        println("热平衡后数据点数: $(length(stable_powers))")
        println("热平衡后总平均功率: $(mean_stable_power)")
        println("热平衡后陷阱打开时平均功率: $(mean_stable_on)")
        println("热平衡后陷阱关闭时平均功率: $(mean_stable_off)")
        
        positive_stable = stable_powers .> 0
        println("热平衡后正功率比例: $(100 * sum(positive_stable) / length(stable_powers))%")
    end
end

