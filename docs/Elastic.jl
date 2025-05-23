"""
@author xyc
@email:22307110070m.fudan.edu.cn
We calculate the elastic properties of Al,Cu and diamond in 0K using structure optimization and in finite temperature using MD simulation.
EMA potential of Al and Cu are implemented in ema.jl 
SW potential for diamond are implemented in sw.jl
Different MD simulation for NPT and NVT are implemented in MD.jl
    include:MD for NPT ensemble:
                        Andersen Nose-Hoover method: Intergrate by 3 order Runge Kuta or multistep of Liouville operator and Tort Decomposition
                        Multistep of Andersen Langvin method
                    NVT ensemble:
                        Nose-Hoover method:Intergrate by 3 order Runge Kuta
                    NVE ensemble:Integrate by 3 order Runge Kuta 
    multi-threaded force computation is implemented on the CPU.
"""
module Elastic

include("modle.jl")
include("MD.jl")
include("pimd.jl")
include("cellmin.jl")
# include("mema.jl")
include("deformer.jl")
include("ema.jl")
include("visualize.jl")
include("wave.jl")
include("sw.jl")
include("molecule.jl")
include("water.jl")
using .Model
# using .MEMA
using .MD
using .PIMD
using .cellmin
using .Deformer
using .EMA
using .Wave
using .SWPotential
using .Visualize
using .MoleculeUtils
using .WaterModel

modules = [Model, MD, cellmin, Deformer,EMA,Visualize,Wave,SWPotential,PIMD,MoleculeUtils,
WaterModel,SWPotential
]

for mod in modules
    for name in names(mod, all=true)
        if !(name in (:eval, :include, :using, :import, :export, :module, :end))
            @eval export $name
        end
    end
end
function set_unit(unit="lj")
    Model.UNIT = unit
    try
        para=getpara()
        @info "unit: $unit, \npara:$para"
    catch
        throw(ArgumentError("Error: Unknown unit system $UNIT"))
    end
end

export set_unit
end