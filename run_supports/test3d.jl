# 3d test

include("../src/solve_compliance.jl")

# * == 1. Set the domain with Gridap
domain = (0,1,0,0.5,0,0.5)
#partition = (180,90,90)
partition = (30,15,15)
model = CartesianDiscreteModel(domain,partition)
topo  = get_grid_topology(model)
Ω     = Triangulation(model)

labels = get_face_labeling(model)
add_tag_from_tags!(labels,"dirichlet",[13,15,17,19,25]) 
add_tag_from_tags!(labels,"load",[14,16,18,20,26])

dir_tags = ["dirichlet"]
neu_tags = ["load"]
uD(x) = VectorValue(0.0,0.0,0.0)  
g(x) = VectorValue(0.0,1e3*(x[3]<=0.1)*(x[2]>=0.2)*(x[2]<=0.3),-1e3*(x[2]<=0.1)*(x[3]>=0.2)*(x[3]<=0.3))

xc = get_cellcenter_coordinates(Ω)
#= X = [xc[k][1] for k in 1:length(xc)]  # just for masked region
Y = [xc[k][2] for k in 1:length(xc)]
Z = [xc[k][3] for k in 1:length(xc)] =#

opts = ShapeOptParams()
opts.outname = "out_3d"
opts.boundary_labels = labels
opts.dirichlet_tags = dir_tags
opts.neumann_tags = neu_tags
opts.uD = [uD]
opts.g = [g]
#opts.masked_region = @. convert(Float64,√((X - 0.5)^2 + (Y - 0.25)^2) <= 0.1)
opts.vol_target = 0.2 # fraction of total volume
opts.tolremont = 2 # 20 by default
opts.Δt_min = 1e-7 # 1e-5 by default

# remark: optimization region is when mask = 0. Other values may have different usages

ϕ_(x) = -Signum.( sin.(4*2π*x[1]).*cos.(7*2π*x[2]).*cos.(7*2π*x[3]) ,β=2)
ϕ = lazy_map(ϕ_,xc)
phi = vec(collect(get_array(ϕ)))

E = (1e8,1e5)
ν = 0.3
solve_compliance(Ω,phi,E,ν,opts)
