include("../src/solve_compliance.jl")

# * == 1. Set the domain with Gridap
domain = (0,1,0,0.5)
partition = (180,90)
model = CartesianDiscreteModel(domain,partition)
topo  = get_grid_topology(model)
Ω     = Triangulation(model)

labels = get_face_labeling(model)
add_tag_from_tags!(labels,"dirichlet",[1,3,7]) 
add_tag_from_tags!(labels,"load",8) 

dir_tags = ["dirichlet"]
neu_tags = "load" # array?
uD(x) = VectorValue(0.0,0.0)  
g(x) = VectorValue(0.0,-10*(x[2]>=0.22)*(x[2]<=0.28))

xc = get_cellcenter_coordinates(Ω)

mask = 1 # test

opts = ShapeOptParams()
opts.outname = "out_01"
opts.boundary_labels = labels
opts.dir_tags = dir_tags
opts.neu_tags = neu_tags
opts.uD = [uD]
opts.g = g
opts.mask = mask
#=
opts.each_save = 10  # set the steps to save data
opts.each_reinit = 5 # select when to apply reinitialization
opts.max_iter = 4000 # set maximum number of iterations
opts.vol_penal = 0.04 # volume penalty
opts.Δt = 0.025      # Time step
opts.Δt_min = 1e-5   # minimal time step allowable
opts.curv_penal = 0  # Penalty factor for curvature during the advection.
opts.tolremont = 20  # Tolerance to relax descent condition in objective function
=#

ϕ_(x) = -Signum.( sin.(4*2π*x[1]).*cos.(5*2π*x[2]) ,β=2)
ϕ = lazy_map(ϕ_,xc)
phi = vec(collect(get_array(ϕ))) ##! AVOID! Too slow

E = (1e8,1e5)
ν = 0.3
solve_compliance(Ω::Triangulation,phi,E,ν,sop::ShapeOptParams)