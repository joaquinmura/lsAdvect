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
neu_tags = ["load"]
uD(x) = VectorValue(0.0,0.0)  
g(x) = VectorValue(0.0,-10*(x[2]>=0.22)*(x[2]<=0.28))

xc = get_cellcenter_coordinates(Ω)

mask = 1 # test

opts = ShapeOptParams()
opts.outname = "out_01"
opts.boundary_labels = labels
opts.dirichlet_tags = dir_tags
opts.neumann_tags = neu_tags
opts.uD = [uD]
opts.g = [g]
opts.masked_region = mask


ϕ_(x) = -Signum.( sin.(4*2π*x[1]).*cos.(5*2π*x[2]) ,β=2)
ϕ = lazy_map(ϕ_,xc)
phi = vec(collect(get_array(ϕ))) ##! AVOID! Too slow

E = (1e8,1e5)
ν = 0.3
solve_compliance(Ω,phi,E,ν,opts)