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
X = [xc[k][1] for k in 1:length(xc)]
Y = [xc[k][2] for k in 1:length(xc)]

opts = ShapeOptParams()
opts.outname = "out_01"
opts.boundary_labels = labels
opts.dirichlet_tags = dir_tags
opts.neumann_tags = neu_tags
opts.uD = [uD]
opts.g = [g]
opts.masked_region = @. convert(Float64,√((X - 0.5)^2 + (Y - 0.25)^2) <= 0.1)
opts.vol_target = 0.3 # fraction of total volume

# remark: optimization region is when mask = 0. Other values may have different usages

ϕ_(x) = -Signum.( sin.(4*2π*x[1]).*cos.(5*2π*x[2]) ,β=2)
ϕ = lazy_map(ϕ_,xc)
phi = vec(collect(get_array(ϕ))) ##! AVOID! Too slow

E = (1e8,1e5)
ν = 0.3
solve_compliance(Ω,phi,E,ν,opts)
#solve_compliance_NL(Ω,phi,E,ν,opts) # not yet