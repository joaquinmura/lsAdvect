using LevelSetAdvection
using Test

# test for Compliance
using Plots
gr()

# auxiliars
mean(x) = sum(x)/length(x)
mkpath("out")

# * == 1. Set the domain with Gridap
domain = (0,1,0,0.5)
partition = (180,90)
model = CartesianDiscreteModel(domain,partition)
topo  = get_grid_topology(model)
Ω     = Triangulation(model)

# and get some info
n      = num_cells(model) # num. elements 
nv     = num_vertices(model) 
dim    = num_dims(model) # 2

# extracts coordinates from the cell centers
xc = get_cellcenter_coordinates(Ω)
d_max = maximum(get_cell_diameter(Ω))

# * == 2. Set boundary conditions
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"dirichlet",[1,3,7]) 
add_tag_from_tags!(labels,"load",8) 

uD(x) = VectorValue(0.0,0.0)  
g(x) = VectorValue(0.0,-10*(x[2]>=0.22)*(x[2]<=0.28))

order = 1
reffe = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
V0 = TestFESpace(model,reffe, conformity=:H1, dirichlet_tags=["dirichlet"]) #  dirichlet_masks=[(true,false), (true,true)])

U  = TrialFESpace(V0,[uD])
degree = 2*order
Ω  = Triangulation(model)
dΩ = Measure(Ω,degree)
Γ  = BoundaryTriangulation(model,tags="load")
dΓ = Measure(Γ,degree)

# * == 3. Levelset initialization
#ϕ_(x) = min.((x[1] .- 0.25).^2 .+ (x[2] .- 0.25).^2 - 0.1^2,(x[1] .- 0.1).^2 .+ (x[2] .- 0.1).^2 - 0.08^2)
ϕ_(x) = -Signum.( sin.(4*2π*x[1]).*cos.(4*4π*x[2]) ,β=2)
ϕ = lazy_map(ϕ_,xc)

each_save  = 10
each_reinit = 2
max_iter   = 100

phi = vec(collect(get_array(ϕ))) ##! AVOID! Too slow
phi = ReinitHJ2d_update(topo,xc,phi,20,scheme="RK2",Δt=0.1*d_max)


# * == 4. Material setting
E = 10e9 * (phi.<=0) + 1e2*(phi.>0)
ν = 0.3

function E_to_C(E)
  
  λ = (E * ν)/((1+ν)*(1-2*ν))
  μ = E/(2*(1+ν))

  C1111 = 2.0*μ + λ
  C1122 = λ
  C1112 = 0.0
  C2222 = 2.0*μ + λ
  C2212 = 0.0
  C1212 = μ
  SymFourthOrderTensorValue(C1111,C1112,C1122,C1112,C1212,C2212,C1122,C2212,C2222)
end
C = lazy_map(E_to_C,E)


# Linear elasticity evaluation: First time
σ(u) = C⊙ε(u)
a(u,v) = ∫( ε(v) ⊙ σ(u) )dΩ
l(v) = ∫(v⋅g)dΓ

op = AffineFEOperator(a,l,U,V0)
uh = solve(op)

# * == 5. Compliance speed
compliance = [] # init

η = 1.0 # 0.2 # volume penalty
area = collect(get_array(∫(1.0)dΩ))
V(u) = σ(u) ⊙ ε(u)
Vc = collect(get_array(∫(V(uh))dΩ)) ./ area #! SLOW
# limiting speed near the boundary
#restric = @. exp(-abs(Vc)/(2*d_max))
#restric = @. restric*(restric>0.45)
#Vc .*= restric
Vc /= mean(Vc) #maximum(Vc)
println("[0] min,max(V) = ",minimum(Vc)," , ",maximum(Vc))
Δt = 0.2*d_max #/maximum(Vc) #<<<< does not allow 'maximum' over lazy array

writevtk(Ω,"out/elasticity_000",cellfields=["uh"=>uh,"epsi"=>ε(uh),"sigma"=>σ(uh)],celldata=["speed"=>Vc,"E"=>E,"phi"=>phi])

# * == 6. Optimization Loop

push!(compliance,sum(l(uh)))

for k in 1:max_iter
  global phi #? this is annoying
  ∇ϕ = upwind2d_step(topo,xc,phi,Vc.-η)

  #global phi = phi .- Δt*∇ϕ # Scope error in Julia
  for i in eachindex(phi)
    phi[i] -= Δt*∇ϕ[i]
  end
  #ϕ = lazy_map(=,ϕ - Δt*∇ϕ) # not working

  # Reinitialization step
  if mod(k,each_reinit)==0
    phi = ReinitHJ2d_update(topo,xc,phi,10,scheme="RK2",Δt=0.1*d_max)
  end

  global E = 10.0e9 * (phi.<=0) + 1e2*(phi.>0)
  global C = lazy_map(E_to_C,E)

  global op = AffineFEOperator(a,l,U,V0) # needed to update C into the problem
  global uh = solve(op)

  global Vc = collect(get_array(∫(V(uh))dΩ)) ./ area
  push!(compliance,sum(l(uh)))

  Vc /= mean(Vc) #maximum(Vc)
  println("[$(k)] min,max(V) = ",minimum(Vc)," , ",maximum(Vc))

  if mod(k,each_save)==0
    println("iter: ",k)
    writevtk(Ω,"out/elasticity_"*lpad(k,3,"0"),cellfields=["uh"=>uh,"epsi"=>ε(uh),"sigma"=>σ(uh)],celldata=["speed"=>Vc,"E"=>E,"phi"=>phi])
  end

  # from https://stackoverflow.com/questions/30789256/is-there-a-way-to-plot-graph-in-julia-while-executing-loops
  pp = plot(compliance, yaxis=:log10)
  display(pp)
  sleep(0.1)

end
