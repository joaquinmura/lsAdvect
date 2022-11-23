#using LevelSetAdvection
include("../src/LevelSetAdvection.jl")
using Test
using Formatting

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
each_reinit = 2000
max_iter   = 100
compliance = [] # init
η = 0*0.1 #0.2 # volume penalty
Δt = 0.4*d_max #/maximum(Vc) #<<<< does not allow 'maximum' over lazy array


phi = vec(collect(get_array(ϕ))) ##! AVOID! Too slow
phi = ReinitHJ2d_update(topo,xc,phi,20,scheme="Upwind",Δt=0.1*d_max)


# * == 4. Material setting
E₀ = 1e8
E₁ = 1e2
E = E₀ * (phi.<=0) + E₁*(phi.>0)
ν = 0.3

println(" --- Compliance minimization ---");
println(" Parameters:")
println(" E(void,solid) = ($E₀,$E₁)")
println(" Poisson ratio = $ν")
println(" Max. number of iterations   : $max_iter")
println(" Magnitude of Volume penalty : $η")
println(" Time step                   : $Δt")
println(" Characteristic mesh size    : $d_max")
println("\n\n");

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
function σ(u,E)
  C = lazy_map(E_to_C,E)
  return C⊙ε(u)
end
a(u,v) = ∫( ε(v) ⊙ σ(u,E) )dΩ
l(v) = ∫(v⋅g)dΓ

function solve_elasticity(E)
  op = AffineFEOperator(a,l,U,V0)
  return solve(op)
end


# * == 5. Compliance speed
area = collect(get_array(∫(1.0)dΩ))

V(u,E) = σ(u,E) ⊙ ε(u)

Vc(uh,E) = collect(get_array(∫(V(uh,E))dΩ)) ./ area #! SLOW

# * == 6. Solve first iteration
uh  = solve_elasticity(E) # the very first time
Vc_ = Vc(uh,E)

push!(compliance,sum(l(uh)))

# limiting speed near the boundary
#restric = @. exp(-abs(Vc)/(2*d_max))
#restric = @. restric*(restric>0.45)
#Vc .*= restric
Vc_ /= abs(mean(Vc_)) #mean(Vc) #maximum(Vc)

printfmtln("[000] min,max(V) = ({:.4e} , {:.4e})",minimum(Vc_),maximum(Vc_))

writevtk(Ω,"out/elasticity_000",cellfields=["uh"=>uh,"epsi"=>ε(uh),"sigma"=>σ(uh,E)],celldata=["speed"=>Vc_,"E"=>E,"phi"=>phi])



# * == 6. Optimization Loop

for k in 1:max_iter
  global phi,Vc_ #? this is annoying
  phi0 = phi
  ∇ϕ = upwind2d_step(topo,xc,phi,Vc_ .- η)

  
  for i in eachindex(phi)
    phi[i] -= Δt*∇ϕ[i] # this works
  end 
  
  #ϕ = lazy_map(=,ϕ .- Δt.*∇ϕ) # not working
  #ϕ = lazy_map(-,Δt.*∇ϕ) #! ϕ is not updated! just local var!!
  #phi = lazy_map(-,phi,Δt.*∇ϕ) 

  # Reinitialization step
  if mod(k,each_reinit)==0
    phi = ReinitHJ2d_update(topo,xc,phi,50,scheme="Upwind",Δt=0.1*d_max) # antes: 10 iter
  end
#! Sin reinit no actualiza phi ... por que?????

  Eₕ = E₀ * (phi.<=0) + E₁*(phi.>0)

  uₕ  = solve_elasticity(Eₕ) # the very first time
  Vc_ = Vc(uₕ,Eₕ)
  
  push!(compliance,sum(l(uₕ)))

  Vc_ /= abs(mean(Vc_)) #mean(Vc) #maximum(Vc)
  printfmtln("[{:03d}] min,max(V) =  ({:.4e} , {:.4e})",k,minimum(Vc_),maximum(Vc_))

  @show sum(isnan.(phi))
  @show norm(phi - phi0)

  if mod(k,each_save)==0
    println("iter: ",k)
    writevtk(Ω,"out/elasticity_"*lpad(k,3,"0"),cellfields=["uh"=>uₕ,"epsi"=>ε(uₕ),"sigma"=>σ(uₕ,Eₕ)],celldata=["speed"=>Vc_,"E"=>Eₕ,"phi"=>phi])
  end

  
  pp = plot(compliance, yaxis=:log10, marker=:circle)
  display(pp)
  sleep(0.1)

  #! Missing control over objective function

end
