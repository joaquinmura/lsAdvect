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

# Iteration parameters
each_save  = 10    # set the steps to save data
each_reinit = 3    # select when to apply reinitialization
max_iter   = 1000   # set maximum number of iterations
compliance = []    # defines array to collect objective function values
η = 0.04           # volume penalty
Δt = 0.3*d_max     # Time step
Δt_min = 1e-5      # minimal time step allowable
curv_penal = 0     # Penalty factor for curvature during the advection.
tolremont = 20     # Tolerance to relax descent condition in objective function

# Initial shape
#ϕ_(x) = min.((x[1] .- 0.25).^2 .+ (x[2] .- 0.25).^2 - 0.1^2,(x[1] .- 0.1).^2 .+ (x[2] .- 0.1).^2 - 0.08^2)
ϕ_(x) = -Signum.( sin.(4*2π*x[1]).*cos.(5*2π*x[2]) ,β=2)
ϕ = lazy_map(ϕ_,xc)
phi = vec(collect(get_array(ϕ))) ##! AVOID! Too slow
phi = ReinitHJ2d_update(Ω,xc,phi,10,scheme="Upwind",Δt=0.1*d_max) # the very first time


# * == 4. Material setting
E₀ = 1e8
E₁ = 1e5 #1e2
E = E₀ * (phi.<=0) + E₁*(phi.>0) # this might be improved with softthresholding or similar ...
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

# Linear elasticity evaluation: First time
function σ(u,E)
  C = lazy_map(E_to_C,E)
  return C⊙ε(u)
end
l(v) = ∫(v⋅g)dΓ

function solve_elasticity(E)
  a(u,v) = ∫( ε(v) ⊙ σ(u,E) )dΩ
  op = AffineFEOperator(a,l,U,V0)
  return solve(op)
end


# * == 5. Compliance speed
area = collect(get_array(∫(1.0)dΩ))

V(u,E) = σ(u,E) ⊙ ε(u)

# Remark: Vc is a `julia` vector array containing cell-averaged evaluations of V(u,E)
Vc(uh,E) = collect(get_array(∫(V(uh,E))dΩ)) ./ area #! SLOW

# * == 6. Solve first iteration
uh  = solve_elasticity(E) # the very first time
Vc_ = Vc(uh,E)

push!(compliance,sum(l(uh)))

# limiting speed near the boundary
restric = @. exp(-abs(Vc_).^2/(2*d_max))
restric = @. restric*(restric>0.45)
Vc_ .*= restric
Vc_ /= maximum(abs.(Vc_)) 

printfmtln("[000] compliance={:.4e}  || min,max(V) = ({:.4e} , {:.4e})",compliance[end],minimum(Vc_),maximum(Vc_))
writevtk(Ω,"out/elasticity_000",cellfields=["uh"=>uh,"epsi"=>ε(uh),"sigma"=>σ(uh,E)],celldata=["speed"=>Vc_,"E"=>E,"phi"=>phi])



  # * == 6. Optimization Loop
let phi=phi,Vc_=Vc_,Δt=Δt
  ∇ϕ=nothing
  accepted_step = true
  for k in 1:max_iter
    # new shape
    phi0 = copy(phi)
    if accepted_step
      ∇ϕ = upwind2d_step(Ω,xc,phi0,Vc_ .- η, curvature_penalty = curv_penal)
    end
    phi0 = lazy_map(-,phi0,Δt.*∇ϕ)

    # Reinitialization step
    if mod(k,each_reinit)==0
      phi0 = ReinitHJ2d_update(Ω,xc,phi0,5,scheme="Upwind",Δt=0.1*d_max)
    end

    # new displacement field
    #Eₕ = E₀ * (phi0.<=0) + E₁*(phi0.>0)
    Eₕ = E₀*(1 .- sH(phi0)) + E₁*sH(phi0)
    uₕ  = solve_elasticity(Eₕ)

    new_compliance = sum(l(uₕ))

    # velocity update
    Vc_ = Vc(uₕ,Eₕ) # 1st eval
    #= apply restriction around {ϕ(x)=0}
    Vc_ /=  maximum(abs.(Vc_))
    restric = @. exp(-abs(Vc_).^2/(6*d_max^2))
    restric = @. restric*(restric>0.4)
    Vc_ .*= restric =#
    # normalization
    Vc_ /=  maximum(abs.(Vc_))

    printfmtln("[{:03d}] compliance={:.4e}  || min,max(V) =  ({:.4e} , {:.4e})",k,new_compliance,minimum(Vc_),maximum(Vc_))

    #* Checking descent
    if new_compliance < compliance[end] * (1 + tolremont/sqrt(k/2))
      # Acepted step
      push!(compliance,new_compliance)
      phi = copy(phi0) # shape update
      accepted_step = true
    else
      # Rejected step
      Δt *= 0.9
      tt = Formatting.format("                     *rejected* actualcompliance={:.4e} :: decreasing Δt to {:.4e}",compliance[end],Δt)
      printstyled(tt,color=:cyan)
      println()
      accepted_step = false
    end

    if mod(k,each_save)==0 && accepted_step
      printstyled("iter: ",k,bold=true,color=:yellow)
      println()
      writevtk(Ω,"out/elasticity_"*lpad(k,3,"0"),cellfields=["uh"=>uₕ,"epsi"=>ε(uₕ),"sigma"=>σ(uₕ,Eₕ)],celldata=["speed"=>Vc_,"E"=>Eₕ,"phi"=>phi0])
    end

    pp = plot(compliance, yaxis=:log10, marker=:circle)
    display(pp)
    sleep(0.1)

    # The End
    if Δt < Δt_min
      println(" Convergence achieved!")
      break
    end
    
  end
end # let