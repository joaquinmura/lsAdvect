#using LevelSetAdvection
include("LevelSetAdvection.jl") # change later to module

using Formatting
using Plots
gr()

Base.@kwdef mutable struct ShapeOptParams
    outname::String = "output"
    boundary_labels
    dirichlet_tags
    neumann_tags
    uD::VectorValue      # set later as list of VectorValues
    g::VectorValue       # same as above
    masked_region        # vector array
    each_save::Int = 10  # set the steps to save data
    each_reinit::Int = 5 # select when to apply reinitialization
    max_iter::Int = 4000 # set maximum number of iterations
    vol_penal::Real = 0.04 # volume penalty
    Δt::Real = 0.025     # Time step
    Δt_min::Real = 1e-5  # minimal time step allowable
    curv_penal::Real = 0 # Penalty factor for curvature during the advection.
    tolremont::Int = 20  # Tolerance to relax descent condition in objective function
end



"""
E[1] for phi<0 and E[2] otherwise
"""
function solve_compliance(Ω::Triangulation,phi,E,ν,sop::ShapeOptParams)
        
    # auxiliars
    println("* Set of solutions will be stored into the $outname folder.")
    mkpath(String)

    # and get some info
    #n      = num_cells(Ω)
    #nv     = num_vertices(Ω) 
    dim        = num_dims(Ω) 
    compliance = []    # defines array to collect objective function values
    volume     = []

    # extracts coordinates from the cell centers
    xc = get_cellcenter_coordinates(Ω)
    d_max = maximum(get_cell_diameter(Ω))

    order = 1
    reffe = ReferenceFE(lagrangian,VectorValue{dim,Float64},order)
    V0 = TestFESpace(model,reffe, conformity=:H1, dirichlet_tags=sop.dirichlet_tags) #  dirichlet_masks=[(true,false), (true,true)])

    U  = TrialFESpace(V0,sop.uD)
    degree = 2*order
    dΩ = Measure(Ω,degree)
    Γ  = BoundaryTriangulation(model,tags=sop.neumann_tags) #! model --> Ω 
    dΓ = Measure(Γ,degree)



    # * == 3. Levelset initialization

    # Iteration parameters


    # Initial shape

    phi = ReinitHJ2d_update(Ω,xc,phi,10,scheme="Upwind",Δt=0.1*d_max) # the very first time


    # * == 4. Material setting
    Eₕ = E[1]*(1 .- sH(phi)) + E[2]*sH(phi)
    νₕ = ν[1]

    println(" --- Compliance minimization ---");
    println(" Parameters:")
    println(" E(void,solid) = ($(E[1]),$(E[2]))")
    println(" Poisson ratio = $νₕ")
    println(" Max. number of iterations   : $(sop.max_iter)")
    println(" Magnitude of Volume penalty : $(sop.vol_penal)")
    println(" Time step                   : $(sop.Δt)")
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

    vol = sum(∫(phi0.<=0)dΩ)
    push!(compliance,sum(l(uh)))
    push!(volume,vol)      

    # limiting speed near the boundary
    Vc_ /= maximum(abs.(Vc_)) 

    printfmtln("[000] compliance={:.4e}  || min,max(V) = ({:.4e} , {:.4e})",compliance[end],minimum(Vc_),maximum(Vc_))
    writevtk(Ω,"out/elasticity_000",cellfields=["uh"=>uh,"epsi"=>ε(uh),"sigma"=>σ(uh,E)],celldata=["speed"=>Vc_,"E"=>E,"phi"=>phi])



    # * == 6. Optimization Loop
    #let phi=phi,Vc_=Vc_,Δt=Δt
    ∇ϕ=nothing
    accepted_step = true
    for k in 1:sop.max_iter
        # new shape
        phi0 = copy(phi)
        if accepted_step
        ∇ϕ = upwind2d_step(Ω,xc,phi0,Vc_ .- sop.vol_penal, curvature_penalty = sop.curv_penal)
        end
        phi0 = lazy_map(-,phi0,sop.Δt .* ∇ϕ)

        # Reinitialization step
        if mod(k,each_reinit)==0
        phi0 = ReinitHJ2d_update(Ω,xc,phi0,5,scheme="Upwind",Δt=0.1*d_max)
        end

        # new displacement field
        Eₕ = E[1]*(1 .- sH(phi0)) + E[2]*sH(phi0)
        uₕ  = solve_elasticity(Eₕ)

        new_compliance = sum(l(uₕ))

        # velocity update
        Vc_ = Vc(uₕ,Eₕ) 
        Vc_ /=  maximum(abs.(Vc_))

        # Volume computation
        vol = sum(∫(phi0.<=0)dΩ)

        printfmtln("[{:03d}] compliance={:.4e} | vol={:.4e}  || min,max(V) =  ({:.4e} , {:.4e})",k,new_compliance,vol,minimum(Vc_),maximum(Vc_))

        #* Checking descent
        if new_compliance < compliance[end] * (1 + tolremont/sqrt(k/2))
            # Acepted step
            push!(compliance,new_compliance)
            push!(volume,vol)      

            #? Slight acceleration
            # Δt *= 1.0025

            phi = copy(phi0) # shape update
            accepted_step = true
            else
            # Rejected step
            sop.Δt *= 0.9
            tt = Formatting.format("                     *rejected* actualcompliance={:.4e} :: decreasing Δt to {:.4e}",compliance[end],sop.Δt)
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
        if sop.Δt < sop.Δt_min
        println(" Convergence achieved!")
        break
        end
        
    end
    #end # let
end