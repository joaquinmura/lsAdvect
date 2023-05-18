#using LevelSetAdvection
include("LevelSetAdvection.jl") # change later to module

using Formatting
using Plots
gr()


"""
    solve_compliance(Ω::Triangulation,phi,E,ν,sop::ShapeOptParams)

Minimal compliance solver using Levelsets and Shape Sensitivity Analysis

This function solves

       min      ∫ f⋅u dΩ
     Ω ∈ U_ad

        s.t.   -div σ = f   in Ω,
                    u = uD  on Γd (Dirichlet)
                   σn = g   on Γn (Neumann)
                   σn = 0   on ∂Ω(Γd ∪ Γn)

For this problem, the domain Ω is embeeded into a larger one D, where 
Ω = { x ∈ D : ϕ(x)<0 }

being ϕ the levelset function for the implicit location of Ω. That function allows
to transport Ω via a Hamilton-Jacobi equation, in order to construct a minimizing sequence 
for the shape optimization problem.


Usage:
compliance,volume = solve_compliance(D::Triangulation,phi::Vector,E,ν,sop::ShapeOptParams)

where the domain D is defined using the `Gridap` finite element package, and the element-wise 
vector phi is such that the Young modulus for each subdomain corresponds to
           {  E[1] when phi(x)<0       
    E(x) = {  E[2] otherwise

The output is given by two vectors containing the values of the objective function (compliance)
and the volume, as the iterations runs.

"""
function solve_compliance(Ω::Triangulation,phi,E,ν,sop::ShapeOptParams)
        
    # auxiliars
    println("* Set of solutions will be stored into the $(sop.outname) folder.")
    mkpath(sop.outname)

    # and get some info
    #n      = num_cells(Ω)
    #nv     = num_vertices(Ω.model) 
    dim        = num_dims(Ω) 
    compliance = []    # defines array to collect objective function values
    Lagrangian = []    # Augmented Lagrangian values
    volume     = []
    η_vol      = []    # Penalty parameter for volume constraint
    λ_vol      = []    # Lagrange multiplier for volume constraint
    h_vol      = []    # volume constraint
    
    λ⁰ = 0.2  # initial Lagrange multiplier
    η⁰ = 0.1
    Δη = 0.1
    η_max = 5
    nR = 25  # penalty relaxation iteration

    reinit_iter = [10,20] # Amount of reinitialization steps
    reinit_iter_thres = 400 # iteration that triggers reinit_iter[2] instead of reinit_iter[1]

    # extracts coordinates from the cell centers
    xc = get_cellcenter_coordinates(Ω)
    d_max = maximum(get_cell_diameter(Ω))

    domain_diameter = get_estimated_domain_diameter(Ω) # for phi limiter

    # defines masked region, if any
    mask_reg = ones(size(phi))
    if length(sop.masked_region)>0
        printstyled("Considering a masked region\n",bold=true,color=:blue)
        global mask_reg = convert.(Float64,abs.(sop.masked_region) .< 1e-3)
        if length(mask_reg)<=1
            @show length(mask_reg)
            error("Working domain is empty!. Check 'masked_region' parameter.")
        end
    end
    opt_region = sH(2*(mask_reg .- 0.5),slope=75)

    # Finite Elements setting
    order = 1
    reffe = ReferenceFE(lagrangian,VectorValue{dim,Float64},order)
    V0 = TestFESpace(model,reffe, conformity=:H1, dirichlet_tags=sop.dirichlet_tags) #  dirichlet_masks=[(true,false), (true,true)])

    U  = TrialFESpace(V0,sop.uD)
    degree = 2*order
    dΩ = Measure(Ω,degree)
    Γ  = BoundaryTriangulation(Ω.model,tags=sop.neumann_tags)
    dΓ = Measure(Γ,degree)



    # * == 3. Levelset initialization

    # Formatting initial shape
    phi = ReinitHJ2d_update(Ω,xc,phi,maximum(reinit_iter),scheme="Upwind",Δt=0.1*d_max) # the very first time


    # * == 4. Material setting
    Eₕ = E[1]*(1 .- sH(phi)) + E[2]*sH(phi)
    νₕ = ν[1]

    function E_to_C(E;ν=νₕ)
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
    V(u,E) = σ(u,E) ⊙ ε(u) 

    #= old style
    # Remark: Vc is a `julia` vector array containing cell-averaged evaluations of V(u,E)
    area = collect(get_array(∫(1.0)dΩ))
    Vc(uh,E) = opt_region .* collect( get_array(∫(V(uh,E))dΩ)) ./ area #! SLOW
    =#

    function Vc(u,E,ϕ)
        #TODO ϕ could be ::Vector{Float64} or LazyArray
        return velocity_regularization(Ω,ϕ,V(u,E))
    end

    # * == 6. Solve first iteration
    uh  = solve_elasticity(Eₕ) # the very first time
    Vc_ = opt_region .* Vc(uh,Eₕ,phi)

    # * == 6.5. Volume constraint
    vol_target = sop.vol_target * sum(∫(1.0)dΩ)
    vol = sum(∫(phi.<=0)dΩ)
    
    push!(volume,vol)      
    push!(h_vol,vol - vol_target) 
    push!(η_vol,η⁰)
    push!(λ_vol,λ⁰)

    push!(compliance,sum(l(uh)))
    push!(Lagrangian,compliance[1] + λ⁰*h_vol[1] + η⁰/2*(h_vol[1])^2)

#TODO now following https://www.sciencedirect.com/science/article/pii/S0045782518305619 "Level set based shape optimization using trimmed hexahedral meshes" CMAME 2019
#? eqs (17) - (18)

    # limiting speed near the boundary
    Vc_ /= maximum(abs.(Vc_)) 

    printfmtln("[000] compliance={:.4e}  || min,max(V) = ({:.4e} , {:.4e})",compliance[end],minimum(Vc_),maximum(Vc_))
    writevtk(Ω,sop.outname*"/elasticity_000",
        cellfields=["uh"=>uh,"epsi"=>ε(uh),"sigma"=>σ(uh,Eₕ)],
        celldata=["speed"=>Vc_,"E"=>Eₕ,"phi"=>phi,"opt_region"=>opt_region]
    )


    # * == 7. Optimization Loop

    println(" --- Compliance minimization ---");
    println(" Parameters:")
    println(" E(void,solid) = ($(E[1]),$(E[2]))")
    println(" Poisson ratio = $νₕ")
    println(" Max. number of iterations   : $(sop.max_iter)")
    println(" Magnitude of Volume penalty : $(sop.vol_penal)")
    println(" Time step                   : $(sop.Δt)")
    println(" Characteristic mesh size    : $d_max")
    println(" Volume  (target)            : $vol_target")
    println("\n\n");


    let phi=phi,Vc_=Vc_

    ∇ϕ=nothing
    accepted_step = true

    for k in 1:sop.max_iter
        # new shape
        phi0 = copy(phi)

        if accepted_step
            # Augmented Lagrangian scheme for volume control
            if k<=nR
                global λ¹ = λ⁰
                global η¹ = η⁰
            else
                global λ¹ = λ_vol[end] .+ η_vol[end].*(volume[end] .- vol_target)
                global η¹ = min(η_vol[end] + Δη,η_max)
            end

            # Velocity update
            vel_N = Vc_ .- (λ¹.*opt_region)
            ∇ϕ = upwind2d_step(Ω,xc,phi0, vel_N, curvature_penalty = sop.curv_penal)

            push!(λ_vol,λ¹)
            push!(η_vol,η¹)
        else
            push!(λ_vol,λ_vol[end])
            push!(η_vol,η_vol[end])
        end

        phi0 = lazy_map(-,phi0,sop.Δt .* ∇ϕ)

        # Reinitialization step
        if mod(k,sop.each_reinit)==0
            riter = reinit_iter[1]*(k<reinit_iter_thres) + reinit_iter[2]*(k>=reinit_iter_thres)
            phi0 = ReinitHJ2d_update(Ω,xc,phi0,riter,scheme="Upwind",Δt=0.25*d_max)
            phi0 = limiter(phi0,h=domain_diameter)
        end

        # new displacement field
        Eₕ = E[1]*(1 .- sH(phi0)) + E[2]*sH(phi0)
        uₕ  = solve_elasticity(Eₕ)

        new_compliance = sum(l(uₕ))

        # velocity update
        Vc_ = opt_region .* Vc(uₕ,Eₕ,phi0) 
        Vc_ /=  maximum(abs.(Vc_))
        
        # Volume computation
        vol = sum(∫(phi0.<=0)dΩ)

        new_h = vol - vol_target
        new_lagrangian = new_compliance + λ_vol[end]*new_h + η_vol[end]/2*(new_h)^2

        printfmtln("[{:03d}] lagrangian={:.4e} | compliance={:.4e} | vol={:.4e}  || (λ,η) =  ({:.4e} , {:.4e})",k,new_lagrangian,new_compliance,vol,λ_vol[end],η_vol[end])

        #* Checking descent
        if new_lagrangian < Lagrangian[end] * (1 + sop.tolremont/sqrt(k/2))
            # Acepted step
            push!(compliance,new_compliance)
            push!(volume,vol)      
            push!(h_vol,new_h)           
            push!(Lagrangian,new_lagrangian)

            #? Slight acceleration
            sop.Δt *= 1 + 0.8*log(1 +k*0.5)/7*0.005 # goes from 1.0 to 1.0035 in 2200 iterations

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

        if mod(k,sop.each_save)==0 && accepted_step
        printstyled("iter: ",k,bold=true,color=:yellow)
        println()
        writevtk(Ω,sop.outname*"/elasticity_"*lpad(k,3,"0"),cellfields=["uh"=>uₕ,"epsi"=>ε(uₕ),"sigma"=>σ(uₕ,Eₕ)],celldata=["speed"=>Vc_,"E"=>Eₕ,"phi"=>phi0])
        end


        pp = plot(compliance, yaxis=:log10, marker=:circle, label="Compliance")
        plot!(pp,Lagrangian, yaxis=:log10, marker=:circle, label="Lagrangian")
        ppv = plot(volume, marker=:square , label="Volume")
        plot!(ppv,[1,length(volume)],vol_target*[1,1], lc=:black, lw=2, label="Target")
        ppp = plot(pp,ppv, layout = (1,2))
        display(ppp)
        sleep(0.01)

        # The End
        if sop.Δt < sop.Δt_min
        println(" Convergence achieved!")
        break
        end
        
    end
    end # let

    return compliance,volume
end