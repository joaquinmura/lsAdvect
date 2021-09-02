module LevelSetAdvection

export  get_cellcenter_coordinates, get_cell_diameter,
        upwind2d_step, ReinitHJ2d_update, Signum

##############################################
using Reexport
@reexport using Gridap
@reexport using Gridap.Geometry # get_grid_topology
@reexport using Gridap.ReferenceFEs # get_faces
@reexport using Gridap.TensorValues
##############################################

# Auxiliar functions
Signum(x;β=2.0) = tanh.(β.*x)    
SigNum(x;ϵ=0.1) = x/√(x^2 + ϵ^2) 

Sₑ(ϕ;ϵ=0.1) = 0.5/ϵ.*(1 .+ cos.(π*ϕ/ϵ)).*(abs.(ϕ).<=ϵ)
Hₑ(ϕ;ϵ=0.1) = 1.0.*(ϕ.>ϵ) .+ 0.5*(1 .+ ϕ/ϵ .+ 1/ϵ*sin.(π*ϕ/ϵ)).*(abs.(ϕ).<=ϵ)


"""
Godunov Hamiltonian 2D:
Hg(sgnϕ,a,b,c,d) ≈ |∇ϕ|

"""
function Hg(sgnϕ,a,b,c,d)
    h = 0

    aM = a*(a>=0)
    am = a - aM
    bM = b*(b>=0)
    bm = b - bM
    cM = c*(c>=0)
    cm = c - cM
    dM = d*(d>=0)
    dm = d - dM

    if sgnϕ>0
        h = √(max(am^2,bM^2) + max(cm^2,dM^2) )
    else
        h = √(max(aM^2,bm^2) + max(cM^2,dm^2) )
    end

    return h
end



"""
Get cell centered coordinates
x,y = get_center_coord(Ω::Triangulation)
"""
function get_cellcenter_coordinates(Ω::Triangulation)
    #=
    # This is slower
    xy = get_cell_points(Ω) # all vertices in cells
    xc = [sum(xy.cell_phys_point[i])/length(xy.cell_phys_point[i]) for i in 1:length(xy.cell_phys_point)] # center at cell
    =#
    # this is faster
    xy = get_cell_coordinates(Ω) # same as get_cell_points(topo)
    midpoint(xs) = sum(xs)/length(xs)
    xc = lazy_map(midpoint,xy)
    return xc
end



"""
Get the cell diameters
(perhaps there is a better way)

diam::Vector = get_cell_diameter(Ω::Triangulation)
"""
function get_cell_diameter(Ω::Triangulation)
    xy = get_cell_coordinates(Ω)
    #diam(x) = maximum(norm.(x[][:] - x[][end:-1:1])) #! not all combinations! [1.2.3.4] - [4,3,2,1] ... but seems to be enough?!
    #return lazy_map(diam,xy) # not working
    #? not optimal
    d = zeros(num_cells(Ω))
    for (k,x) in enumerate(xy)
        d_ij = 0.0        
        for i in eachindex(x)
            for j in i:length(x)
                d_ij = max(d_ij,norm(x[i] - x[j]))
            end
        end
        d[k] = d_ij
    end

    return d
end



"""
Local Finite Difference derivatives across faces

returns ϕx⁺, ϕx⁻, ϕy⁺, ϕy⁻
"""
function local_FD_derivatives(ϕ,S,N,W,E,xc)
    #? Trick: We repeat to BoundsError when 'e' is on the boundary 
    S = repeat(S,2) 
    N = repeat(N,2)
    E = repeat(E,2)
    W = repeat(W,2)

    # Finite differences: According to the ordering in Gridap #!CHECK for unstructured!
    # * Here we avoid division by 0 on the boundary
    ϕyM = (ϕ[N[2]] - ϕ[N[1]])/(xc[N[2]][2]-xc[N[1]][2] + 1e-30)  # Dy⁺(ϕ)  point upwards
    ϕym = (ϕ[S[2]] - ϕ[S[1]])/(xc[S[2]][2]-xc[S[1]][2] + 1e-30)  # Dy⁻(ϕ)  point downwards
    ϕxM = (ϕ[E[2]] - ϕ[E[1]])/(xc[E[2]][1]-xc[E[1]][1] + 1e-30)  # Dx⁺(ϕ)  point to the right
    ϕxm = (ϕ[W[2]] - ϕ[W[1]])/(xc[W[2]][1]-xc[W[1]][1] + 1e-30)  # Dx⁻(ϕ)  point to the left
    
    return ϕxM,ϕxm,ϕyM,ϕym
end



"""
simple Upwind 2D step

We want to solve

∂ϕ/∂t + V|∇ϕ| = 0

       ϕ(t=0) = ϕ₀

Via Godunov FD scheme:
`Gϕ = upwind2d_step(G::GridTopology,xc::AbstractArray,ϕ::AbstractArray,V::AbstractArray)`

where 
Gϕ ≈ V|∇ϕ|

Then, update your function as
ϕ(n+1) = ϕ(n) - Δt⋅Gϕ(n)
"""
function upwind2d_step(topo::GridTopology,xc,ϕ::AbstractArray,V::AbstractArray)

    # Recover mesh connectivity
    nelem = num_cells(topo) #length(topo.cell_type)
   
    if nelem != length(ϕ)
        error("Vector ϕ has different length (",length(ϕ),") to number of cells in the mesh (",nelem,")")
    end

    dim = num_cell_dims(topo)
    cell_to_faces  = get_faces(topo,dim,dim-1)
    face_to_cells  = get_faces(topo,dim-1,dim)

    if dim !=2
        error("Sorry: 3D is not implemented yet in `ls_advection.jl`")
    end

    g = Array{Float64,1}(undef,length(ϕ)) 
    
    for e in 1:nelem

        # Recovering neighbourhood around element 'e'
        S,N,W,E = face_to_cells[cell_to_faces[e]] # should have 4 terms in 2D-structured grid
        ϕxM,ϕxm,ϕyM,ϕym = local_FD_derivatives(ϕ,S,N,W,E,xc)

        # Godunov:
        g⁺ = √(max(min(ϕxM,0)^2,max(ϕxm,0)^2) + max(min(ϕyM,0)^2,max(ϕym,0)^2))
        g⁻ = √(max(max(ϕxM,0)^2,min(ϕxm,0)^2) + max(max(ϕyM,0)^2,min(ϕym,0)^2))

        V⁺ = max(V[e],0)  # upwind with V magnitude
        V⁻ = min(V[e],0)

        g[e] = V⁺ * g⁺ + V⁻ * g⁻ 
    end

    return g
end



"""
solves some time steps for

    ∂ϕ/∂t + sign(ϕ₀)(|∇ϕ|-1) = 0     in the Grid 

                      ϕ(t=0) = ϕ₀    
"""
function ReinitHJ2d_update(topo::GridTopology,xc,ϕ::AbstractArray,niter::Int;vmax::AbstractFloat=1.0,scheme="RK2",show_errormsg::Bool=false,Δt::Real=0.1)
    # Euler & RK2 copyed from 
    # "*On reinitilizing level set functions*", C. Min, J Comput.Phys 299:8 (2010) DOI:10.1016/j.jcp.2009.12.032

    #TODO: Check if we are on QUADs (and structured!?)

    # Recover mesh connectivity
    nelem = length(topo.cell_type)
   
    if nelem != length(ϕ)
        error("Vector ϕ has different length (",length(ϕ),") to number of cells in the mesh (",nelem,")")
    end

    dim = num_cell_dims(topo)
    cell_to_faces  = get_faces(topo,dim,dim-1)
    face_to_cells  = get_faces(topo,dim-1,dim)

    if dim !=2
        error("Sorry: 3D is not implemented yet in `ls_advection.jl`")
    end

    #TODO: Choose the signum function: TEST required!
    #signψ₀ = sign.(ϕ)
    signψ₀ = Signum(ϕ)
    #signψ₀ = SigNum.(ϕ,ϵ=dx)
    #signψ₀ = lazy_map(SigNum(x,ϵ=1.2*dx)->x,ϕ) # not working

    CFL = 0.3 # later: Let the user choose it

    if lowercase(scheme) == "upwind"
        #? Upwind scheme
        # slow convergence and show oscillations near the boundaries, even with smooth signum function and Laplacian regularization

        # init
        δ = Array{Float64,1}(undef,length(ϕ)) ###

        res0 = 1e+10        

        for iter in 1:niter
            for e in 1:nelem

                # Recovering neighbourhood around element 'e'
                S,N,W,E = face_to_cells[cell_to_faces[e]] # should be 4 in 2D!
                ϕxM,ϕxm,ϕyM,ϕym = local_FD_derivatives(ϕ,S,N,W,E,xc)

                if signψ₀[e]<0
                    δ[e] = √(max(min(ϕxM,0)^2,max(ϕxm,0)^2) + max(min(ϕyM,0)^2,max(ϕym,0)^2)) - 1.0 # g⁻ + 1
                else
                    δ[e] = √(max(max(ϕxM,0)^2,min(ϕxm,0)^2) + max(max(ϕyM,0)^2,min(ϕym,0)^2)) - 1.0 # g⁺ - 1
                end
                δ[e] = δ[e]*signψ₀[e] #- ν*Δϕ  # when applies the smoothed signum

            end

            # This is the scheme with Laplacian regularization:
            # ϕ(n+1) = ϕ(n) - Δt⋅sign(ϕ₀)⋅(|∇ϕ(n)| - 1) + Δt⋅ν⋅Δϕ
            # ϕ(n+1) ≈ ϕ(n) - Δt[ -(ϕ₀<0)⋅g⁻ + (ϕ₀>0)⋅g⁺ - sign(ϕ₀) ] + Δt⋅ν⋅Δϕ
            # ϕ(n+1) ≈ ϕ(n) - Δt[ -(ϕ₀<0)⋅(g⁻-1) + (ϕ₀>0)⋅(g⁺-1) ] + Δt⋅ν⋅Δϕ
            # ϕ(n+1) ≈ ϕ(n) - Δt⋅[sign(ϕ₀)⋅δ(n) - ν⋅Δϕ(n)]
            
            res1 = norm(δ)

            if res0<res1
                Δt *= 0.9
                if show_errormsg
                    print("R.")
                end
            end
    
            ϕ = ϕ - Δt .* δ 

            if show_errormsg
                println("residue[$(iter)]: ",res1*Δt)
            end
            res0=res1
        end

    elseif lowercase(scheme) == "euler"
        #? Forward Euler method using one-sided ENO Finite Differences
        # better than upwind

        ν = 0.001  # smoothing ... just in case

        ϕ1 = copy(ϕ)
        for iter in 1:niter
            ϕold = ϕ
            
            for e in 1:nelem

                # Recovering neighbourhood around element 'e'
                S,N,W,E = face_to_cells[cell_to_faces[e]] # should be 4 in 2D!
                ϕxM,ϕxm,ϕyM,ϕym = local_FD_derivatives(ϕ,S,N,W,E,xc)

                ϕ[e] = ϕ[e] - Δt*signψ₀[e]*(Hg(signψ₀[e],ϕxM,ϕxm,ϕyM,ϕym) - 1.0) #+ ν*Δϕ
            end

            if show_errormsg
                error_ = norm(ϕ - ϕold)/norm(ϕold)
                println("error[$(iter)] = ",error_)
            end
        end

    elseif lowercase(scheme) == "rk2"
        #? TVD Runge–Kutta method
        ϕ1 = ϕ
        ϕ2 = ϕ
        
        for iter in 1:niter
            ϕold = ϕ

            # step 1 in RK2
            for e in 1:nelem

                # Recovering neighbourhood around element 'e'
                S,N,W,E = face_to_cells[cell_to_faces[e]] # should be 4 in 2D!
                ϕxM,ϕxm,ϕyM,ϕym = local_FD_derivatives(ϕ,S,N,W,E,xc)

                ϕ1[e] = ϕ[e] - Δt*signψ₀[e]*(Hg(signψ₀[e],ϕxM,ϕxm,ϕyM,ϕym) - 1.0)
            end
            # step 2 in RK2
            for e in 1:nelem

                # Recovering neighbourhood around element 'e'
                S,N,W,E = face_to_cells[cell_to_faces[e]] # should be 4 in 2D!
                ϕxM,ϕxm,ϕyM,ϕym = local_FD_derivatives(ϕ,S,N,W,E,xc)

                ϕ2[e] = ϕ1[e] - Δt*signψ₀[e]*(Hg(signψ₀[e],ϕxM,ϕxm,ϕyM,ϕym) - 1.0)
            end
            # step 3 in RK2
            ϕ = (ϕ1 + ϕ2)/2.0

            if show_errormsg
                error_ = norm(ϕ - ϕold)/norm(ϕold)
                println("error[$(iter)] = ",error_)
            end
        end

    end # if scheme ...

    return ϕ
end

end
