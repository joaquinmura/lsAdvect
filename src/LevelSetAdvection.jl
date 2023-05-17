#=
module LevelSetAdvection

export  get_cellcenter_coordinates, get_cell_diameter,
        upwind2d_step, ReinitHJ2d_update, Signum
=#
# check last 'end' in this file

##############################################
using Reexport
@reexport using Gridap
@reexport using Gridap.Geometry # get_grid_topology
@reexport using Gridap.ReferenceFEs # get_faces
@reexport using Gridap.TensorValues
##############################################

mutable struct ShapeOptParams
    outname::String
    boundary_labels::FaceLabeling
    dirichlet_tags::Vector{String}
    neumann_tags ::Vector{String}
    uD::Vector{Function}      # set later as list of VectorValues
    g::Vector{Function}       # same as above
    masked_region::AbstractArray # set masked region element-wise
    each_save::Int       # set the steps to save data
    each_reinit::Int     # select when to apply reinitialization
    max_iter::Int        # set maximum number of iterations
    vol_penal::Real      # volume penalty
    Δt::Real             # Time step
    Δt_min::Real         # minimal time step allowable
    curv_penal::Real     # Penalty factor for curvature during the advection.
    tolremont::Int       # Tolerance to relax descent condition in objective function
    vol_target::Real  #

    function ShapeOptParams(outname="output", each_save=10,
                each_reinit=3, max_iter=4000, vol_penal=0.03,
                Δt=7e-3, Δt_min=1e-5, curv_penal=0, tolremont=20,
                vol_target=0.1)
        shapeOptParams = new()
        shapeOptParams.outname = outname
        shapeOptParams.each_save = each_save
        shapeOptParams.each_reinit = each_reinit
        shapeOptParams.max_iter = max_iter
        shapeOptParams.vol_penal = vol_penal
        shapeOptParams.Δt = Δt
        shapeOptParams.Δt_min = Δt_min
        shapeOptParams.curv_penal = curv_penal
        shapeOptParams.tolremont = tolremont
        shapeOptParams.vol_target = vol_target
        return shapeOptParams
    end

end




# Auxiliar functions
Signum(x;β=2.0) = tanh.(β.*x)    
SigNum(x;ϵ=0.1) = x/√(x^2 + ϵ^2) 

#Sₑ(ϕ;ϵ=0.1) = 0.5/ϵ.*(1 .+ cos.(π*ϕ/ϵ)).*(abs.(ϕ).<=ϵ)
#Hₑ(ϕ;ϵ=0.1) = @. 1.0*(ϕ>ϵ) + 0.5*(1 + ϕ/ϵ + 1/ϵ*sin(π*ϕ/ϵ))*(abs(ϕ)<=ϵ)

limiter(x;h::Real=1) = @. x*(abs(x)<=h) + h*(x>h) - h*(x<-h)



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
Smoothed step function
    sH(x;slope::Real=10)

"""
function sH(x;slope::Real=50)
    k = π*slope/2
    return @. (1 + tanh(k*x))/2
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
For instance, works with cartesian grids
"""
function get_estimated_domain_diameter(Ω::Triangulation)
    # list with vertex coordinates
    xy = get_vertex_coordinates(Ω.model.grid_topology)

    # get "corner" nodes from cartesian grid, assuming cartesian ordering
    node1 = minimum(get_cell_node_ids(Ω)[1])
    node2 = maximum(get_cell_node_ids(Ω)[end])
    # Rem: this should be closed to diagonal corners ... but cannot be assessed!

    return norm(xy[node2] - xy[node1])
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
    #= original structure of N,S,E,W
     N = [real north , center]
     S = [center , real south]
     E = [real east , center]
     W = [center, real west]

     length(N) = 1 when there is no 'real north' cell. This is why the repetition avoids undefs
    =#
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
idx2ij(idx,grid_partition)

converts an index 'idx' to a pair (i,j) according to the tuple grid_partition = (ncols,nrows).

Recall that the numbering in Gridap goes from left to right, and then going upwards.
"""
function idx2ij(idx,grid_partition)
    ncols = grid_partition[1]
    i = ceil.(Int,idx./ncols)
    j = 1 .+ mod.(idx .- 1,ncols)
    return i,j
end

"""
ij2idx(idx,grid_partition)

converts a pair (i,j) to an index 'idx' according to the tuple grid_partition = (ncols,nrows).

Recall that the numbering in Gridap goes from left to right, and then going upwards.
"""
function ij2idx(i,j,grid_partition)
    j .+ (i .- 1).*grid_partition[1]
end

"""
global_first_derivatives(Ω::Triangulation,ϕ::AbstractArray)

Use central differences and returns ϕ_x and ϕ_y for inner cells, whenever Ω is constructed as CartesianDiscreteModel
"""
function global_first_derivatives(Ω::Triangulation,ϕ::AbstractArray)
    grid_partition = size(Ω.grid.cell_node_ids)
    nelem = prod(grid_partition)

    ϕx = zeros(Float64,nelem)
    ϕy = zeros(Float64,nelem)

    # interior points only
    for j in 2:grid_partition[1]-1
        for i in 2:grid_partition[2]-1
            ij = ij2idx(i,j,grid_partition)
            i1j = ij2idx(i+1,j,grid_partition)
            ij1 = ij2idx(i,j+1,grid_partition)
            i_1j = ij2idx(i-1,j,grid_partition)
            ij_1 = ij2idx(i,j-1,grid_partition)

            ϕx[ij] = (ϕ[i1j] - ϕ[i_1j])/2
            ϕy[ij] = (ϕ[ij1] - ϕ[ij_1])/2
        end
    end
    
    return ϕx,ϕy
end

"""
global_second_derivatives(Ω::Triangulation,ϕ::AbstractArray)

Use central differences and returns ϕ_xx,ϕ_yy,ϕ_xy for inner cells, whenever Ω is constructed as CartesianDiscreteModel
"""
function global_second_derivatives(Ω::Triangulation,ϕ::AbstractArray)
    grid_partition = size(Ω.grid.cell_node_ids)
    nelem = prod(grid_partition)

    ϕxx = zeros(Float64,nelem)
    ϕxy = zeros(Float64,nelem)
    ϕyy = zeros(Float64,nelem)

    # interior points only
    for j in 2:grid_partition[1]-1
        for i in 2:grid_partition[2]-1
            ij = ij2idx(i,j,grid_partition)
            i1j = ij2idx(i+1,j,grid_partition)
            ij1 = ij2idx(i,j+1,grid_partition)
            i1j1 = ij2idx(i+1,j+1,grid_partition)
            i_1j = ij2idx(i-1,j,grid_partition)
            ij_1 = ij2idx(i,j-1,grid_partition)
            i_1j_1 = ij2idx(i-1,j-1,grid_partition)
            i1j_1 = ij2idx(i+1,j-1,grid_partition)
            i_1j1 = ij2idx(i-1,j+1,grid_partition)

            ϕxx[ij] = ϕ[i1j] - 2*ϕ[ij] + ϕ[i_1j]
            ϕyy[ij] = ϕ[ij1] - 2*ϕ[ij] + ϕ[ij_1]
            ϕxy[ij] = 0.25*(ϕ[i1j1] - ϕ[i_1j1] - ϕ[i1j_1] + ϕ[i_1j_1])
        end
    end
    
    return ϕxx,ϕyy,ϕxy
end


"""
global_curvature(Ω::Triangulation,ϕ)

Estimation of global curvature around levelsets {x : ϕ(x)=const.}. It returns κ = div(n) for each cell in Ω.
"""
function global_curvature(Ω::Triangulation,ϕ::AbstractArray;add_limiter::Bool=false)
    #* Remark: Δx and Δy cancels out here
    ϕx,ϕy = global_first_derivatives(Ω,ϕ)
    ϕxx,ϕyy,ϕxy = global_second_derivatives(Ω,ϕ)
    
    den_κ = @. √(ϕx^2 + ϕy^2) + 1e-5 # avoids division by
    κ = @. (ϕxx*ϕy^2 - 2.0*ϕx*ϕy*ϕxy + ϕyy*ϕx^2)/den_κ^3

    if add_limiter
        # set κ between ±100
        κ = max.(-100,min.(100,κ))
    end

    return κ
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
function upwind2d_step(Ω::Triangulation,xc,ϕ::AbstractArray,V::AbstractArray;curvature_penalty::Real=0)

    # Recover mesh connectivity
    nelem = num_cells(Ω)
   
    if nelem != length(ϕ)
        error("Vector ϕ has different length (",length(ϕ),") to number of cells in the mesh (",nelem,")")
    end

    dim = num_cell_dims(Ω)
    cell_to_faces  = get_faces(Ω.model.grid_topology,dim,dim-1)
    face_to_cells  = get_faces(Ω.model.grid_topology,dim-1,dim)

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

    if curvature_penalty>0
        #= We got an extra term, such that
        ϕ_new = ϕ_old - Δt⋅g + Δt⋅penalty⋅κ = ϕ_old - Δt⋅(g - penalty⋅κ)
        =#
        κ = global_curvature(Ω,ϕ,add_limiter=true)
        for e in 1:nelem
            g[e] -=  curvature_penalty * κ[e]
        end
    end

    return g
end



"""
solves some time steps for

    ∂ϕ/∂t + sign(ϕ₀)(|∇ϕ|-1) = 0     in the Grid 

                      ϕ(t=0) = ϕ₀    
"""
function ReinitHJ2d_update(Ω::Triangulation,xc,ϕ::AbstractArray,niter::Int;vmax::AbstractFloat=1.0,scheme="RK2",show_errormsg::Bool=false,Δt::Real=0.1)
    # Euler & RK2 copyed from 
    # "*On reinitilizing level set functions*", C. Min, J Comput.Phys 299:8 (2010) DOI:10.1016/j.jcp.2009.12.032

    #TODO: Check if we are on QUADs (and structured!?)

    # Recover mesh connectivity
    nelem = num_cells(Ω)
   
    if nelem != length(ϕ)
        error("Vector ϕ has a different length (",length(ϕ),") compared to the number of cells in the mesh (",nelem,")")
    end

    dim = num_cell_dims(Ω)
    cell_to_faces  = get_faces(Ω.model.grid_topology,dim,dim-1)
    face_to_cells  = get_faces(Ω.model.grid_topology,dim-1,dim)

    if dim !=2
        error("Sorry: 3D is not implemented yet in `ls_advection.jl`")
    end

    #TODO: Choose the signum function: TEST required!
    #signψ₀ = sign.(ϕ)
    #signψ₀ = Signum(ϕ)
    signψ₀ = SigNum.(ϕ,ϵ=0.5) # 0.1
       

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

                #=
                if signψ₀[e]<0
                    δ[e] = √(max(min(ϕxM,0)^2,max(ϕxm,0)^2) + max(min(ϕyM,0)^2,max(ϕym,0)^2)) # g⁻ 
                else
                    δ[e] = √(max(max(ϕxM,0)^2,min(ϕxm,0)^2) + max(max(ϕyM,0)^2,min(ϕym,0)^2)) # g⁺ 
                end
                δ[e] = signψ₀[e]*(δ[e] - 1.0) #- ν*Δϕ  # when applies the smoothed signum
                =#

                g⁺ = √(max(min(ϕxM,0)^2,max(ϕxm,0)^2) + max(min(ϕyM,0)^2,max(ϕym,0)^2))
                g⁻ = √(max(max(ϕxM,0)^2,min(ϕxm,0)^2) + max(max(ϕyM,0)^2,min(ϕym,0)^2))

                V⁺ = max(signψ₀[e],0)  # upwind with V=signψ₀ magnitude
                V⁻ = min(signψ₀[e],0)

                δ[e] = V⁺ * (g⁺ - 1.0) + V⁻ * (g⁻ - 1.0)
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
            ϕold = ϕ1
            
            for e in 1:nelem
                # Recovering neighbourhood around element 'e'
                S,N,W,E = face_to_cells[cell_to_faces[e]] # should be 4 in 2D!
                ϕxM,ϕxm,ϕyM,ϕym = local_FD_derivatives(ϕ,S,N,W,E,xc)

                sψ = signψ₀[e]
                ϕ1[e] -= Δt*sψ*(Hg(sψ,ϕxM,ϕxm,ϕyM,ϕym) - 1.0) #+ ν*Δϕ
            end

            if show_errormsg
                error_ = norm(ϕ1 - ϕold)/norm(ϕold)
                println("error[$(iter)] = ",error_)
            end
        end
        ϕ = ϕ1

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

                ϕ2[e] = ϕ1[e] - Δt*signψ₀[e]*( Hg(signψ₀[e],ϕxM,ϕxm,ϕyM,ϕym) - 1.0)
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


## Other functions that applies on levelsets


"""
Extended regularization

Solves   ∫ ϵ²∇V⋅∇ϕ + Vϕ dΩ = -∫ χ_{ϕ=0} j(u)ϕ dΩ
"""
function extended_regularization(Ω::Triangulation,phi::Vector{Float64},j::Function;eps::Real=1e-3,thres::Real=0.75)
    # ∫ ϵ^2*∇V⋅∇ϕ + Vϕ dΩ = -∫ χ_{ϕ=0} j(u)ϕ dΩ
    order = 1
    reffe = ReferenceFE(lagrangian,Float64,order)
    V = TestFESpace(model,reffe, conformity=:H1) #  dirichlet_masks=[(true,false), (true,true)])
    U  = TrialFESpace(V,[])
    degree = 2*order
    dΩ = Measure(Ω,degree)
    #Γ  = BoundaryTriangulation(Ω.model,tags="boundary")
    #dΓ = Measure(Γ,degree)
    delta_reg = @. (exp(-phi^2) > thres)
    dΣ = DiracDelta(Ω.model,delta_reg)
    ϵ² = eps^2

    a(v,ϕ) = ∫(ϵ²*∇(v)⋅∇(ϕ) + v*ϕ)dΩ
    l(ϕ) = ∫(-j*ϕ)dΣ

    op = AffineFEOperator(a,l,U,V)
    vh = solve(op)

    return vh
end


function velocity_regularization(Ω::Triangulation,phi,velo;thres::Real=0.75)
  #TODO: phi could be a ::Vector{Float64} or ::LazyArray !!
    #TODO: typeof(velo) = Gridap.CellData.OperationCellField{ReferenceDomain} is not CellField nor FEFunction
    # ∫ ∇V⋅∇ϕ dΩ = ∫ χ_{ϕ=0} velo ϕ dΩ
    reffe = ReferenceFE(lagrangian,Float64,1)
    V = TestFESpace(model,reffe, conformity=:H1)
    U  = TrialFESpace(V)
    dΩ = Measure(Ω,2)
    #Γ  = BoundaryTriangulation(Ω.model,tags="boundary")
    #dΓ = Measure(Γ,2)
    delta_reg = @. convert(Float64,exp(-phi^2) >= thres)
    #dΣ = DiracDelta(Ω.model,delta_reg)

    a(v,ϕ) = ∫(∇(v)⋅∇(ϕ))dΩ + ∫((1e12.*delta_reg)*v*ϕ)dΩ
    #l(ϕ) = ∫(velo*ϕ)dΣ

    rhs = (1e12.*delta_reg)*velo 

    l(ϕ) = ∫(rhs*ϕ)dΩ

    op = AffineFEOperator(a,l,U,V)
    vh = solve(op)

    # vh is projected onto element-wise grid
    V0 = FESpace(model,ReferenceFE(lagrangian,Float64,0))
    vh0 = interpolate_everywhere(vh,V0)

    return get_free_dof_values(vh0) #,get_free_dof_values(vh)
end



function secant_lag_ctrl(λ_,vol,vol_,k;Vmax=10,Nvol = 10)
    if k==1
        λ = 10
    elseif k==2
        λ = 0
    else
        Vini = 10
        #Vmax_k = Vmax + (Vini - Vmax)*max(0,1-k/Nvol)

        @show k
        @show λ_
        @show vol_
        λ = λ_[k-1] - (λ_[k-1] - λ_[k-2])/(Vini - Vmax)*Nvol * (vol - vol_[k-2]) * max(0,1-(k-1)/Nvol)
    end
    return λ
end


#end # of module
