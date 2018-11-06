#!/usr/bin/env python
# c) 2001-2017 Simula Research Laboratory ALL RIGHTS RESERVED
# Authors: Henrik Finsberg
# END-USER LICENSE AGREEMENT
# PLEASE READ THIS DOCUMENT CAREFULLY. By installing or using this
# software you agree with the terms and conditions of this license
# agreement. If you do not accept the terms of this license agreement
# you may not install or use this software.

# Permission to use, copy, modify and distribute any part of this
# software for non-profit educational and research purposes, without
# fee, and without a written agreement is hereby granted, provided
# that the above copyright notice, and this license agreement in its
# entirety appear in all copies. Those desiring to use this software
# for commercial purposes should contact Simula Research Laboratory AS: post@simula.no
#
# IN NO EVENT SHALL SIMULA RESEARCH LABORATORY BE LIABLE TO ANY PARTY
# FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES,
# INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE
# "PULSE-ADJOINT" EVEN IF SIMULA RESEARCH LABORATORY HAS BEEN ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE. THE SOFTWARE PROVIDED HEREIN IS
# ON AN "AS IS" BASIS, AND SIMULA RESEARCH LABORATORY HAS NO OBLIGATION
# TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
# SIMULA RESEARCH LABORATORY MAKES NO REPRESENTATIONS AND EXTENDS NO
# WARRANTIES OF ANY KIND, EITHER IMPLIED OR EXPRESSED, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY OR FITNESS
import shutil
from .args import *


def getColorCorrespondingTovalue(val, min_val, max_val, hue = "blue_white_red"):

    if hue == "blue_white_red":
        numColorNodes = 3
        color = np.array([[0.0, 0.0, 1.0],    # blue
                          [1.0, 1.0, 1.0],    # white
                          [1.0, 0.0, 0.0]]).T     # red

    elif hue == "blue_red":
        numColorNodes = 2
        color = np.array([[0.0, 0.0, 1.0],    # blue
                          [1.0, 0.0, 0.0]]).T     # red

    elif hue == "rainbow":
        numColorNodes = 3
        color = np.array([[0.0, 0.0, 1.0],    # blue
                          [0.0, 1.0, 0.0],    # green
                          [1.0, 0.0, 0.0]]).T     # red
        
    
    

    for i in range(numColorNodes-1):
        currFloor = min_val + (i / float(numColorNodes - 1)) * (max_val-min_val)
        currCeil = min_val + ((i + 1) / float(numColorNodes - 1)) * (max_val-min_val)
        

        if ((val >= currFloor) and (val <= currCeil)):
            
            currFraction = (val - currFloor) / float(currCeil - currFloor)
       
            r = color[0][i] * (1.0 - currFraction) + color[0][i + 1] * currFraction
            g = color[1][i] * (1.0 - currFraction) + color[1][i + 1] * currFraction
            b = color[2][i] * (1.0 - currFraction) + color[2][i + 1] * currFraction
            
    return r,g,b

def make_video(imgpath, keys, moviepath):
    
    # Make sure that the images exists
    for k in keys:
        if not os.path.exists(imgpath % int(k)):
            if os.path.exists(os.path.splitext(imgpath % int(k))[0]):
                shutil.move(os.path.splitext(imgpath % int(k))[0],
                            imgpath % int(k))
            else:
                raise IOError("File {} does not exist".format(imgpath & int(k)))

    os.system("ffmpeg -y -loglevel panic -framerate 12 -i {0}  -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p {1}".format(imgpath, moviepath))
    
def snap_shot(fun, path, rnge = (0,0.1), clip = False, min_val = 0.0, max_val = 0.4, hue = "blue_white_red", 
              colorbar = True, colorbar_align = "vertical", colorbar_only = False, title = "$\gamma$"):
    """Plot snap shot using vtk
    
    Args:
      fun (:py:class`dolfin.Function`)
        The function you would like to plot
      path (string)
        Path to where to save the figure
      name (string)
        Name of the object you want to plot
      wild (string)
        Additional information about the object (optional)

    """
    import vtk
    from dolfinplot import VTK_Dolfin, VTK_DolfinScalar
    if colorbar_only:
        vtkfun = VTK_Dolfin(rnge)
        height = 0.17
        width = 0.8
    else:
        vtkfun = VTK_DolfinScalar(fun)
        height = 0.8
        width = 0.17
        
    legend = vtk.vtkScalarBarActor()
    lookupTable = vtk.vtkLookupTable()

    lookupTable.SetScaleToLinear();
    lookupTable.SetRange(min_val,max_val);

    numColors = 100
    lookupTable.SetNumberOfTableValues(numColors);

    for i in range(numColors):
        val = min_val + (i / float(numColors)) * (max_val-min_val)
        r, g, b = getColorCorrespondingTovalue(val, min_val, max_val, hue)
        lookupTable.SetTableValue(i, r, g, b)


    lookupTable.Build()
    
    vtkfun.mapper.SetScalarRange((min_val, max_val))
    vtkfun.mapper.SetLookupTable(lookupTable)
    vtkfun.SetRepresentation("surface_w_edges")

    if clip:
        vtkfun.Clip(origin=(4.5, 0.0, 0.0), normal = (0,-1,1), overlay = False)
        vtkfun.clipmapper.SetScalarRange((min_val, max_val))
        vtkfun.clipmapper.SetLookupTable(lookupTable)
        
    # scalarbar = vtk.vtkScalarBarActor()
    # scalarbar.SetLookupTable(map_mesh.GetLookupTable())
    # textprop = vtk.vtkTextProperty()
    # textprop.SetColor((0,0,0))
    # ren.AddActor2D(scalarbar)

    if colorbar:
        legend.SetLookupTable(lookupTable)
        legend.SetTitle("f")
        if colorbar_align == "vertical":
            legend.SetOrientationToHorizontal()
            legend.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
            legend.GetPositionCoordinate().SetValue(0.714, 0.84)
            legend.SetWidth(0.2515)
            legend.SetHeight(0.1)

        else:
            pass
            
        legend.GetTitleTextProperty().ItalicOff()
        legend.GetTitleTextProperty().SetColor(0,0,0)
        legend.GetTitleTextProperty().SetBold(False)
        legend.GetTitleTextProperty().SetFontFamilyAsString("Sans Serif")
        legend.GetLabelTextProperty().ItalicOff()
        legend.GetLabelTextProperty().SetColor(0,0,0)
        legend.GetLabelTextProperty().SetBold(False)
        legend.GetLabelTextProperty().SetFontFamilyAsString("Sans Serif")

    

    # ren.SetBackground(1,1,1)

    
    # if colorbar:
    #     vtkfun.SetColorBar(title = " ".join([name, wild]), width = width, height=height,
    #                        orientation = colorbar_align,label_fmt = '%.1f')
    # vtkfun.SetEdgeColor((0.6,0, 0))

    # Plot side
    vtkfun.camera.SetPosition(-18.0, 0.0, 0.0)
    vtkfun.camera.SetFocalPoint(4.5, 0.0, 0.0)
    vtkfun.camera.SetViewUp(0.0160, -0.68, 0.7242)
    vtkfun.Render(view="side", size = (1200, 800))

    if colorbar:
        vtkfun.renderer.AddActor2D(legend)
    vtkfun.Save(fname = path)
    shutil.move(path, path + ".png")

    # if not colorbar_only:
    #     # Plot front
    #     vtkfun.camera.SetPosition(-6.0, 15.0, -15.0)
    #     vtkfun.camera.SetFocalPoint(4.5, 0, 0)
    #     vtkfun.camera.SetViewUp(-0.8856, 0.3162, -0.3400)
    #     
    #     vtkfun.clipmapper.SetScalarRange(rnge)
    #     vtkfun.clipmapper.SetLookupTable(vtkfun.colorLookupTable)
    #     vtkfun.Render(view="custom", size = (1200, 800))
    #     vtkfun.Save(fname = path + "_side")
    #     shutil.move(path + "_side", path + "_side.png")

def make_snapshots(fs, us, spacestr, outdir, params):

    if not os.path.exists(outdir):
        os.makedirs(outdir)
            
    from ..patient_data import FullPatient
    from .utils import asint
    
    patient = FullPatient(**params["Patient_parameters"])
    # Mesh
    mesh = patient.mesh
    # Mesh that we move
    moving_mesh = dolfin.Mesh(mesh)

    family, degree = spacestr.split("_")
    space = dolfin.FunctionSpace(moving_mesh, family, int(degree))
    f = dolfin.Function(space)
    disp_space = dolfin.VectorFunctionSpace(mesh, "CG", 2)
    u = dolfin.Function(disp_space)
    u_prev = dolfin.Function(disp_space)
    u_diff = dolfin.Function(disp_space)

    V = dolfin.VectorFunctionSpace(mesh, "CG", 1)

    sm = dolfin.Function(dolfin.FunctionSpace(moving_mesh, "DG", 0), name = "AHA-zone")
    sm.vector()[:] = patient.strain_markers.array()

    
    times = sorted(us.keys(), key=asint)
    path = "/".join([outdir, "time_{}"])
    path_sm = "/".join([outdir, "time_aha_{}"])

    for i,t in enumerate(times):

        u.vector()[:] = us[t]
        u_diff.vector()[:] = u.vector() - u_prev.vector()
        d = dolfin.interpolate(u_diff, V)
        dolfin.ALE.move(moving_mesh, d)

        f.vector()[:] = fs[t]
        
        snap_shot(f, path.format(t), min_val = -0.25, max_val = 0.25, colorbar = True, colorbar_align = "vertical", title ="f")
        

        snap_shot(sm, path_sm.format(t), min_val = 1, max_val = 17, colorbar = True, colorbar_align = "horizontal", title ="AHA", hue = "rainbow")
        
        u_prev.assign(u)
        
def ply_to_polydata(fname):
    import vtk
    reader = vtk.vtkPLYReader()
    reader.SetFileName(fname)
    reader.Update()
    polydata = reader.GetOutput()
    return polydata

def write_to_vtp(fname, polydata):
    import vtk
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(fname+".vtp")

    if vtk.VTK_MAJOR_VERSION <= 5:
        writer.SetInput(polydata)
    else:
        writer.SetInputData(polydata)
    writer.Write()

def vtk_add_field(grid, fun):
    
    V = fun.function_space()
    family = V.ufl_element().family()
    degree = V.ufl_element().degree()

    if fun.value_rank() > 0 :
        idx = np.column_stack([ V.sub(i).dofmap().dofs()
                    for i in xrange(0, V.num_sub_spaces()) ])
        fval = fun.vector().array()[idx]
    else :

        if family in ['Discontinuous Lagrange'] :
            fval = fun.vector().array()

        elif family in ['Real']:
            fval = fun.vector().array()[0]*np.ones(int(grid.GetNumberOfPoints()))

        elif family in ["Quadrature"]:
            
            # Take the average over the quadrature points within
            # a given cell. Visualize as pointwise cell averages
            fval_cell = np.zeros((V.mesh().num_cells(),))
            for c in dolfin.cells(V.mesh()) :
                idx = c.index()
                fval_cell[idx] = fun.vector()[V.dofmap().cell_dofs(idx)].mean()

            # Visualize at the vertices by reducing the value at the
            # nearby quadrature points to one value at the vertex

            # Compute connvectivity between dimension 1 and 3
            V.mesh().init(1,3)
            fval_vert = np.zeros((V.mesh().num_vertices(),))
            for v in dolfin.vertices(V.mesh()) :
                idx = v.index()
                # Find the cells who have v as a vertex
                c = V.mesh().topology()(1,3)(idx)
                # Find the dofs for the quadrature points within these cells
                dofs = reduce(np.union1d, [V.dofmap().cell_dofs(ci) for ci in c])
                # Take the average of the value at these dofs
                fval_vert[idx] = fun.vector()[dofs].mean()

            
            
       
        else:
            vtd = dolfin.vertex_to_dof_map(V)
            fval_tmp = fun.vector().array()
            fval = np.zeros(len(fval_tmp))
            fval = fval_tmp[vtd]
           

    if fun.name() == 'displacement' :
        # add zero columns if necessary
        gdim = V.num_sub_spaces()
        fval = np.hstack([fval, np.zeros((fval.shape[0], 3-gdim))])

    from tvtk.array_handler import array2vtk
    if family in ['Discontinuous Lagrange']:
        funvtk = array2vtk(fval)
        funvtk.SetName(fun.name())
        grid.GetCellData().AddArray(funvtk)
        
    elif family in ["Quadrature"]:
        funvtk_cell = array2vtk(fval_cell)
        funvtk_cell.SetName(fun.name())
        grid.GetCellData().AddArray(funvtk_cell)

        funvtk_vert = array2vtk(fval_vert)
        funvtk_vert.SetName(fun.name())
        grid.GetPointData().AddArray(funvtk_vert)
        
        
    else :
        funvtk = array2vtk(fval)
        funvtk.SetName(fun.name())
        grid.GetPointData().AddArray(funvtk)



def dolfin2vtu(mesh):
    """Convert dolfin mesh to vtk unstructured grid

    :param mesh: dolfin mesh
    :returns: mesh in vtk unstructured grid format
    :rtype: vtk.vtkUnstructuredGrid

    """
    import vtk
    domain = mesh.ufl_domain()
    gdim = domain.geometric_dimension()
    mdim = domain.topological_dimension()
    order = 1
    # coordinates of the mesh
    coords = mesh.coordinates().copy()

   
    # connectivity
    conn = mesh.cells()

    coords = np.hstack([coords, np.zeros((coords.shape[0], 3-gdim))])

    # only these are supported by dolfin
    vtk_shape = { 1 : { 1 : vtk.VTK_LINE,
                        2 : vtk.VTK_TRIANGLE,
                        3 : vtk.VTK_TETRA },
                  2 : { 1 : vtk.VTK_QUADRATIC_EDGE,
                        2 : vtk.VTK_QUADRATIC_TRIANGLE,
                        3 : vtk.VTK_QUADRATIC_TETRA } }[order][mdim]

   
    # create the grid
    from tvtk.array_handler import array2vtkPoints, array2vtkCellArray
    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(array2vtkPoints(coords))
    grid.SetCells(vtk_shape, array2vtkCellArray(conn))
    return grid

def dolfin2polydata(mesh):
    """Convert dolfin mesh to vtk polydata

    :param mesh: dolfin mesh
    :returns: mesh in vtk polydata format
    :rtype: vtk.vtkPolyData

    """
    import vtk
    from tvtk.array_handler import array2vtkPoints, array2vtkCellArray
    
    gdim = mesh.geometry().dim()
    mdim = mesh.topology().dim()

    order = 1
    # coordinates
    pts = array2vtkPoints(mesh.coordinates())
    # connectivity
    dim = mesh.topology().dim()
    ndof = dolfin.Cell(mesh, 0).num_vertices()
    conn = mesh.topology()(dim, 0)().reshape((-1, ndof))
    elms = array2vtkCellArray(conn)

    grid = vtk.vtkPolyData()
    grid.SetPoints(pts)

    # only these are supported by dolfin
    vtk_shape = { 1 : { 1 : vtk.VTK_LINE, 2 : vtk.VTK_TRIANGLE, 3 : vtk.VTK_TETRA },
                  2 : { 1 : vtk.VTK_QUADRATIC_EDGE,
                        2 : vtk.VTK_QUADRATIC_TRIANGLE,
                        3 : vtk.VTK_QUADRATIC_TETRA } }[1][mdim]

    grid.SetLines(elms)

    return grid
def polyDataToActor(polydata, layer):
    """Wrap the provided vtkPolyData object in a 
    mapper and an actor, returning the actor.
    """
    import vtk

    if layer == "mesh":
        mapper = vtk.vtkPolyDataMapper()
    else:
        mapper = vtk.vtkDataSetMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
        
        mapper.SetInput(polydata)
    else:
        mapper.SetInputData(polydata)

        if layer == "mesh":
            mapper.SetScalarModeToUsePointData()
 
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        if layer == "mesh":
            actor.GetProperty().SetLineWidth(3.0)
      
        else:
            actor.GetProperty().SetColor((0,1,0))
            actor.GetProperty().SetOpacity(0.4)
            actor.GetProperty().EdgeVisibilityOff()
            
    return actor, mapper

def write_pvd(pvd_name, fname, time_stamps):
    

    time_form = """<DataSet timestep="{}" part="0" file="{}" />"""

    body="""<?xml version="1.0"?>
    <VTKFile type="Collection" version="0.1">
    <Collection>
    {}
    </Collection>
    </VTKFile>
    """.format(" ".join(time_form.format(time_stamps[i],fname.format(i)) for i in range(len(time_stamps))))

    with open(pvd_name, "w") as f:
        f.write(body)

def write_to_vtk(grid, name):
    import vtk
    writer = vtk.vtkXMLUnstructuredGridWriter()

    if vtk.VTK_VERSION >= 6.0:
        writer.SetInputData(grid)
    else:
        writer.SetInput(grid)
        
    writer.SetFileName(name)
    writer.Write()

def add_stuff(mesh, name, *args):
    grid = dolfin2vtu(mesh)

    for f in args:
        vtk_add_field(grid, f)

    write_to_vtk(grid, name)

def get_transformation_matrix(patient, time):

    from mesh_generation.mesh_utils import get_round_off_buffer, load_echo_geometry
    from mesh_generation.surface import get_geometric_matrix
    
    
    round_off = get_round_off_buffer(patient.name(), time)
    echo_surfaces = load_echo_geometry(patient.parameters["echo_path"],
                                       time)
    endo_verts_orig = np.ones((echo_surfaces["endo_verts"].shape[0],4))
    endo_verts_orig[:,:3] = np.copy(echo_surfaces["endo_verts"])
    # echo_surfaces = patient.get_original_echo_surfaces()
    
    T = get_geometric_matrix(echo_surfaces["strain_mesh"],
                             endo_verts_orig,
                             round_off, second_layer = False)
    return T

def save_surface_to_dolfinxml(patient, time, surfdir = "surface_files"):
    
    echo_surfaces = patient.get_original_echo_surfaces()

    # T = patient.transformation_matrix
    
    T = get_transformation_matrix(patient, patient.passive_filling_begins)
    
    endo = echo_surfaces["endo"]
    endo_faces = np.array(endo['indices'])
    endo_verts_orig = 100*np.array(endo['vertices'])[time,:,:]
    endo_verts_tmp = np.ones((endo_verts_orig.shape[0],4))
    endo_verts_tmp[:,:3] = np.copy(endo_verts_orig)
    endo_verts = T.dot(endo_verts_tmp.T).T[:,:3]

    editor = dolfin.MeshEditor()
    mesh = dolfin.Mesh()
    editor.open(mesh, 2, 3)
    editor.init_vertices(len(endo_verts))
    editor.init_cells(len(endo_faces))

    for i,v in enumerate(endo_verts):
        editor.add_vertex(i,v)

    for j, c in enumerate(endo_faces):
        editor.add_cell(j,np.array(c, dtype=np.uintp))

    editor.close()
    # plot(mesh, interactive = True)

        
    endoname = "/".join([surfdir, "echo_endo_{}.xml".format(time)])
    f = dolfin.File(endoname)
    f << mesh
    return endoname

def write_to_polydata(fname, grid):
    import vtk
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(fname)
    writer.SetInputData(grid)
    writer.Write()

def get_submesh(mesh, endo_marker):
    
    bmesh = dolfin.BoundaryMesh(mesh, "exterior")
    ffun = dolfin.MeshFunction("size_t", mesh, 2, mesh.domains())
    mapping = bmesh.entity_map(2)
    
    for cell in dolfin.cells(bmesh):
        bmesh.domains().set_marker((cell.index(), ffun[mapping[cell.index()]]), 2)

    ffun2 = dolfin.MeshFunction("size_t", bmesh, 2, bmesh.domains())  
    endo_submesh = dolfin.SubMesh(bmesh, ffun2, endo_marker)
    return endo_submesh
