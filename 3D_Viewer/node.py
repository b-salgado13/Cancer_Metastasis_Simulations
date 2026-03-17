import random
import numpy
from OpenGL.GL import *
from OpenGL.GLU import *
import math


# ---------------------------------------------------------------------------
# VBO mesh helper
# ---------------------------------------------------------------------------

class _VBOMesh:
    """
    Owns three GPU buffers (vertex positions, normals, triangle indices) for a
    single primitive shape.  Replaces OpenGL display lists, which are removed
    in core-profile contexts and carry a per-call CPU overhead.

    draw() binds the buffers and issues a single glDrawElements call — one GPU
    round-trip regardless of vertex count, vs. one glCallList that still
    replays every individual GL command internally.
    """

    def __init__(self, vertices, normals, indices):
        """
        Parameters
        ----------
        vertices : numpy.ndarray, dtype float32, shape (N*3,)
            Interleaved x,y,z positions.
        normals  : numpy.ndarray, dtype float32, shape (N*3,)
            Per-vertex normals (unit length), same ordering as vertices.
        indices  : numpy.ndarray, dtype uint32, shape (M,)
            Triangle list: every three consecutive values form one triangle.
        """
        self.index_count = len(indices)

        # Allocate three separate buffer objects (positions, normals, indices).
        # Calling glGenBuffers once per buffer avoids PyOpenGL version quirks
        # where glGenBuffers(3) may return a scalar instead of a sequence.
        self.vbo_v = int(glGenBuffers(1))
        self.vbo_n = int(glGenBuffers(1))
        self.vbo_i = int(glGenBuffers(1))

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_v)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_n)
        glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_STATIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vbo_i)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        # Leave no buffer bound after setup.
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

    def draw(self):
        """Issue one indexed draw call using the stored VBOs."""
        # GL_NORMALIZE tells the fixed-function pipeline to renormalise every
        # normal after it has been multiplied by the inverse-transpose of the
        # current modelview matrix.  This is necessary because Node.render()
        # pushes a scaling_matrix onto the stack: a uniform scale s transforms
        # unit normals to length 1/s (inverse-transpose rule), which shifts the
        # diffuse dot-product away from its correct value.  Bump spheres scaled
        # to 0.3 would otherwise have normals of length ~3.3, clamping the
        # diffuse term to 1 and producing unnaturally flat, over-bright patches.
        glEnable(GL_NORMALIZE)

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_v)
        glVertexPointer(3, GL_FLOAT, 0, None)

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_n)
        glNormalPointer(GL_FLOAT, 0, None)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vbo_i)
        glDrawElements(GL_TRIANGLES, self.index_count, GL_UNSIGNED_INT, None)

        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

        glDisable(GL_NORMALIZE)

    def delete(self):
        """Free GPU memory (call when the OpenGL context is still current)."""
        glDeleteBuffers(1, [self.vbo_v])
        glDeleteBuffers(1, [self.vbo_n])
        glDeleteBuffers(1, [self.vbo_i])


# ---------------------------------------------------------------------------
# Procedural geometry generators
# ---------------------------------------------------------------------------

def _make_sphere_data(stacks=30, slices=30, radius=0.5):
    """
    Build a UV-sphere matching the original gluSphere(radius=0.5, 30, 30) call.

    Returns (vertices, normals, indices) as float32/uint32 numpy arrays.
    The normal at each vertex equals the unit-length position vector, which is
    correct for a sphere centred at the origin.
    """
    verts = []
    norms = []
    idxs  = []

    for i in range(stacks + 1):
        phi     = math.pi * i / stacks          # 0 (north pole) → π (south pole)
        sin_phi = math.sin(phi)
        cos_phi = math.cos(phi)
        for j in range(slices + 1):
            theta = 2.0 * math.pi * j / slices  # 0 → 2π around equator
            nx = sin_phi * math.cos(theta)
            ny = cos_phi
            nz = sin_phi * math.sin(theta)
            verts += [nx * radius, ny * radius, nz * radius]
            norms += [nx, ny, nz]

    # Two triangles per quad cell in the (stacks × slices) grid.
    #
    # Vertex layout for one quad (viewed from outside the sphere):
    #   v0 ------ v1       row i   (north)
    #   |        / |
    #   |      /   |
    #   v2 ------ v3       row i+1 (south)
    #
    # Counter-clockwise winding when seen from outside requires:
    #   triangle A : v0, v1, v3   (top-left → top-right → bottom-right)
    #   triangle B : v0, v3, v2   (top-left → bottom-right → bottom-left)
    #
    # The previous order [v0,v2,v1 / v1,v2,v3] was clockwise from outside,
    # so GL_CULL_FACE(GL_BACK) silently discarded the front hemisphere and
    # the interior normals (pointing away from the light) produced flat/dark
    # shading on the visible back hemisphere.
    for i in range(stacks):
        for j in range(slices):
            v0 = i * (slices + 1) + j
            v1 = v0 + 1
            v2 = v0 + (slices + 1)
            v3 = v2 + 1
            idxs += [v0, v1, v3,
                     v0, v3, v2]

    return (numpy.array(verts, dtype=numpy.float32),
            numpy.array(norms, dtype=numpy.float32),
            numpy.array(idxs,  dtype=numpy.uint32))


def _make_cube_data():
    """
    Build a unit cube (±0.5 on each axis) with per-face flat normals.

    24 unique vertices (4 per face) allow each face to have a distinct normal,
    which produces correct flat shading under OpenGL's fixed-function lighting.
    36 indices (2 triangles × 3 vertices × 6 faces).

    Returns (vertices, normals, indices) as float32/uint32 numpy arrays.
    """
    # Each tuple: (outward normal,  4 CCW vertices as seen from outside)
    face_defs = [
        (( 0,  0,  1), [(-0.5,-0.5, 0.5),( 0.5,-0.5, 0.5),( 0.5, 0.5, 0.5),(-0.5, 0.5, 0.5)]),
        (( 0,  0, -1), [(-0.5,-0.5,-0.5),(-0.5, 0.5,-0.5),( 0.5, 0.5,-0.5),( 0.5,-0.5,-0.5)]),
        (( 0,  1,  0), [(-0.5, 0.5,-0.5),(-0.5, 0.5, 0.5),( 0.5, 0.5, 0.5),( 0.5, 0.5,-0.5)]),
        (( 0, -1,  0), [(-0.5,-0.5,-0.5),( 0.5,-0.5,-0.5),( 0.5,-0.5, 0.5),(-0.5,-0.5, 0.5)]),
        (( 1,  0,  0), [( 0.5,-0.5,-0.5),( 0.5, 0.5,-0.5),( 0.5, 0.5, 0.5),( 0.5,-0.5, 0.5)]),
        ((-1,  0,  0), [(-0.5,-0.5,-0.5),(-0.5,-0.5, 0.5),(-0.5, 0.5, 0.5),(-0.5, 0.5,-0.5)]),
    ]
    verts = []
    norms = []
    idxs  = []
    base  = 0
    for normal, corners in face_defs:
        for v in corners:
            verts += list(v)
            norms += list(normal)
        # Split the quad into two CCW triangles.
        idxs += [base, base+1, base+2,
                 base, base+2, base+3]
        base += 4

    return (numpy.array(verts, dtype=numpy.float32),
            numpy.array(norms, dtype=numpy.float32),
            numpy.array(idxs,  dtype=numpy.uint32))


# ---------------------------------------------------------------------------
# Module-level primitive singletons  (populated by init_primitives)
# ---------------------------------------------------------------------------

_SPHERE_MESH = None   # _VBOMesh for the sphere
_CUBE_MESH   = None   # _VBOMesh for the cube
G_OBJ_PLANE  = None   # Display list for the reference grid (used by viewer.py)


def init_primitives():
    """
    Upload sphere and cube geometry to the GPU as VBOs, and compile the
    reference grid as a display list.

    Must be called after the OpenGL context has been created (i.e. after
    glutCreateWindow) but before any rendering or node construction.
    """
    global _SPHERE_MESH, _CUBE_MESH, G_OBJ_PLANE

    # --- Sphere VBO ---
    _SPHERE_MESH = _VBOMesh(*_make_sphere_data(stacks=30, slices=30, radius=0.5))

    # --- Cube VBO ---
    _CUBE_MESH = _VBOMesh(*_make_cube_data())

    # --- Reference grid (display list kept for backward compatibility with
    #     the glCallList(node.G_OBJ_PLANE) call in viewer.py) ---
    G_OBJ_PLANE = glGenLists(1)
    glNewList(G_OBJ_PLANE, GL_COMPILE)
    glBegin(GL_LINES)
    glColor3f(0.8, 0.8, 0.8)
    for i in range(-20, 21):
        glVertex3f(i,   0, -20)
        glVertex3f(i,   0,  20)
        glVertex3f(-20, 0,   i)
        glVertex3f( 20, 0,   i)
    glEnd()
    glEndList()

class Color(object):
    COLORS = [
        (1.0, 0.0, 0.0), # Red
        (0.0, 1.0, 0.0), # Green
        (0.0, 0.0, 1.0), # Blue
        (1.0, 1.0, 0.0), # Yellow
        (1.0, 0.0, 1.0), # Magenta
        (0.0, 1.0, 1.0), # Cyan
        (1.0, 1.0, 1.0), # White
        (0.5, 0.5, 0.5), # Grey
    ]
    MIN_COLOR = 0
    MAX_COLOR = len(COLORS) - 1

class AABB(object):
    def __init__(self, center, size):
        self.center = numpy.array(center, dtype=float)
        self.size = numpy.array(size, dtype=float)

    def scale(self, scale_factor):
        self.size *= scale_factor

    def ray_hit(self, start, direction, modelview):
        """
        Check if ray intersects AABB.
        Ray is defined by start and direction (in Eye Space).
        modelview transforms from Local Space (AABB Space) to Eye Space.
        """
        # Transform ray to AABB space
        # Ray_Local = inv(ModelView) * Ray_Eye

        inv_mv = numpy.linalg.inv(modelview)

        # Transform start point
        # append 1 for position
        p1 = numpy.append(start, 1.0)
        p1_local = numpy.dot(inv_mv, p1)
        start_local = p1_local[:3] / p1_local[3]

        # Transform direction
        # Direction is a vector, w=0
        d1 = numpy.append(direction, 0.0)
        d1_local = numpy.dot(inv_mv, d1)
        direction_local = d1_local[:3]
        # normalize
        norm = numpy.linalg.norm(direction_local)
        if norm == 0:
            return False, 0
        direction_local /= norm

        # Now check intersection with AABB in local space
        # Slab method
        t_min = -float('inf')
        t_max = float('inf')

        box_min = self.center - self.size
        box_max = self.center + self.size

        for i in range(3):
            if direction_local[i] != 0:
                t1 = (box_min[i] - start_local[i]) / direction_local[i]
                t2 = (box_max[i] - start_local[i]) / direction_local[i]

                t_min = max(t_min, min(t1, t2))
                t_max = min(t_max, max(t1, t2))
            elif start_local[i] < box_min[i] or start_local[i] > box_max[i]:
                return False, 0

        if t_max >= t_min and t_max >= 0:
            dist = t_min if t_min > 0 else t_max
            # Return True and distance
            return True, dist

        return False, 0

class Node(object):
    """ Base class for scene elements """
    def __init__(self):
        self.color_index = random.randint(Color.MIN_COLOR, Color.MAX_COLOR)
        self.aabb = AABB([0.0, 0.0, 0.0], [0.5, 0.5, 0.5])
        self.translation_matrix = numpy.identity(4)
        self.scaling_matrix = numpy.identity(4)
        self.selected = False
        self.depth = 0 # for moving
        self.selected_loc = None # for moving
        self.color = None # Explicit RGB color

    def render(self):
        """ renders the item to the screen """
        glPushMatrix()
        glMultMatrixf(numpy.transpose(self.translation_matrix))
        glMultMatrixf(self.scaling_matrix)

        if self.color is not None:
            cur_color = self.color
        else:
            cur_color = Color.COLORS[self.color_index]

        glColor3f(cur_color[0], cur_color[1], cur_color[2])
        if self.selected:  # emit light if the node is selected
            glMaterialfv(GL_FRONT, GL_EMISSION, [0.3, 0.3, 0.3])

        self.render_self()

        if self.selected:
            glMaterialfv(GL_FRONT, GL_EMISSION, [0.0, 0.0, 0.0])
        glPopMatrix()

    def render_self(self):
        raise NotImplementedError(
            "The Abstract Node Class doesn't define 'render_self'")

    def translate(self, x, y, z):
        self.translation_matrix = numpy.dot(
            self.translation_matrix,
            translation([x, y, z]))

    def scale(self, up):
        s =  1.1 if up else 0.9
        self.scaling_matrix = numpy.dot(self.scaling_matrix, scaling([s, s, s]))
        self.aabb.scale(s)

    def rotate_color(self, forwards):
        self.color_index += 1 if forwards else -1
        if self.color_index > Color.MAX_COLOR:
            self.color_index = Color.MIN_COLOR
        if self.color_index < Color.MIN_COLOR:
            self.color_index = Color.MAX_COLOR

    def pick(self, start, direction, mat):
        """
        Return whether or not the ray hits the object

        Consume:
        start, direction form the ray to check
        mat is the modelview matrix to transform the ray by
        """

        # transform the modelview matrix by the current translation
        # newmat = MV * T
        newmat = numpy.dot(mat, self.translation_matrix)

        results = self.aabb.ray_hit(start, direction, newmat)
        return results

    def select(self, select=None):
       """ Toggles or sets selected state """
       if select is not None:
           self.selected = select
       else:
           self.selected = not self.selected

class Primitive(Node):
    def __init__(self):
        super(Primitive, self).__init__()
        self.call_list = None

    def render_self(self):
        if self.call_list is not None:
            glCallList(self.call_list)

class Sphere(Primitive):
    """
    Sphere primitive backed by a VBO mesh.

    render_self() reads the module-level _SPHERE_MESH at draw time rather than
    caching it at construction time.  This is essential because CancerCell
    creates bump Sphere instances during scene setup, before init_primitives()
    has been called and the GPU buffers exist.
    """
    def render_self(self):
        if _SPHERE_MESH is not None:
            _SPHERE_MESH.draw()


class Cube(Primitive):
    """
    Cube primitive backed by a VBO mesh.  Same deferred-lookup pattern as Sphere.
    """
    def render_self(self):
        if _CUBE_MESH is not None:
            _CUBE_MESH.draw()

class HierarchicalNode(Node):
    def __init__(self):
        super(HierarchicalNode, self).__init__()
        self.child_nodes = []

    def render_self(self):
        for child in self.child_nodes:
            child.render()

    def pick(self, start, direction, mat):
        # Pick against AABB of the group?
        # Or pick against children?
        # The article implies HierarchicalNode is just a container, but SnowFigure has an AABB.
        # "self.aabb = AABB([0.0, 0.0, 0.0], [0.5, 1.1, 0.5])"
        # So it picks against the group AABB first?
        # But for accurate picking, we might want to pick children.
        # However, the code follows the simple AABB check for the node.

        # transform mat by self.translation
        newmat = numpy.dot(mat, self.translation_matrix)

        hit, dist = self.aabb.ray_hit(start, direction, newmat)

        # If we hit the group AABB, does that select the group?
        # Yes, based on the description, SnowFigure is treated as a single object.
        return hit, dist

    def translate(self, x, y, z):
        # Override to update children? No, the base class updates translation_matrix
        # and render() applies it. Children are rendered relative to parent.
        super(HierarchicalNode, self).translate(x, y, z)

    def scale(self, up):
        super(HierarchicalNode, self).scale(up)
        # Should we scale children?
        # Base scale() updates scaling_matrix which applies to all children during render.
        # But it also calls aabb.scale().
        # So children don't need explicit update.

def translation(displacement):
    t = numpy.identity(4)
    t[0, 3] = displacement[0]
    t[1, 3] = displacement[1]
    t[2, 3] = displacement[2]
    return t

def scaling(scale):
    s = numpy.identity(4)
    s[0, 0] = scale[0]
    s[1, 1] = scale[1]
    s[2, 2] = scale[2]
    s[3, 3] = 1
    return s