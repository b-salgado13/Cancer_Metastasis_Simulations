import random
import numpy
from OpenGL.GL import *
from OpenGL.GLU import *
import math

# Global constants for display lists
G_OBJ_SPHERE = None
G_OBJ_CUBE = None
G_OBJ_PLANE = None

def init_primitives():
    global G_OBJ_SPHERE, G_OBJ_CUBE, G_OBJ_PLANE

    # Sphere
    G_OBJ_SPHERE = glGenLists(1)
    glNewList(G_OBJ_SPHERE, GL_COMPILE)
    quad = gluNewQuadric()
    gluSphere(quad, 0.5, 30, 30) # Radius 0.5 to match 1.0 diameter
    gluDeleteQuadric(quad)
    glEndList()

    # Cube
    G_OBJ_CUBE = glGenLists(1)
    glNewList(G_OBJ_CUBE, GL_COMPILE)
    glBegin(GL_QUADS)
    # Front Face
    glNormal3f(0.0, 0.0, 1.0)
    glVertex3f(-0.5, -0.5,  0.5)
    glVertex3f( 0.5, -0.5,  0.5)
    glVertex3f( 0.5,  0.5,  0.5)
    glVertex3f(-0.5,  0.5,  0.5)
    # Back Face
    glNormal3f(0.0, 0.0, -1.0)
    glVertex3f(-0.5, -0.5, -0.5)
    glVertex3f(-0.5,  0.5, -0.5)
    glVertex3f( 0.5,  0.5, -0.5)
    glVertex3f( 0.5, -0.5, -0.5)
    # Top Face
    glNormal3f(0.0, 1.0, 0.0)
    glVertex3f(-0.5,  0.5, -0.5)
    glVertex3f(-0.5,  0.5,  0.5)
    glVertex3f( 0.5,  0.5,  0.5)
    glVertex3f( 0.5,  0.5, -0.5)
    # Bottom Face
    glNormal3f(0.0, -1.0, 0.0)
    glVertex3f(-0.5, -0.5, -0.5)
    glVertex3f( 0.5, -0.5, -0.5)
    glVertex3f( 0.5, -0.5,  0.5)
    glVertex3f(-0.5, -0.5,  0.5)
    # Right face
    glNormal3f(1.0, 0.0, 0.0)
    glVertex3f( 0.5, -0.5, -0.5)
    glVertex3f( 0.5,  0.5, -0.5)
    glVertex3f( 0.5,  0.5,  0.5)
    glVertex3f( 0.5, -0.5,  0.5)
    # Left Face
    glNormal3f(-1.0, 0.0, 0.0)
    glVertex3f(-0.5, -0.5, -0.5)
    glVertex3f(-0.5, -0.5,  0.5)
    glVertex3f(-0.5,  0.5,  0.5)
    glVertex3f(-0.5,  0.5, -0.5)
    glEnd()
    glEndList()

    # Plane (Grid)
    G_OBJ_PLANE = glGenLists(1)
    glNewList(G_OBJ_PLANE, GL_COMPILE)
    glBegin(GL_LINES)
    glColor3f(0.8, 0.8, 0.8)
    for i in range(-20, 21):
        glVertex3f(i, 0, -20)
        glVertex3f(i, 0, 20)
        glVertex3f(-20, 0, i)
        glVertex3f(20, 0, i)
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

    def render(self):
        """ renders the item to the screen """
        glPushMatrix()
        glMultMatrixf(numpy.transpose(self.translation_matrix))
        glMultMatrixf(self.scaling_matrix)
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
    """ Sphere primitive """
    def __init__(self):
        super(Sphere, self).__init__()
        self.call_list = G_OBJ_SPHERE

class Cube(Primitive):
    """ Cube primitive """
    def __init__(self):
        super(Cube, self).__init__()
        self.call_list = G_OBJ_CUBE

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
