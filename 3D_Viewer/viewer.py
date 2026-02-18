import sys
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy
from scene import Scene
from interaction import Interaction
from node import init_primitives
import node
from cancer_cell import CancerCell

class Viewer(object):
    def __init__(self):
        """ Initialize the viewer. """
        self.init_interface()
        self.init_opengl()
        init_primitives()
        self.init_scene()
        self.init_interaction()

    def init_interface(self):
        """ initialize the window and register the render function """
        glutInit(sys.argv) # glutInit needs sys.argv
        glutInitWindowSize(640, 480)
        glutCreateWindow(b"Cancer Cell Modeller")
        glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB | GLUT_DEPTH) # Add GLUT_DEPTH
        glutDisplayFunc(self.render)

    def init_opengl(self):
        """ initialize the opengl settings to render the scene """
        self.inverseModelView = numpy.identity(4)
        self.modelView = numpy.identity(4)

        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)

        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, [0, 0, 1, 0])
        glLightfv(GL_LIGHT0, GL_SPOT_DIRECTION, [0, 0, -1])

        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_COLOR_MATERIAL)
        glClearColor(0.4, 0.4, 0.4, 0.0)

    def init_scene(self):
        """ initialize the scene object and initial scene """
        self.scene = Scene()
        self.create_sample_scene()

    def create_sample_scene(self):
        # Create three cancer cells and add them to the scene at fixed positions
        positions = [(0, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 2)]
        for pos in positions:
            cancer_node = CancerCell()
            cancer_node.translate(pos[0], pos[1], pos[2])
            self.scene.add_node(cancer_node)

    def init_interaction(self):
        """ init user interaction and callbacks """
        self.interaction = Interaction()
        self.interaction.register_callback('pick', self.pick)
        self.interaction.register_callback('move', self.move)
        self.interaction.register_callback('place', self.place)
        self.interaction.register_callback('rotate_color', self.rotate_color)
        self.interaction.register_callback('scale', self.scale)

    def main_loop(self):
        glutMainLoop()

    def render(self):
        """ The render pass for the scene """
        self.init_view()

        glEnable(GL_LIGHTING)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Load the modelview matrix from the current state of the trackball
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        loc = self.interaction.translation
        glTranslated(loc[0], loc[1], loc[2])
        glMultMatrixf(self.interaction.trackball.matrix)

        # store the inverse of the current modelview.
        # glGetFloatv returns column-major matrix as 1D array or 4x4 array?
        # PyOpenGL returns a numpy array usually.
        # We need to be careful with layout.
        currentModelView = glGetFloatv(GL_MODELVIEW_MATRIX)
        self.modelView = numpy.transpose(currentModelView)
        self.inverseModelView = numpy.linalg.inv(numpy.transpose(currentModelView))

        # render the scene. This will call the render function for each object
        # in the scene
        self.scene.render()

        # draw the grid
        glDisable(GL_LIGHTING)
        # Use a simpler grid drawing if display list fails or just call it
        # Assuming G_OBJ_PLANE is available globally from node module
        if node.G_OBJ_PLANE is not None:
             glCallList(node.G_OBJ_PLANE)

        glPopMatrix()

        # flush the buffers so that the scene can be drawn
        glFlush()

    def init_view(self):
        """ initialize the projection matrix """
        xSize, ySize = glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT)
        aspect_ratio = float(xSize) / float(ySize) if ySize > 0 else 1.0

        # load the projection matrix. Always the same
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        glViewport(0, 0, xSize, ySize)
        gluPerspective(70, aspect_ratio, 0.1, 1000.0)
        glTranslated(0, 0, -15)

    def get_ray(self, x, y):
        """
        Generate a ray beginning at the near plane, in the direction that
        the x, y coordinates are facing

        Consumes: x, y coordinates of mouse on screen
        Return: start, direction of the ray
        """
        self.init_view()

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # get two points on the line.
        # gluUnProject takes winX, winY, winZ, model, proj, view
        # We can pass model=None, proj=None to use current state?
        # Or fetch current state.
        # But we just reset Projection and ModelView.
        # So we rely on current state.

        # OpenGL coordinate system for y is inverted relative to window system?
        # glut passes y from top-left. OpenGL uses bottom-left.
        # But gluUnProject expects window coordinates.
        # However, we've already inverted y in Interaction.
        # Interaction passes `y = ySize - screen_y`.
        # So `y` here is OpenGL window coordinate (bottom-left origin).

        start = numpy.array(gluUnProject(x, y, 0.001))
        end = numpy.array(gluUnProject(x, y, 0.999))

        # convert those points into a ray
        direction = end - start
        direction = direction / numpy.linalg.norm(direction)

        return (start, direction)

    def pick(self, x, y):
        """ Execute pick of an object. Selects an object in the scene. """
        start, direction = self.get_ray(x, y)
        self.scene.pick(start, direction, self.modelView)

    def move(self, x, y):
        """ Execute a move command on the scene. """
        start, direction = self.get_ray(x, y)
        self.scene.move_selected(start, direction, self.inverseModelView)

    def rotate_color(self, forwards):
        """
        Rotate the color of the selected Node.
        Boolean 'forward' indicates direction of rotation.
        """
        self.scene.rotate_selected_color(forwards)

    def scale(self, up):
        """ Scale the selected Node. Boolean up indicates scaling larger."""
        self.scene.scale_selected(up)

    def place(self, shape, x, y):
        """ Execute a placement of a new primitive into the scene. """
        start, direction = self.get_ray(x, y)
        self.scene.place(shape, start, direction, self.inverseModelView)

if __name__ == "__main__":
    viewer = Viewer()
    viewer.main_loop()
