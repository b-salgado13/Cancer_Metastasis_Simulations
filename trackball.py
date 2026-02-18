import numpy
import math

class Trackball(object):
    def __init__(self, theta=0, distance=15):
        self.theta = theta
        self.distance = distance
        self.renorm = 97
        self.trackball_radius = 0.8
        self.rotation = numpy.array([0, 0, 0, 1], dtype=float) # quaternion x, y, z, w
        self.matrix = numpy.identity(4)

    def drag_to(self, x, y, dx, dy):
        """
        Update the rotation based on mouse drag.
        x, y: current mouse position (normalized or screen coords, handled internally)
        dx, dy: change in mouse position
        """
        # Create a rotation quaternion from the drag
        # This is a simplified trackball implementation
        # We'll rotate around the axis perpendicular to the drag direction

        if dx == 0 and dy == 0:
            return

        # Simple approach: map dx to rotation around Y, dy to rotation around X
        # Adjust sensitivity as needed
        sensitivity = 0.5

        # Create quaternion for rotation around X (pitch)
        q_x = self._axis_angle_to_quaternion([1, 0, 0], dy * sensitivity)

        # Create quaternion for rotation around Y (yaw)
        q_y = self._axis_angle_to_quaternion([0, 1, 0], dx * sensitivity)

        # Combine rotations: new_rot = q_y * q_x * old_rot
        # Note: Order matters. This applies world-space rotations.
        # If we want local rotations, we might multiply differently.
        # But for a simple trackball examining an object, this usually works.

        # Let's use the provided inputs more directly to simulate a trackball
        # Ideally we map 2D points to 3D sphere and find rotation between them.
        # But the article interface just passes dx, dy.

        # Let's try to update the internal rotation state.
        # We'll accumulate rotations.

        self.rotation = self._quaternion_multiply(q_y, self.rotation)
        self.rotation = self._quaternion_multiply(q_x, self.rotation)

        # Normalize to prevent drift
        norm = numpy.linalg.norm(self.rotation)
        self.rotation /= norm

        self.matrix = self._quaternion_to_matrix(self.rotation)


    def _axis_angle_to_quaternion(self, axis, angle_degrees):
        angle_radians = math.radians(angle_degrees)
        axis = numpy.array(axis)
        axis = axis / numpy.linalg.norm(axis)

        s = math.sin(angle_radians / 2.0)
        c = math.cos(angle_radians / 2.0)

        return numpy.array([axis[0]*s, axis[1]*s, axis[2]*s, c])

    def _quaternion_multiply(self, q1, q2):
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 + y1*w2 + z1*x2 - x1*z2
        z = w1*z2 + z1*w2 + x1*y2 - y1*x2

        return numpy.array([x, y, z, w])

    def _quaternion_to_matrix(self, q):
        x, y, z, w = q

        # Rotation matrix from quaternion
        #     | 1 - 2y^2 - 2z^2    2xy - 2zw      2xz + 2yw       0 |
        # R = | 2xy + 2zw        1 - 2x^2 - 2z^2  2yz - 2xw       0 |
        #     | 2xz - 2yw          2yz + 2xw      1 - 2x^2 - 2y^2 0 |
        #     | 0                  0              0               1 |

        m = numpy.identity(4)

        m[0, 0] = 1 - 2*y*y - 2*z*z
        m[0, 1] = 2*x*y - 2*z*w
        m[0, 2] = 2*x*z + 2*y*w

        m[1, 0] = 2*x*y + 2*z*w
        m[1, 1] = 1 - 2*x*x - 2*z*z
        m[1, 2] = 2*y*z - 2*x*w

        m[2, 0] = 2*x*z - 2*y*w
        m[2, 1] = 2*y*z + 2*x*w
        m[2, 2] = 1 - 2*x*x - 2*y*y

        return m
