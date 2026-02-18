import random
import numpy
import math
from node import HierarchicalNode, Sphere, AABB

class CancerCell(HierarchicalNode):
    def __init__(self):
        super(CancerCell, self).__init__()

        # Central body of the cell
        body = Sphere()
        body.color = (1, 0.4, 0.4) # Main center ball color
        self.child_nodes.append(body)

        # Add bumps (smaller spheres)
        num_bumps = 15
        radius = 0.5 # Radius of the central sphere (since diameter is 1.0)

        for _ in range(num_bumps):
            bump = Sphere()
            bump.color = (1, 0.698, 0.4) # Small bumps color

            # Random position on surface
            theta = random.uniform(0, 2 * math.pi)
            phi = random.uniform(0, math.pi)

            x = radius * math.sin(phi) * math.cos(theta)
            y = radius * math.sin(phi) * math.sin(theta)
            z = radius * math.cos(phi)

            # Move bump to surface
            bump.translate(x, y, z)

            # Scale bump down
            # We need to scale it down significantly.
            # Sphere is diameter 1. We want diameter maybe 0.3?
            # scale 0.3
            # But scaling is incremental in Node.scale.
            # We can manually set scaling matrix or call scale multiple times?
            # Or just implement set_scale?
            # Node.scale multiplies current matrix.

            s = 0.3
            scale_mat = numpy.identity(4)
            scale_mat[0, 0] = s
            scale_mat[1, 1] = s
            scale_mat[2, 2] = s
            bump.scaling_matrix = numpy.dot(bump.scaling_matrix, scale_mat)
            bump.aabb.scale(s)

            self.child_nodes.append(bump)

        # Update AABB to cover everything
        # Central sphere is radius 0.5.
        # Bumps are at radius 0.5, with radius 0.15 (0.5 * 0.3).
        # So max extent is roughly 0.65.
        # Box should be roughly [-0.65, 0.65].
        # Initial AABB is [-0.25, 0.25]? No, initial AABB is center 0, size 0.5.
        # Size 0.5 means extent from center is 0.5. So [-0.5, 0.5].
        # So it covers the central sphere.
        # We need to expand it to cover bumps.
        self.aabb.size = numpy.array([0.7, 0.7, 0.7])
