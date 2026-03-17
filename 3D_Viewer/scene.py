import sys
import numpy
from node import Sphere, Cube, HierarchicalNode


# ---------------------------------------------------------------------------
# Octree — spatial index for O(log n) ray-pick queries
# ---------------------------------------------------------------------------

class _OctreeNode:
    """
    One cell of the octree.  Leaf cells hold scene nodes directly; internal
    cells delegate to eight equally-sized children.

    All positions are in world/object space (the space where each scene node's
    translation_matrix lives), so the ray must be transformed into that same
    space before querying — see Octree.candidates().

    Conservative ray-box test
    -------------------------
    The slab method is used to test whether the ray hits this cell's bounding
    box.  An `expand` margin is added to each face of the box so that a scene
    node whose AABB extends slightly past the cell boundary is never silently
    missed.  This can produce a handful of false-positive candidates (nodes
    whose full AABB test then rejects the ray), but it guarantees zero false
    negatives, which is the correct trade-off for a picking accelerator.
    """

    # Maximum scene nodes in a leaf before it splits into 8 children.
    # 12 is a good balance: small enough to prune most of the scene per query,
    # large enough to avoid very deep trees for evenly-distributed tumour cells.
    MAX_ITEMS = 12

    # Hard cap on tree depth — prevents infinite recursion when many nodes
    # share the same position (e.g. coincident cells in a dense tumour core).
    MAX_DEPTH = 8

    def __init__(self, center, half_size, depth=0):
        self.center    = numpy.asarray(center, dtype=float)
        self.half_size = float(half_size)
        self.depth     = depth
        self.children  = None   # list[_OctreeNode] × 8, or None if leaf
        self.items     = []     # scene nodes (only populated in leaves)

    # ------------------------------------------------------------------
    # Insertion
    # ------------------------------------------------------------------

    def _child_index(self, pos):
        """Return 0–7: which octant does *pos* fall into?"""
        return ((1 if pos[0] >= self.center[0] else 0) |
                (2 if pos[1] >= self.center[1] else 0) |
                (4 if pos[2] >= self.center[2] else 0))

    def _child_center(self, idx):
        """World-space centre of child octant *idx*."""
        q = self.half_size * 0.5
        return self.center + numpy.array([
            q if (idx & 1) else -q,
            q if (idx & 2) else -q,
            q if (idx & 4) else -q,
        ])

    def _subdivide(self):
        """Split this leaf into 8 children and redistribute its items."""
        child_hs      = self.half_size * 0.5
        self.children = [
            _OctreeNode(self._child_center(i), child_hs, self.depth + 1)
            for i in range(8)
        ]
        for node in self.items:
            pos = node.translation_matrix[:3, 3]
            self.children[self._child_index(pos)].insert(node)
        self.items = []  # internal nodes hold no items

    def insert(self, scene_node):
        """Insert *scene_node* into the appropriate leaf."""
        pos = scene_node.translation_matrix[:3, 3]
        if self.children is not None:
            self.children[self._child_index(pos)].insert(scene_node)
            return
        self.items.append(scene_node)
        if (len(self.items) > self.MAX_ITEMS and
                self.depth < self.MAX_DEPTH):
            self._subdivide()

    # ------------------------------------------------------------------
    # Ray query
    # ------------------------------------------------------------------

    def _ray_hits_box(self, start, direction, expand):
        """
        Slab-method ray vs. AABB test.

        *expand* inflates every face of this cell's bounding box by that many
        world-space units, providing the conservative margin needed to avoid
        missing border nodes.
        """
        t_min = -float('inf')
        t_max =  float('inf')
        hs    = self.half_size + expand
        box_min = self.center - hs
        box_max = self.center + hs

        for i in range(3):
            d = direction[i]
            if abs(d) > 1e-9:
                t1 = (box_min[i] - start[i]) / d
                t2 = (box_max[i] - start[i]) / d
                t_min = max(t_min, min(t1, t2))
                t_max = min(t_max, max(t1, t2))
            elif start[i] < box_min[i] or start[i] > box_max[i]:
                # Ray is parallel to this slab and outside it — miss.
                return False

        return t_max >= t_min

    def query_ray(self, start, direction, expand):
        """Return all scene nodes in cells that the ray passes through."""
        if not self._ray_hits_box(start, direction, expand):
            return []
        if self.children is not None:
            result = []
            for child in self.children:
                result.extend(child.query_ray(start, direction, expand))
            return result
        return list(self.items)


class Octree:
    """
    Thin wrapper that builds the root _OctreeNode from a flat list of scene
    nodes and exposes a single candidates() method.

    Parameters
    ----------
    nodes       : list of scene Node objects
    node_radius : conservative AABB half-extent of the largest node type.
                  Used as the `expand` margin in ray-box tests so that a node
                  right on a cell boundary is never missed.  For CancerCell the
                  AABB size is 0.7, so the default of 1.0 provides ample margin.
    """

    def __init__(self, nodes, node_radius=1.0):
        self._node_radius = node_radius
        self._root = None

        if not nodes:
            return

        # Derive a bounding cube from all node positions plus some padding so
        # no node sits exactly on the outermost cell face.
        positions = numpy.array([n.translation_matrix[:3, 3] for n in nodes],
                                dtype=float)
        mins   = positions.min(axis=0) - node_radius
        maxs   = positions.max(axis=0) + node_radius
        center = (mins + maxs) * 0.5
        # half_size must be large enough to contain the furthest axis.
        half_size = float(((maxs - mins) * 0.5).max())

        self._root = _OctreeNode(center, half_size)
        for node in nodes:
            self._root.insert(node)

    def candidates(self, start_world, direction_world):
        """
        Return the subset of scene nodes whose octree cells are intersected by
        the ray (start_world, direction_world).  The caller must still run the
        precise AABB test on each returned node.

        Parameters
        ----------
        start_world     : (3,) array — ray origin in world/object space
        direction_world : (3,) array — unit ray direction in world/object space
        """
        if self._root is None:
            return []
        return self._root.query_ray(start_world, direction_world,
                                    expand=self._node_radius)


# ---------------------------------------------------------------------------

class Scene(object):

    # the default depth from the camera to place an object at
    PLACE_DEPTH = 15.0

    def __init__(self):
        # The scene keeps a list of nodes that are displayed
        self.node_list = list()
        # Keep track of the currently selected node.
        # Actions may depend on whether or not something is selected
        self.selected_node = None
        # Spatial index — built lazily on the first pick() call and
        # invalidated whenever the node list or a node's position changes.
        self._octree = None

    def add_node(self, node):
        """ Add a new node to the scene """
        self.node_list.append(node)
        self._octree = None   # stale — rebuilt on next pick()

    def render(self):
        """ Render the scene. """
        for node in self.node_list:
            node.render()

    def pick(self, start, direction, mat):
        """
        Execute selection via a two-phase ray test.

        Phase 1 — Octree cull (O(log n)):
            The octree is built in world/object space (the space where each
            node's translation_matrix lives).  The ray from viewer.get_ray()
            is computed with an identity modelview, so it arrives here in the
            same camera space used by AABB.ray_hit.  We first transform it
            into world space with inv(mat) so it matches the octree's
            coordinate frame, then ask the octree for candidate nodes.

        Phase 2 — Precise AABB test (O(k), k ≪ n):
            Only the octree candidates undergo the full AABB.ray_hit test
            (which already handles the modelview transform correctly).

        start, direction : ray in camera/eye space
        mat              : the current modelview matrix (world → eye)
        """
        # Deselect whatever was previously selected.
        if self.selected_node is not None:
            self.selected_node.select(False)
            self.selected_node = None

        # ---- Phase 1: octree cull ----------------------------------------
        # Build the index lazily so the first pick after any add_node/move
        # triggers a rebuild automatically.
        if self._octree is None and self.node_list:
            self._octree = Octree(self.node_list)

        if self._octree is not None:
            # Transform the camera-space ray into world/object space.
            inv_mat = numpy.linalg.inv(mat)

            start4     = numpy.append(start, 1.0)
            dir4       = numpy.append(direction, 0.0)
            start_w    = inv_mat.dot(start4)[:3]
            dir_w_raw  = inv_mat.dot(dir4)[:3]
            dir_w_norm = numpy.linalg.norm(dir_w_raw)
            dir_w      = (dir_w_raw / dir_w_norm
                          if dir_w_norm > 1e-9 else dir_w_raw)

            candidates = self._octree.candidates(start_w, dir_w)
        else:
            candidates = self.node_list

        # ---- Phase 2: precise AABB test on candidates only ----------------
        mindist      = sys.maxsize
        closest_node = None
        for node in candidates:
            hit, distance = node.pick(start, direction, mat)
            if hit and distance < mindist:
                mindist, closest_node = distance, node

        # If we hit something, keep track of it.
        if closest_node is not None:
            closest_node.select()
            closest_node.depth        = mindist
            closest_node.selected_loc = start + direction * mindist
            self.selected_node        = closest_node

    def place(self, shape, start, direction, inv_modelview):
        """
        Place a new node.

        Consume:
        shape the shape to add
        start, direction describes the Ray to move to
        inv_modelview is the inverse modelview matrix for the scene
        """
        new_node = None
        if shape == 'sphere': new_node = Sphere()
        elif shape == 'cube': new_node = Cube()
        elif shape == 'figure':
            # We need to import SnowFigure or similar if we want it.
            # Since user wants CancerCell, maybe default to that or Sphere?
            # Or just pass if unknown.
            # The article has 'figure' -> SnowFigure. I'll just skip it for now unless I implement SnowFigure.
            pass

        if new_node is None:
            return

        self.add_node(new_node)

        # place the node at the cursor in camera-space
        translation = (start + direction * self.PLACE_DEPTH)

        # convert the translation to world-space
        pre_tran = numpy.array([translation[0], translation[1], translation[2], 1])
        translation = inv_modelview.dot(pre_tran)

        new_node.translate(translation[0], translation[1], translation[2])

    def move_selected(self, start, direction, inv_modelview):
        """
        Move the selected node, if there is one.

        Consume:
        start, direction describes the Ray to move to
        mat is the modelview matrix for the scene
        """
        if self.selected_node is None: return

        # Find the current depth and location of the selected node
        node = self.selected_node
        depth = node.depth
        oldloc = node.selected_loc

        # The new location of the node is the same depth along the new ray
        newloc = (start + direction * depth)

        # transform the translation with the modelview matrix
        translation = newloc - oldloc
        pre_tran = numpy.array([translation[0], translation[1], translation[2], 0])
        translation = inv_modelview.dot(pre_tran)

        # translate the node and track its location
        node.translate(translation[0], translation[1], translation[2])
        node.selected_loc = newloc
        # Node position changed — the octree is now stale.
        self._octree = None

    def scale_selected(self, up):
        """ Scale the current selection """
        if self.selected_node is None: return
        self.selected_node.scale(up)

    def rotate_selected_color(self, forwards):
        """ Rotate the color of the currently selected node """
        if self.selected_node is None: return
        self.selected_node.rotate_color(forwards)