import numpy as np
import trimesh
from pathlib import Path
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection 
import math
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union

SAVE_NPZ = False
SAVE_JSON = False
DEBUG = True
DEBUG_PLOT = False
MAKE_SVG = True
MAX_SLICE_DEPTH_CM = 0.3
MINIMUM_CONTOUR_SIZE = 1 # cm, minimum side of the bounding box
# A4 dimensions in pixels at 320 PPI
A4_W, A4_H = 2646, 3742  # A4 dimensions at 320 PPI
REQUESTED_PPI = 320  # pixels per inch for the SVG

# scale factor to enlarge the model during slicing.
# theoretically a factor of 10 yields a ~25 cm tall model, so contours will be expressed in cm.
SCALE_FACTOR = 0.5 # for stl 0.05 (model ~400mm), for .obj 10

def centroid(contour):
    return np.mean(contour, axis=0)

def compute_spine(slices, tolerance_deg=10):
    angles = []
    for i, sl in enumerate(slices):
        if len(sl) == 2:
            c1 = centroid(sl[0])
            c2 = centroid(sl[1])
            dx, dy = c2 - c1
            angle = math.degrees(math.atan2(dy, dx))
            angles.append(angle)
    if not angles:
        # no slices with two contours found
        angles = [0.0]  # default to 0 degrees, any angle is valid!

    # normalizes angles to [0,180)
    norm_angles = [a % 180 for a in angles]
    avg_angle = np.mean(norm_angles)
    avg_angle_rad = math.radians(avg_angle)

    # check coherence
    for a in norm_angles:
        if abs(a - avg_angle) > tolerance_deg:
            raise ValueError(f"Incoherent angles: {norm_angles}")
    
    # define the spine imaginary line
    dx, dy = math.cos(avg_angle_rad), math.sin(avg_angle_rad)

    # rotational matrix to later align the spine with the X axis
    R = np.array([
        [dx, -dy],
        [dy,  dx]
    ])

    length = 1e6
    p0 = (-dx*length, -dy*length)
    p1 = ( dx*length,  dy*length)
    line = LineString([p0, p1])

    # verify intersection with all contours
    for i, sl in enumerate(slices):
        for j, contour in enumerate(sl):
            poly = Polygon(contour)
            if not poly.intersects(line):
                raise ValueError(f"Slice {i}, contour {j} not intersecting spine line")

    # all good, now I can create a special rectangular contour representing the spine. 
    # The rectangle is centered at the origin, with width = maximum contour width, and height = MAX_SLICE_DEPTH_CM * 2 * number of slices
    max_width = 0.0
    for sl in slices:   
        for contour in sl:
            bbox_w, bbox_h = bounding_box_size(contour)
            # update max with with diagonal of bounding box
            diag = math.sqrt(bbox_h**2 + bbox_w**2)
            if diag > max_width:
                max_width = diag

    # width on y is MAX_SLICE_DEPTH_CM
    spine_width = MAX_SLICE_DEPTH_CM 
    half_h = spine_width / 2

    # width on x is max contour width
    half_w = max_width / 2

    # rettangolo in XY
    spine_contour = np.array([
        (-half_w, -half_h),
        ( half_w, -half_h),
        ( half_w,  half_h),
        (-half_w,  half_h),
        (-half_w, -half_h)
    ])

    # applica rotazione
    spine_contour = spine_contour @ R.T

    # translate to center of the model (TODO: currently at origin)
    # spine_contour += np.array([cx, cy])
    return spine_contour


def slices_points_from_stl(stl_path,
                           spacing_cm=MAX_SLICE_DEPTH_CM*2,
                           unit_in='cm',
                           up_axis='Z'):
    """
    Load an STL and return a list of slices.
    Each element is a list of contours; each contour is a list of (x,y) points in cm.

    Params:
      stl_path: path to the .stl file
      spacing_cm: distance between planes (cm)
      unit_in: units of the vertices ('m','cm','mm') -- always converted to cm internally
      up_axis: model 'up' axis ('X','Y','Z','-X','-Y','-Z') -- used to rotate the model to align +Z up
    """

    # if a .mtl was passed, try to resolve the corresponding geometry file
    p = Path(stl_path)
    if p.suffix.lower() == ".mtl":
        found = None
        for ext in (".obj", ".stl", ".ply", ".glb", ".gltf"):
            cand = p.with_suffix(ext)
            if cand.exists():
                found = cand
                break
        if found is None:
            raise FileNotFoundError(f"Provided .mtl but no geometry file (.obj/.stl/.ply/.glb) with the same name was found in: {p.parent}")
        stl_path = str(found)

    # factor to convert mesh units to cm
    unit_to_cm = {'m': 100.0, 'cm': 1.0, 'mm': 0.1}
    if unit_in not in unit_to_cm:
        raise ValueError("unit_in must be 'm'|'cm'|'mm'")

    mesh = trimesh.load_mesh(stl_path, force='mesh', process=True)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    # scale to cm (modifies the mesh in-place; Z is now in cm)
    mesh.apply_scale(unit_to_cm[unit_in] * SCALE_FACTOR)

    # if the model 'up' axis is not Z, rotate to align it to +Z
    def _axis_vector(axis):
        a = axis.lower()
        return {
            'x': np.array([1., 0., 0.]),
            '-x': np.array([-1., 0., 0.]),
            'y': np.array([0., 1., 0.]),
            '-y': np.array([0., -1., 0.]),
            'z': np.array([0., 0., 1.]),
            '-z': np.array([0., 0., -1.])
        }.get(a, np.array([0., 0., 1.]))

    if up_axis is not None and up_axis.lower() != 'z':
        v = _axis_vector(up_axis)
        target = np.array([0., 0., 1.])
        vn = v / np.linalg.norm(v)
        dot = np.dot(vn, target)
        # identical or opposite case
        if not np.allclose(dot, 1.0):
            if np.allclose(dot, -1.0):
                # 180 deg rotation about any perp axis (X chosen)
                axis = np.array([1., 0., 0.])
                angle = np.pi
            else:
                axis = np.cross(vn, target)
                axis = axis / np.linalg.norm(axis)
                angle = float(np.arccos(np.clip(dot, -1.0, 1.0)))
            rot = trimesh.transformations.rotation_matrix(angle, axis)
            mesh.apply_transform(rot)

    z_min = mesh.bounds[0][2]

    n_slices = int(np.ceil((mesh.bounds[1][2] - z_min) / spacing_cm))

    # generate n_slices planes starting from z_min with step spacing_cm
    # note: if the total available space > n_slices*spacing then only the first n_slices planes will be generated
    z_positions = [z_min + i * spacing_cm for i in range(n_slices)]

    all_slices = []
    for z in z_positions:
        section = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        if section is None:
            all_slices.append([])  # no contour in this plane
            continue

        # to_planar returns (path2d, transform). Polygons are shapely polygons.
        path2d, _ = section.to_planar()
        contours = []
        # path2d.polygons_full -> list of shapely polygons, each polygon has exterior.coords
    
        v = np.array(getattr(path2d, "vertices", [])) if hasattr(path2d, "vertices") else np.array([])
        if v.size:
            entities = getattr(path2d, "entities", [])
            for ent in entities:
                pts_idx = getattr(ent, "points", None)
                if pts_idx is None:
                    pts_idx = getattr(ent, "points_index", None)
                # skip if still None or empty
                if pts_idx is None:
                    continue
                # normalize into a list (handles scalars / numpy)
                if isinstance(pts_idx, (int, np.integer)):
                    pts_idx = [int(pts_idx)]
                else:
                    try:
                        pts_idx = [int(x) for x in pts_idx]
                    except Exception:
                        continue
                if len(pts_idx) == 0:
                    continue
                try:
                    pts = [tuple(map(float, v[idx])) for idx in pts_idx]
                    if pts and pts[0] != pts[-1]:
                        pts.append(pts[0])
                    contours.append(np.array(pts))
                except Exception:
                    continue
            # If we couldn't obtain per-entity contours, save all vertices as a single contour
            if not contours:
                contours.append(np.array([tuple(pt) for pt in v.tolist()]))
       
        all_slices.append(contours)

    return all_slices

def bounding_box(contour, border=0):
    xs = contour[:,0]
    ys = contour[:,1]
    return xs.min()-border, ys.min()-border, xs.max()+border, ys.max()+border

def bounding_box_size(contour):
    x_min, y_min, x_max, y_max = bounding_box(contour)
    return (x_max - x_min, y_max - y_min)

def translate(contour, dx, dy):
    return contour + np.array([dx, dy])

def contour_to_svg_path(contour):
    pts = " ".join(f"{int(x)},{int(y)}" for x,y in contour)
    return f"<polygon points=\"{pts}\" fill=\"none\" stroke=\"red\"/>"

def create_placements(contours_sorted, idx, contour, placements, x_placement, y_placement):
    sl_index = contours_sorted[idx][0]
    placements.append((sl_index, contour, x_placement, y_placement))
    contours_sorted.pop(idx)  # remove the used contour

def pack_sorted_contours(contours_sorted, border=25):

    placements = []
    x_cursor, y_cursor = 0, 0
    row_height = 0
    
    i = 0
    while i < len(contours_sorted):
        contour = contours_sorted[i][1]   # read without modifying the list
        xmin, ymin, xmax, ymax = bounding_box(contour, border)
        w, h = xmax-xmin, ymax-ymin

        if x_cursor + w > A4_W:
             # before wrapping to new line, try to fill with small ones found at the end of the array
            space_left = A4_W - x_cursor
            j = len(contours_sorted) - 1 
            while j>=0:
                sc = contours_sorted[j][1]
                sxmin, symin, sxmax, symax = bounding_box(sc, border)
                sw, sh = sxmax-sxmin, symax-symin
                if sw <= space_left and sh <= row_height:
                    create_placements(contours_sorted, j, sc, placements, x_cursor - sxmin, y_cursor - symin)
                    j -= 1
                    
                    x_cursor += sw
                    space_left = A4_W - x_cursor
                else:
                    break # break of the mini loop for small contours
                j -= 1

            # now wrap to a new line
            x_cursor = 0
            y_cursor += row_height
            row_height = 0
        
        if y_cursor + h > A4_H:
             # before changing page, try to fill with small ones
            space_down = A4_H - y_cursor
            space_left = A4_W - x_cursor

            j = len(contours_sorted) - 1
            while j>=0:
                sc = contours_sorted[j][1]
                sxmin, symin, sxmax, symax = bounding_box(sc , border)
                sw, sh = sxmax-sxmin, symax-symin
                if sh <= space_down and sw <= space_left:
                    create_placements(contours_sorted, j, sc, placements, x_cursor - sxmin, y_cursor - symin)
                    j -= 1

                    x_cursor += sw
                    space_left = A4_W - x_cursor
                else:
                    break # break of the mini loop for small contours
                j -= 1
            break # end of packing for this page
        
        create_placements(contours_sorted, i, contour, placements, x_cursor - xmin, y_cursor - ymin)
        x_cursor += w
        row_height = max(row_height, h)
    
    return placements

def bounding_box_area(contour):
    xmin, ymin = contour[:,0].min(), contour[:,1].min()
    xmax, ymax = contour[:,0].max(), contour[:,1].max()
    return (xmax - xmin) * (ymax - ymin)

def make_svgs(contours):
    # sort from largest to smallest
    contours_sorted = sorted(
        contours,
        key=lambda item: bounding_box_area(item[1]),
        reverse=True
    )

    svgs = []
    
    while contours_sorted: # while contours_sorted contains items
        # try to place as many contours as possible
        placements = pack_sorted_contours(contours_sorted, border=25)

        # build SVG for this page
        svg_parts = [
            f"<svg xmlns='http://www.w3.org/2000/svg' width='{A4_W}' height='{A4_H}' viewBox='0 0 {A4_W} {A4_H}'>",
            f"<rect x='0' y='0' width='{A4_W}' height='{A4_H}' fill='none' stroke='red'/>"
        ]

        for tag, contour, dx, dy in placements:
            c = translate(contour, dx, dy)
            svg_parts.append(contour_to_svg_path(c))

            # compute the center of the piece bounding box
            xmin, ymin, xmax, ymax = bounding_box(c)
            cx = (xmin + xmax) / 2
            cy = (ymin + ymax) / 2

            # write the number in yellow
            svg_parts.append(
                f"<text x='{cx}' y='{cy}' fill='yellow' font-size='60' text-anchor='middle' dominant-baseline='middle'>"
                f"{tag}</text>"
            )
        svg_parts.append("</svg>")
        svgs.append("\n".join(svg_parts))

    for i, svg in enumerate(svgs, start=1):
        with open(f"output_page_{i}.svg", "w") as f:
            f.write(svg)


def contour_to_pixels(contour_cm):
    cm_to_inch = 1.0 / 2.54
    pixels_per_cm = REQUESTED_PPI * cm_to_inch
    contour_px = [(x*pixels_per_cm, y*pixels_per_cm) for (x,y) in contour_cm]
    return contour_px

def add_contour_to_plot(contour, z_bottom, z_top, ax_extrude, face_color='cyan'):
    # remove any final point equal to the first
    if np.array_equal(contour[0], contour[-1]):
        base_pts = contour[:-1]
    else:
        base_pts = contour
    verts_bottom = [(float(pt[0]), float(pt[1]), z_bottom) for pt in base_pts]
    verts_top = [(float(pt[0]), float(pt[1]), z_top) for pt in base_pts]
    n = len(base_pts)
    faces = []
    for i in range(n):
        v0 = verts_bottom[i]
        v1 = verts_bottom[(i + 1) % n]
        v2 = verts_top[(i + 1) % n]
        v3 = verts_top[i]
        faces.append([v0, v1, v2, v3])

    # add top and bottom as polygons
    faces.append(verts_bottom)
    faces.append(list(reversed(verts_top)))
    poly3d = Poly3DCollection(faces, alpha=0.5, facecolors=face_color, edgecolors='r')
    ax_extrude.add_collection3d(poly3d)

if __name__ == "__main__":
    # first, delete any existing SVG files from the working directory
    existing_svgs = list(Path('.').glob('output_page_*.svg'))
    for svg_file in existing_svgs:
        svg_file.unlink()
    
    model = Path("Alpino.stl")

    spacing_cm = MAX_SLICE_DEPTH_CM * 2  # distance between slices in cm
    slices = slices_points_from_stl(str(model), spacing_cm=spacing_cm, unit_in='mm', up_axis='Z')
    # save as compressed npz (optional) — force object dtype to store nested heterogeneous lists
    
    if SAVE_NPZ:
        slices_arr = np.asarray(slices, dtype=object)
        np.savez_compressed("slices_points.npz", slices=slices_arr)

    if SAVE_JSON:
        # also save as JSON (converting tuples -> lists)
        slices_json = [
            [ [ [float(x), float(y)] for (x, y) in contour ] for contour in sl ]
            for sl in slices
        ]
        with open("slices_points.json", "w", encoding="utf-8") as f:
            json.dump({"slices": slices_json}, f, ensure_ascii=False, indent=2)

    if DEBUG:
        print(f"Generated {len(slices)} slices")
        # for each slice print number of contours
        for i, sl in enumerate(slices):
            print(f" Slice {i}: {len(sl)} contour")  

    for i, sl in enumerate(slices):
        if len(sl) > 2:
            slices[i] = []
            continue
        # rebuild the list filtering out too-small contours
        slices[i] = [
            contour for contour in sl
            if not any(dim < MINIMUM_CONTOUR_SIZE for dim in bounding_box_size(contour))
        ]

    # compute spine, if possible
    try:
        spine = compute_spine(slices)
    except ValueError as e:
        print("Errore:", e)
        exit(1)

    if MAKE_SVG:
        # gather all contours into a single list to generate the SVG. Since we must express them in pixels, convert cm to pixels
        all_contours = []
        for sl_index, sl in enumerate(slices):
            for contour_index, contour in enumerate(sl):
                all_contours.append((f"{sl_index}-{contour_index}", np.array(contour_to_pixels(contour)))) 
        
        make_svgs(all_contours)

    fig = plt.figure()
    if DEBUG_PLOT:
        # create a matplotlib 3D view to verify raw slicing results
        ax_points = fig.add_subplot(1,2,1, projection='3d')
        # plot points on the left subplot
        for slice_idx, sl in enumerate(slices):
            z = slice_idx * spacing_cm  
            for contour in sl:
                xs = [pt[0] for pt in contour]
                ys = [pt[1] for pt in contour]
                zs = [z] * len(contour)
                ax_points.plot(xs, ys, zs, marker='o', linestyle='None', markersize=2)
        ax_points.set_title("Slice points")
        ax_points.set_xlabel('X (cm)')
        ax_points.set_ylabel('Y (cm)')
        ax_points.set_zlabel('Z (cm)')
        print("Points visualization completed.")
        ax_extrude = fig.add_subplot(1,2,2, projection='3d')
    else:
        ax_extrude = fig.add_subplot(projection='3d')

    spine_faces = []
    slice_thickness = MAX_SLICE_DEPTH_CM
    # calcola intersezioni slice per slice

    spine_poly = Polygon(spine)
    z = 0.0
    for sl in slices:
        if len(sl) > 0:
            # slice con più contour
            contours = [Polygon(c) for c in sl]   # lista di poligoni
            slice_poly = unary_union(contours)    # unione in un unico poligono
            inter = spine_poly.intersection(slice_poly)
            if not inter.is_empty:
                if inter.geom_type == "Polygon":
                    polys = [inter]
                elif inter.geom_type == "MultiPolygon":
                    polys = list(inter.geoms)
                else:
                # altri tipi (LineString, Point...) li puoi ignorare o gestire a parte
                    polys = []

                bottoms = []
                tops = []
                for poly in polys:
                    bottom = [(x, y, z) for x, y in poly.exterior.coords] # questo è un contour!
                    bottoms.append(np.array(bottom))
                    top    = [(x, y, z + slice_thickness) for x, y in poly.exterior.coords]
                    tops.append(np.array(top))
                    # add_contour_to_plot(bottom, z, z + slice_thickness, ax_extrude, face_color="orange")  # None perché non serve il plot qui
            
            sl.append(bottoms)
            sl.append(tops)
            
        z += slice_thickness
    # da ora ogni slice ha top e bottom del spine aggiunti come ultimi 2 contour
    # ora per ogni slice, verifico l'intersezione tra il suo top e il bottom della slice successiva
    for i in range(len(slices) - 1): 
        sl_current = slices[i]
        sl_next = slices[i + 1]
        if len(sl_current) < 2 or len(sl_next) < 2:
            continue  # skip se non ci sono spine

        bottoms_current = sl_current[-2]  # penultimo elemento
        tops_next = sl_next[-1]           # ultimo elemento

        for bottom in bottoms_current:
            poly_bottom = Polygon(bottom[:, :2])  # solo XY
            for top in tops_next:
                poly_top = Polygon(top[:, :2])    # solo XY
                inter = poly_bottom.intersection(poly_top)
                if not inter.is_empty:
                    if inter.geom_type == "Polygon":
                        polys = [inter]
                    elif inter.geom_type == "MultiPolygon":
                        polys = list(inter.geoms)
                    else:
                        polys = []

                    for poly in polys:
                        contour_pts = np.array([(x, y) for x, y in poly.exterior.coords])
                        z_bottom = i * spacing_cm
                        z_top = (i + 1) * spacing_cm
                        add_contour_to_plot(contour_pts, z_bottom, z_top, ax_extrude, face_color='orange')

    # right subplot: extrusions
    thickness_cm = min(MAX_SLICE_DEPTH_CM, spacing_cm)

    for slice_idx, sl in enumerate(slices):
        # loop esclidendo gli ultimi 2 contour (spine)
        sl = sl[:-2]

        if len(sl) < 2:
            continue  # need at least 2 contours to extrude TO BE REMOVED

        z_bottom = slice_idx * spacing_cm
        z_top = z_bottom + thickness_cm

        for contour in sl:
            if contour is None or contour.shape[0] < 3:
                continue
           
            add_contour_to_plot(contour, z_bottom, z_top, ax_extrude)
  
   
    # add spine to the plot
    z_bottom = 0.0
    z_top = len(slices) * spacing_cm
    # add_contour_to_plot(spine, z_bottom, z_top, ax_extrude, face_color='orange')
    
    ax_extrude.set_title("Extruded slices")
    ax_extrude.set_xlabel('X (cm)')
    ax_extrude.set_ylabel('Y (cm)')
    ax_extrude.set_zlabel('Z (cm)')
    limits = np.r_[ax_extrude.get_xlim3d(), ax_extrude.get_ylim3d(), ax_extrude.get_zlim3d()]
    limits = [np.min(limits, axis=0), np.max(limits, axis=0)]
    ax_extrude.set(xlim3d=limits, ylim3d=limits, zlim3d=limits, box_aspect=(1, 1, 1))
    
    plt.show(block=True)