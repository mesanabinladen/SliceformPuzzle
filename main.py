import numpy as np
import trimesh
from pathlib import Path
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection 
import math
from shapely.geometry import MultiPoint, Polygon, LineString, box
from shapely.ops import unary_union

# CONSTANTS
# A4 dimensions in pixels at 320 PPI
A4_W, A4_H = 2646, 3742  # A4 dimensions at 320 PPI
REQUESTED_PPI = 320  # pixels per inch for the SVG
SAVE_NPZ = False
SAVE_JSON = False
DEBUG = True
DEBUG_PLOT = False
MAKE_SVG = True
ADD_SPINE_TO_PLOT = False
ADD_SLICES_TO_PLOT = False
MINIMUM_CONTOUR_SIZE = 1 # cm, minimum side of the bounding box

# PARAMETERS - TO BE EXPOSED TO THE USER
SLICE_HEIGHT_CM = 0.4 # maximum height of each sheet slice in cm
AIR_GAP_CM = 0.2        # gap between slices in cm
# scale factor to enlarge the model during slicing.
# theoretically a factor of 10 yields a ~25 cm tall model, so contours will be expressed in cm.
SCALE_FACTOR = 0.5 # for stl 0.05 (model ~400mm), for .obj 10

def centroid(contour):
    return np.mean(contour, axis=0)

def compute_base_spine(slices, tolerance_deg=10):
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

    # rotate all slices to -R  
    for sl in slices:
        for contour in sl:
            contour[:] = [np.dot(R.T, np.array([x, y])) for x, y in contour]

    length = 1e6
    # After rotating contours by R.T, the spine should lie on the X axis.
    # Make the spine an exact X-axis line from -length to +length.
    p0 = (-length, 0)
    p1 = ( length, 0)   
    line = LineString([p0, p1])

    # verify intersection with all contours
    for i, sl in enumerate(slices):
        for j, contour in enumerate(sl):
            poly = Polygon(contour)
            if not poly.intersects(line):
                raise ValueError(f"Slice {i}, contour {j} not intersecting spine line")

    # all good, now I can create a special rectangular contour representing the spine. 
    # The rectangle is centered at the origin, with width = maximum contour width, and height = (SLICE_HEIGHT_CM + AIR_GAP_CM) * number of slices
    max_width = 0.0
    for sl in slices:   
        for contour in sl:
            bbox_w, bbox_h = bounding_box_size(contour)
            # update max with with diagonal of bounding box
            diag = math.sqrt(bbox_h**2 + bbox_w**2)
            if diag > max_width:
                max_width = diag

    # width on y is SLICE_HEIGHT_CM
    spine_width = SLICE_HEIGHT_CM 
    half_h = spine_width / 2

    # width on x is max contour width
    half_w = max_width / 2

    # rectangle in XY
    spine_contour = np.array([
        (-half_w, -half_h),
        ( half_w, -half_h),
        ( half_w,  half_h),
        (-half_w,  half_h),
        (-half_w, -half_h)
    ])

    return spine_contour


def slices_points_from_stl(stl_path,
                           unit_in='cm',
                           up_axis='Z'):
    """
    Load an STL and return a list of slices.
    Each element is a list of contours; each contour is a list of (x,y) points in cm.

    Params:
      stl_path: path to the .stl file
      unit_in: units of the vertices ('m','cm','mm') -- always converted to cm internally
      up_axis: model 'up' axis ('X','Y','Z','-X','-Y','-Z') -- used to rotate the model to align +Z up
    """
    spacing_cm = SLICE_HEIGHT_CM + AIR_GAP_CM

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

    # Bounds in Z (in cm)
    z_min = float(mesh.bounds[0][2])
    z_max = float(mesh.bounds[1][2])

    # number of slices
    n_slices = int(np.ceil((z_max - z_min) / spacing_cm))

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

def add_contour_to_plot(contour, z_bottom, z_top, face_color="cyan"):
    global slices_plot 

    # remove any final point equal to the first
    if np.array_equal(contour[0], contour[-1]):
        base_pts = contour[:-1]
    else:
        base_pts = contour
    verts_bottom = [(float(pt[0]), float(pt[1]), z_bottom) for pt in base_pts]
    verts_top = [(float(pt[0]), float(pt[1]), z_top) for pt in base_pts]

    # appendo tutte le tuple (z,x) alla lista del profilo della spina, prendendoli da verts_bottom e verts_top a patto che y sia >=0

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
    slices_plot.add_collection3d(poly3d)

def clip_polygon_x_shapely(geom: Polygon, keep_pct: float = 0.50, anchor: str = 'left') -> Polygon:
    """
    Clippa (accorcia) una geometria lungo l'asse X mantenendo una percentuale della larghezza,
    usando l'intersezione con un rettangolo (box). Funziona con Polygon e MultiPolygon.

    Parametri:
      geom      : Polygon | MultiPolygon
                  Geometria di input.
      keep_pct  : float in (0, 1]
                  Percentuale di larghezza da mantenere (es. 0.70 = 70%).
      anchor    : 'left' | 'right'
                  - 'left'  → mantieni la parte sinistra (taglio a destra)
                  - 'right' → mantieni la parte destra (taglio a sinistra)

    Ritorna:
      Polygon | MultiPolygon | GeometryCollection (eventualmente vuota) a seconda dell'intersezione.

    Note:
      - I buchi (interiors) dei poligoni vengono gestiti automaticamente da Shapely.
      - Se l'intersezione è vuota (il poligono cade tutto fuori), si ottiene una geometria vuota.
      - Per mantenere una fascia centrale (invece di sinistra/destra), vedi l'esempio più sotto.
    """
    if not (0.0 < keep_pct <= 1.0):
        raise ValueError("keep_pct deve essere in (0, 1].")

    if anchor not in ('left', 'right'):
        raise ValueError("anchor deve essere 'left' oppure 'right'.")

    if geom.is_empty:
        return geom

    # Bounds della geometria
    minx, miny, maxx, maxy = geom.bounds
    width = maxx - minx
    if width <= 0.0:
        # Nessuna larghezza in X: ritorna com'è
        return geom

    if anchor == 'left':
        # Mantieni la parte sinistra: x ∈ [minx, minx + keep_pct * width]
        x_clip = minx + keep_pct * width
        clipper = box(minx, miny, x_clip, maxy)
    else:  # anchor == 'right'
        # Mantieni la parte destra: x ∈ [maxx - keep_pct * width, maxx]
        x_clip = maxx - keep_pct * width
        clipper = box(x_clip, miny, maxx, maxy)

    # Intersezione
    return geom.intersection(clipper)

def add_face_to_spine_contour(line_bottom, z_bottom, z_top):
    global full_spine_face_zx

    tmp_contour = []

    # order line_bottom points by increasing x
    line_bottom = sorted(line_bottom, key=lambda pt: pt[0])

    # remove duplicates in x (keep first occurrence)
    unique_line_bottom = []
    for x, y in line_bottom:
        if x not in [pt[0] for pt in unique_line_bottom]:
            unique_line_bottom.append((x, y))
    line_bottom = unique_line_bottom

    # add Z coordinate to close the loop, in clockwise order
    for x, y in line_bottom:
        tmp_contour.append( (z_bottom, x) )  # (z,x)
    for x, y in reversed(line_bottom):
        tmp_contour.append( (z_top, x) )     # (z,x) 

    tmp_contour.append(tmp_contour[0])  # close the loop

    full_spine_face_zx.append(tmp_contour)

def compute_spine_intersections(slices, spine):

    spacing_cm = SLICE_HEIGHT_CM + AIR_GAP_CM

    spine_poly = Polygon(spine)
    z = 0.0
    for i in range(len(slices) - 1): 
        sl_contours = slices[i]
        if len(sl_contours) > 0:
            # slice con più contour
            contours = [Polygon(c) for c in sl_contours]   # list of shapely polygons
    
            slice_poly = unary_union(contours)    # union of all polygons in the slice
            inter = spine_poly.intersection(slice_poly)
            
            if not inter.is_empty:
                if inter.geom_type == "Polygon":
                    polys = [inter]
                elif inter.geom_type == "MultiPolygon":
                    polys = list(inter.geoms)
                    # clean app from too small polygons
                    polys = [p for p in polys if p.area >= 1e-2]

                else:
                # igore other geometry types
                    polys = []

                bottoms = []
                tops = []

                if len(polys) != len(sl_contours):
                    print(f"ERROR: slice {i} number of intersection polygons ({len(polys)}) differ from number of original contours ({len(sl_contours)})")
                    exit(1)

                ctr_idx = 0
                for poly, contour in zip(polys, sl_contours):
                    
                    # if distance of the center of the poligon is positive along the positive spine direction, 
                    # remove 30% of the original polygon in positive X, else remove 30% from the negative x of the polygon
                    c = poly.centroid
                  
                    if c.x >= 0:
                        # remove 30% in positive X direction
                        cutted_poly = clip_polygon_x_shapely(poly, keep_pct=0.6, anchor='left')
                    else:
                        # remove 30% in negative X direction
                        cutted_poly = clip_polygon_x_shapely(poly, keep_pct=0.6, anchor='right')

                    bottom = [(x, y, z) for x, y in poly.exterior.coords] # questo è un contour!
                    bottoms.append(np.array(bottom))
                    top    = [(x, y, z + SLICE_HEIGHT_CM) for x, y in poly.exterior.coords]
                    tops.append(np.array(top))
                    
                    # Y an Z points are to be added to the spine final.shape...
                    cutted_bottom = [(x, y, z) for x, y in cutted_poly.exterior.coords] # questo è un contour!
                    
                    # aggiorno anche il contour nella slice, rimuovendo la parte di intersezione con cutted_poly
                    cutted_slice = Polygon(contour).difference(cutted_poly)
                    if cutted_slice.geom_type == "MultiPolygon":
                        if DEBUG:
                            print(f"WARNING: slice {i} contour produced MultiPolygon after spine cut; using largest polygon only.")
                        # keep only the largest polygon
                        largest_area = 0.0
                        largest_poly = None
                        for p in cutted_slice.geoms:
                            if p.area > largest_area:
                                largest_area = p.area
                                largest_poly = p    
                        cutted_slice = largest_poly
                    
                    sl_contours[ctr_idx] = np.array([(x, y) for x, y in cutted_slice.exterior.coords])

                    z_bottom = i * spacing_cm
                    z_top = z_bottom + SLICE_HEIGHT_CM
                    
                    # create a contour on the spine profile, only if y> (SLICE_HEIGHT_CM/2 - Eps) and y <  (SLICE_HEIGHT_CM/2 + Eps) 
                    line_bottom = [(x, y) for x, y in cutted_poly.exterior.coords if y <= (SLICE_HEIGHT_CM / 2 + 1e-3) and y >= (SLICE_HEIGHT_CM / 2 - 1e-3)]

                    add_face_to_spine_contour(line_bottom, z_bottom, z_top)
                    
                    if ADD_SPINE_TO_PLOT:
                        add_contour_to_plot(cutted_bottom, z_bottom, z_top, face_color='green')
                    
                    ctr_idx += 1

            sl_contours.append(bottoms)
            sl_contours.append(tops)
            
        z += spacing_cm

    if ADD_SLICES_TO_PLOT:
        add_slices_to_plot(slices)

    # now every slice has spine top and bottom added as last 2 contours
    contours_extruded = []
    
    # now, for each slice, check the intersection between its bottom and the top of the next slice and create extrusions to connect them
    for i in range(len(slices) - 1): 
        sl_current = slices[i]
        sl_next = slices[i + 1]
        if len(sl_current) < 2 or len(sl_next) < 2:
            continue  # skip if not enough contours

        bottoms_current = sl_current[-2]  # last but one element
        tops_next = sl_next[-1]           # last element

        for bottom in bottoms_current:
            poly_bottom = Polygon(bottom[:, :2])  # XY only
            for top in tops_next:
                poly_top = Polygon(top[:, :2])    # XY only
                inter = poly_bottom.intersection(poly_top)
                if not inter.is_empty:
                    if inter.geom_type == "Polygon":
                        polys = [inter]
                    elif inter.geom_type == "MultiPolygon":
                        polys = list(inter.geoms)
                    else:
                        polys = []

                    for poly in polys:
                        z_bottom = i * spacing_cm + SLICE_HEIGHT_CM
                        z_top = (i + 1) * spacing_cm
                        contours_extruded.append([z_bottom, z_top, np.array([(x, y) for x, y in poly.exterior.coords])])

                        # create a contour on the spine profile, only if y> (SLICE_HEIGHT_CM/2 - Eps) and y <  (SLICE_HEIGHT_CM/2 + Eps) 
                        line_bottom = [(x, y) for x, y in poly.exterior.coords if y <= (SLICE_HEIGHT_CM / 2 + 1e-3) and y >= (SLICE_HEIGHT_CM / 2 - 1e-3)]

                        add_face_to_spine_contour(line_bottom, z_bottom, z_top)
                        
        if ADD_SPINE_TO_PLOT:
            for z_bottom, z_top, extrusion in contours_extruded:
                add_contour_to_plot(extrusion, z_bottom, z_top, face_color='orange')

def add_slices_to_plot(slices, spacing_cm = SLICE_HEIGHT_CM + AIR_GAP_CM, thickness_cm = SLICE_HEIGHT_CM, max_contour_length=2):
    global slices_plot

    for slice_idx, sl in enumerate(slices):
        # loop excluding the last 2 contours (spine)
        sl = sl[:-2]

        if len(sl) < max_contour_length:
            continue  # need at least 2 contours to extrude TO BE REMOVED

        z_bottom = slice_idx * spacing_cm
        z_top = z_bottom + thickness_cm

        for contour in sl:
            if contour is None or contour.shape[0] < 3:
                continue
            
            add_contour_to_plot(contour, z_bottom, z_top)

if __name__ == "__main__":
    # first, delete any existing SVG files from the working directory
    existing_svgs = list(Path('.').glob('output_page_*.svg'))
    for svg_file in existing_svgs:
        svg_file.unlink()
    
    model = Path("Alpino.stl")

    slices = slices_points_from_stl(str(model), unit_in='mm', up_axis='Z')
    # save as compressed npz (optional) — force object dtype to store nested heterogeneous lists
    
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

    # compute spine, if possible, rotate slices and prepare the intersection slot in each slice
    try:
        spine = compute_base_spine(slices)
    except ValueError as e:
        print("Error calculating spine:", e)
        exit(1)

    # prepare matplotlib 3D plot
    fig = plt.figure()
    if DEBUG_PLOT:
        # create a matplotlib 3D view to verify raw slicing results
        points_plot = fig.add_subplot(1,2,1, projection='3d')
        # plot points on the left subplot
        for slice_idx, sl in enumerate(slices):
            z = slice_idx * (SLICE_HEIGHT_CM + AIR_GAP_CM)  
            for contour in sl:
                xs = [pt[0] for pt in contour]
                ys = [pt[1] for pt in contour]
                zs = [z] * len(contour)
                points_plot.plot(xs, ys, zs, marker='o', linestyle='None', markersize=2)
        points_plot.set_title("Slice points")
        points_plot.set_xlabel('X (cm)')
        points_plot.set_ylabel('Y (cm)')
        points_plot.set_zlabel('Z (cm)')
        print("Points visualization completed.")
        slices_plot = fig.add_subplot(1,2,2, projection='3d')
    else:
        slices_plot = fig.add_subplot(projection='3d')

    slices_plot.set_title("Extruded slices")
    slices_plot.set_xlabel('X (cm)')
    slices_plot.set_ylabel('Y (cm)')
    slices_plot.set_zlabel('Z (cm)')

    # compute intersections between spine and each slice

    full_spine_face_zx = [] # lista di punti sul piano zx da cui ricavare il profilo della spina

    compute_spine_intersections(slices, spine)
    
    merged = unary_union([Polygon(face) for face in full_spine_face_zx])
    if merged.geom_type == "Polygon":
        merged_faces = [merged]
    elif merged.geom_type == "MultiPolygon":    
        merged_faces = list(merged.geoms)
    else:
        merged_faces = []

    merged_contour = [(x, z) for x, z in merged.exterior.coords] # questo è un contour!

    if ADD_SPINE_TO_PLOT:
        for face in full_spine_face_zx:
            zs = [pt[0] for pt in face]
            xs = [pt[1] for pt in face]
            ys = [0.0] * len(face)  # spine lies on Y=0
            slices_plot.plot(xs, ys, zs, marker='o', linestyle='-', color='blue', markersize=2)
    
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

    if MAKE_SVG:
        # gather all slices contours into a single list to generate the SVG. Since we must express them in pixels, convert cm to pixels
        all_contours = []
        for sl_index, sl in enumerate(slices):
            sl = sl[:-2]
            for contour_index, contour in enumerate(sl):
                all_contours.append((f"{sl_index}-{contour_index}", np.array(contour_to_pixels(contour)))) 
        
        all_contours.append(("spine", np.array(contour_to_pixels(merged_contour))))
        
        make_svgs(all_contours)
   

    # prepare plot plotting and adjusting limits
    limits = np.r_[slices_plot.get_xlim3d(), slices_plot.get_ylim3d(), slices_plot.get_zlim3d()]
    limits = [np.min(limits, axis=0), np.max(limits, axis=0)]
    
    # Metti il piano X–Z di fronte: Y perpendicolare allo schermo
    slices_plot.view_init(elev=0, azim=-90)    # prova anche azim=-90 se preferisci invertire destra/sinistra

    # Proiezione ortografica (niente prospettiva/isometria)
    slices_plot.set_proj_type('ortho')

    slices_plot.set(xlim3d=limits, ylim3d=limits, zlim3d=limits, box_aspect=(1, 1, 1))
    
    plt.show(block=True)