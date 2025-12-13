import numpy as np
import trimesh
from pathlib import Path
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection 

SAVE_NPZ = False
SAVE_JSON = False
DEBUG = True
DEBUG_PLOT = False
MAX_SLICE_DEPTH_CM = 0.3

def slices_points_from_stl(stl_path,
                           n_slices=100,
                           spacing_cm=MAX_SLICE_DEPTH_CM*2,
                           unit_in='cm',
                           up_axis='Z'):
    """
    Carica uno STL e restituisce una lista di n_slices elementi.
    Ogni elemento è una lista di contour; ogni contour è una lista di (x,y) punti in cm.

    Params:
      stl_path: percorso file .stl
      n_slices: numero di piani da campionare
      spacing_cm: distanza tra piani (cm)
      unit_in: unità in cui i vertici sono interpretati ('m','cm','mm') -- converte sempre in cm internamente
    """

    # se è stato passato un .mtl proviamo a risolvere il file geometria corrispondente
    p = Path(stl_path)
    if p.suffix.lower() == ".mtl":
        found = None
        for ext in (".obj", ".stl", ".ply", ".glb", ".gltf"):
            cand = p.with_suffix(ext)
            if cand.exists():
                found = cand
                break
        if found is None:
            raise FileNotFoundError(f"File .mtl fornito, ma non ho trovato file geometria (.obj/.stl/.ply/.glb) con lo stesso nome in: {p.parent}")
        stl_path = str(found)

    # fattore per convertire mesh unità in cm
    unit_to_cm = {'m': 100.0, 'cm': 1.0, 'mm': 0.1}
    if unit_in not in unit_to_cm:
        raise ValueError("unit_in deve essere 'm'|'cm'|'mm'")

    mesh = trimesh.load_mesh(stl_path, force='mesh', process=True)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    # scala in cm (modifica la mesh in-place; ora z è in cm)
    mesh.apply_scale(unit_to_cm[unit_in]*10)

 # se l'asse "up" del modello non è Z, ruotiamo per allinearlo a +Z
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
        # caso identico o opposto
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
    # generiamo n_slices piani partendo da z_min con step spacing_cm
    # attenzione: se lo spazio totale > n_slices*spacing allora verranno solo i primi n_slices piani
    z_positions = [z_min + i * spacing_cm for i in range(n_slices)]

    all_slices = []
    for z in z_positions:
        section = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        if section is None:
            all_slices.append([])  # nessuna contour in questo piano
            continue

        # to_planar restituisce (path2d, transform). Le polygoni sono shapely polygons.
        path2d, _ = section.to_planar()
        contours = []
        # path2d.polygons_full -> lista di shapely polygons, ogni polygon ha exterior.coords
    
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
                # normalizza in lista (gestisce anche scalari / numpy)
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
                    contours.append(pts)
                except Exception:
                    continue
            # Se non siamo riusciti a ricavare per-entità, salva tutti i vertici come unico contour
            if not contours:
                contours.append([tuple(pt) for pt in v.tolist()])
       
        all_slices.append(contours)

    return all_slices
    
if __name__ == "__main__":
    model = Path("Alpino.mtl")

    spacing_cm = MAX_SLICE_DEPTH_CM * 2  # distanza tra slice in cm
    slices = slices_points_from_stl(str(model), n_slices=150, spacing_cm=spacing_cm, unit_in='cm', up_axis='Y')
    # salva come npz compatto (opzionale) — forziamo object dtype per salvare liste annidate non omogenee
    
    if SAVE_NPZ:
        slices_arr = np.asarray(slices, dtype=object)
        np.savez_compressed("slices_points.npz", slices=slices_arr)

    if SAVE_JSON:
        # salva anche come JSON (convertendo tuple -> lista)
        slices_json = [
            [ [ [float(x), float(y)] for (x, y) in contour ] for contour in sl ]
            for sl in slices
        ]
        with open("slices_points.json", "w", encoding="utf-8") as f:
            json.dump({"slices": slices_json}, f, ensure_ascii=False, indent=2)

    if DEBUG:
        print(f"Generati {len(slices)} slices")
        # per ogni slice stampo numero di contour
        for i, sl in enumerate(slices):
            print(f" Slice {i}: {len(sl)} contour")  

    # cancello i contour dalle  slice con piu di 2 contour 
    for i, sl in enumerate(slices):
        if len(sl) > 2:
            slices[i] = []

    fig = plt.figure()
    if DEBUG_PLOT:
        # faccio versione matplotlib 3d per verificare i risultati
        ax_points = fig.add_subplot(projection='3d')
        # plot punti sul subplot sinistro
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
        print("Visualizzazione punti completata.")

    ax_extrude = fig.add_subplot(projection='3d')

    # subplot destro: estrusioni
    thickness_cm = min(MAX_SLICE_DEPTH_CM, spacing_cm)

    for slice_idx, sl in enumerate(slices):
        z_bottom = slice_idx * spacing_cm
        z_top = z_bottom + thickness_cm
        for contour in sl:
            if not contour or len(contour) < 3:
                continue
            # rimuovi eventuale punto finale uguale al primo
            if contour[0] == contour[-1]:
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
            # aggiungi top e bottom come poligoni
            faces.append(verts_bottom)
            faces.append(list(reversed(verts_top)))
            poly3d = Poly3DCollection(faces, alpha=0.5, facecolors='cyan', edgecolors='r')
            ax_extrude.add_collection3d(poly3d)
   
    ax_extrude.set_title("Extruded slices")
    ax_extrude.set_xlabel('X (cm)')
    ax_extrude.set_ylabel('Y (cm)')
    ax_extrude.set_zlabel('Z (cm)')
    limits = np.r_[ax_extrude.get_xlim3d(), ax_extrude.get_ylim3d(), ax_extrude.get_zlim3d()]
    limits = [np.min(limits, axis=0), np.max(limits, axis=0)]
    ax_extrude.set(xlim3d=limits, ylim3d=limits, zlim3d=limits, box_aspect=(1, 1, 1))
    
    plt.show()
