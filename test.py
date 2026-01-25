"""
Export riêng 2 file DXF:
1. air_regions.dxf - Các vùng AIR
2. iron_regions.dxf - Vùng IRON = Total - AIR

Workflow trong Flux:
1. Import air_regions.dxf → tạo faces AIR
2. Import iron_regions.dxf → tạo faces IRON
"""

import os
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import warnings

warnings.filterwarnings('ignore')

DXF_SCALE = 1000.0


class Material(Enum):
    IRON = 1
    AIR = 0


# ============================================================================
# TRA Parsing
# ============================================================================
def _find_line_idx(lines: List[str], needle: str) -> int:
    for i, s in enumerate(lines):
        if needle in s:
            return i
    raise ValueError(f"Could not find: {needle!r}")


def parse_tra_nodes(path: str) -> Dict[int, Tuple[float, float]]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()
    coord_idx = _find_line_idx(lines, "Coordinates of the nodes")
    nodes = {}
    for s in lines[coord_idx + 1:]:
        parts = s.split()
        if len(parts) < 3:
            break
        try:
            nid = int(parts[0])
            x = float(parts[1].replace("D", "E"))
            y = float(parts[2].replace("D", "E"))
            nodes[nid] = (x, y)
        except ValueError:
            break
    return nodes


@dataclass
class Tri6Element:
    eid: int
    region: int
    n1: int
    n2: int
    n3: int
    n4: int
    n5: int
    n6: int
    material: Material = Material.IRON

    @property
    def corner_nodes(self):
        return [self.n1, self.n2, self.n3]


def parse_tra_elements(path: str) -> List[Tri6Element]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()
    eidx = _find_line_idx(lines, "Description of elements")
    cidx = _find_line_idx(lines, "Coordinates of the nodes")
    elements = []
    i = eidx + 1
    while i < cidx:
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        parts = s.split()
        if len(parts) >= 3:
            try:
                eid, nper, region = int(parts[0]), int(parts[1]), int(parts[2])
            except ValueError:
                i += 1
                continue
            if nper == 6 and i + 1 < cidx:
                conn = lines[i + 1].split()
                if len(conn) >= 6:
                    n1, n2, n3, n4, n5, n6 = [int(conn[j]) for j in range(6)]
                    elements.append(Tri6Element(eid=eid, region=region,
                                                n1=n1, n2=n2, n3=n3, n4=n4, n5=n5, n6=n6))
                i += 2
                continue
        i += 1
    return elements


# ============================================================================
# NGnet
# ============================================================================
@dataclass
class NGnetConfig:
    r_min: float
    r_max: float
    theta_min: float
    theta_max: float
    n_radial: int = 6
    n_angular: int = 5


class NGnet:
    def __init__(self, config: NGnetConfig):
        self.config = config
        self.centers = []
        self.weights = None
        cfg = config
        r_vals = np.linspace(cfg.r_min, cfg.r_max, cfg.n_radial)
        t_vals = np.linspace(cfg.theta_min, cfg.theta_max, cfg.n_angular)
        dr = (cfg.r_max - cfg.r_min) / max(cfg.n_radial - 1, 1)
        self.sigma = dr * 0.5
        for r in r_vals:
            for t in t_vals:
                self.centers.append((r * np.cos(t), r * np.sin(t)))
        self.weights = np.zeros(len(self.centers))
        print(f"NGnet: {len(self.centers)} centers")

    def set_random_weights(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.weights = np.random.uniform(-1, 1, len(self.centers))

    def phi(self, x, y):
        g = [np.exp(-((x-cx)**2 + (y-cy)**2) / (2*self.sigma**2)) for cx, cy in self.centers]
        s = sum(g)
        if s < 1e-12:
            return 0
        return sum(w * gi / s for w, gi in zip(self.weights, g))

    def get_material(self, x, y):
        return Material.IRON if self.phi(x, y) >= 0 else Material.AIR

    def is_in_design_region(self, x, y):
        cfg = self.config
        r = np.sqrt(x**2 + y**2)
        t = np.arctan2(y, x)
        if t < 0:
            t += 2 * np.pi
        return cfg.r_min <= r <= cfg.r_max and cfg.theta_min <= t <= cfg.theta_max


def assign_materials(elements, nodes, ngnet):
    for e in elements:
        cx = sum(nodes[n][0] for n in e.corner_nodes) / 3
        cy = sum(nodes[n][1] for n in e.corner_nodes) / 3
        if ngnet.is_in_design_region(cx, cy):
            e.material = ngnet.get_material(cx, cy)
        else:
            e.material = Material.IRON


# ============================================================================
# Geometry helpers
# ============================================================================
def element_to_polygon(elem, nodes, scale):
    coords = [(nodes[n][0] * scale, nodes[n][1] * scale) for n in elem.corner_nodes]
    coords.append(coords[0])
    try:
        p = Polygon(coords)
        return p.buffer(0) if not p.is_valid else p
    except:
        return None


def mirror_polygon_45(poly):
    """Mirror across y=x line: (x,y) -> (y,x)"""
    if poly.is_empty or not poly.is_valid:
        return poly
    ext = [(y, x) for x, y in poly.exterior.coords]
    ints = [[(y, x) for x, y in interior.coords] for interior in poly.interiors]
    try:
        m = Polygon(ext, ints)
        return m.buffer(0) if not m.is_valid else m
    except:
        return Polygon()


def polygon_to_list(geom):
    """Convert Polygon/MultiPolygon to list of Polygons"""
    if isinstance(geom, Polygon):
        return [geom] if geom.is_valid and not geom.is_empty else []
    elif isinstance(geom, MultiPolygon):
        return [p for p in geom.geoms if p.is_valid and not p.is_empty]
    return []


# ============================================================================
# DXF Export
# ============================================================================
def write_dxf(out_path: str, polygons: List[Polygon], layer: str = "REGION"):
    """Write polygons to DXF - each polygon as closed polyline"""
    
    def polyline_dxf(coords, layer_name):
        if len(coords) < 3:
            return []
        lines = ["0", "POLYLINE", "8", layer_name, "66", "1", "70", "1",
                 "10", "0.0", "20", "0.0", "30", "0.0"]
        for x, y in coords:
            lines.extend(["0", "VERTEX", "8", layer_name,
                         "10", f"{x:.12g}", "20", f"{y:.12g}", "30", "0.0"])
        lines.extend(["0", "SEQEND"])
        return lines
    
    # Header
    content = [
        "0", "SECTION", "2", "HEADER",
        "9", "$ACADVER", "1", "AC1009",
        "0", "ENDSEC",
        # Tables
        "0", "SECTION", "2", "TABLES",
        "0", "TABLE", "2", "LTYPE", "70", "1",
        "0", "LTYPE", "2", "CONTINUOUS", "70", "0", "3", "Solid", "72", "65", "73", "0", "40", "0.0",
        "0", "ENDTAB",
        "0", "TABLE", "2", "LAYER", "70", "1",
        "0", "LAYER", "2", layer, "70", "0", "62", "7", "6", "CONTINUOUS",
        "0", "ENDTAB",
        "0", "ENDSEC",
        # Blocks
        "0", "SECTION", "2", "BLOCKS", "0", "ENDSEC",
        # Entities
        "0", "SECTION", "2", "ENTITIES",
    ]
    
    for poly in polygons:
        if poly.exterior:
            content.extend(polyline_dxf(list(poly.exterior.coords), layer))
        for interior in poly.interiors:
            content.extend(polyline_dxf(list(interior.coords), layer))
    
    content.extend(["0", "ENDSEC", "0", "EOF", ""])
    
    with open(out_path, "w") as f:
        f.write("\n".join(content))
    
    print(f"  Wrote: {out_path} ({len(polygons)} polygons)")


# ============================================================================
# Main
# ============================================================================
def main(tra_path: str, output_dir: str = "."):
    print("=" * 60)
    print("Export AIR và IRON riêng biệt")
    print("=" * 60)
    
    # Load
    print("\n1. Loading mesh...")
    nodes = parse_tra_nodes(tra_path)
    elements = parse_tra_elements(tra_path)
    print(f"   {len(nodes)} nodes, {len(elements)} elements")
    
    radii = [np.sqrt(x**2 + y**2) for x, y in nodes.values()]
    r_min, r_max = min(radii), max(radii)
    
    # NGnet
    print("\n2. NGnet setup...")
    config = NGnetConfig(
        r_min=r_min + (r_max - r_min) * 0.1,
        r_max=r_max - (r_max - r_min) * 0.1,
        theta_min=0,
        theta_max=np.pi / 4,
        n_radial=6,
        n_angular=5,
    )
    ngnet = NGnet(config)
    ngnet.set_random_weights(seed=42)
    
    # Assign materials
    print("\n3. Assigning materials...")
    assign_materials(elements, nodes, ngnet)
    n_iron = sum(1 for e in elements if e.material == Material.IRON)
    n_air = sum(1 for e in elements if e.material == Material.AIR)
    print(f"   Iron: {n_iron}, Air: {n_air}")
    
    # Convert to polygons
    print("\n4. Building polygons...")
    iron_polys = []
    air_polys = []
    all_polys = []
    
    for e in elements:
        p = element_to_polygon(e, nodes, DXF_SCALE)
        if p and p.is_valid and not p.is_empty:
            all_polys.append(p)
            if e.material == Material.IRON:
                iron_polys.append(p)
            else:
                air_polys.append(p)
    
    # Merge
    print("\n5. Merging...")
    
    # Total region (all elements)
    print("   Merging total...")
    total_union = unary_union(all_polys)
    total_mirrored = mirror_polygon_45(total_union) if isinstance(total_union, Polygon) else unary_union([mirror_polygon_45(p) for p in polygon_to_list(total_union)])
    total_final = unary_union([total_union, total_mirrored])
    total_list = polygon_to_list(total_final)
    print(f"   -> Total: {len(total_list)} region(s)")
    
    # AIR regions
    print("   Merging AIR...")
    if air_polys:
        air_union = unary_union(air_polys)
        air_mirrored = mirror_polygon_45(air_union) if isinstance(air_union, Polygon) else unary_union([mirror_polygon_45(p) for p in polygon_to_list(air_union)])
        air_final = unary_union([air_union, air_mirrored])
        air_list = polygon_to_list(air_final)
    else:
        air_final = Polygon()
        air_list = []
    print(f"   -> AIR: {len(air_list)} region(s)")
    
    # IRON = Total - AIR
    print("   Computing IRON = Total - AIR...")
    if air_final and not air_final.is_empty:
        iron_final = total_final.difference(air_final)
    else:
        iron_final = total_final
    iron_list = polygon_to_list(iron_final)
    print(f"   -> IRON: {len(iron_list)} region(s)")
    
    # Simplify
    simplify_tol = 0.1
    air_list = [p.simplify(simplify_tol) for p in air_list]
    iron_list = [p.simplify(simplify_tol) for p in iron_list]
    
    # Export
    print("\n6. Exporting DXF files...")
    air_path = os.path.join(output_dir, "air_regions.dxf")
    iron_path = os.path.join(output_dir, "iron_regions.dxf")
    
    write_dxf(air_path, air_list, layer="AIR")
    write_dxf(iron_path, iron_list, layer="IRON")
    
    # Visualization
    print("\n7. Creating preview...")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MplPoly
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=150)
        
        # AIR
        ax1 = axes[0]
        ax1.set_aspect("equal")
        ax1.set_title(f"AIR regions ({len(air_list)})")
        for p in air_list:
            if p.exterior:
                ax1.fill(*zip(*p.exterior.coords), facecolor='lightyellow', 
                        edgecolor='orange', linewidth=1.5)
                for interior in p.interiors:
                    ax1.fill(*zip(*interior.coords), facecolor='white', 
                            edgecolor='orange', linewidth=1)
        ax1.autoscale()
        ax1.set_xlabel("x (mm)")
        ax1.set_ylabel("y (mm)")
        
        # IRON
        ax2 = axes[1]
        ax2.set_aspect("equal")
        ax2.set_title(f"IRON regions ({len(iron_list)})")
        for p in iron_list:
            if p.exterior:
                ax2.fill(*zip(*p.exterior.coords), facecolor='steelblue', 
                        edgecolor='darkblue', linewidth=1.5)
                for interior in p.interiors:
                    ax2.fill(*zip(*interior.coords), facecolor='white', 
                            edgecolor='darkblue', linewidth=1)
        ax2.autoscale()
        ax2.set_xlabel("x (mm)")
        ax2.set_ylabel("y (mm)")
        
        plt.tight_layout()
        png_path = os.path.join(output_dir, "air_iron_preview.png")
        fig.savefig(png_path)
        plt.close()
        print(f"  Wrote: {png_path}")
    except Exception as e:
        print(f"  Preview failed: {e}")
    
    print("\n" + "=" * 60)
    print("DONE!")
    print(f"\nFiles để import vào Flux:")
    print(f"  1. {air_path}  <- Import trước, gán vật liệu AIR")
    print(f"  2. {iron_path} <- Import sau, gán vật liệu IRON")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    tra = sys.argv[1] if len(sys.argv) > 1 else "rotor_2.TRA"
    if os.path.exists(tra):
        main(tra)
    else:
        print(f"File not found: {tra}")