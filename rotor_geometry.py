import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# Unit conversion:
# Flux2D .TRA here appears to be in meters (e.g. Rin ~ 0.016 m),
# while you want DXF in millimeters (Rin ~ 16 mm).
DXF_SCALE = 1000.0  # multiply coordinates by this before writing DXF/preview


def _find_line_idx(lines: List[str], needle: str) -> int:
    for i, s in enumerate(lines):
        if needle in s:
            return i
    raise ValueError(f"Could not find section header: {needle!r}")


def parse_tra_nodes(path: str) -> Dict[int, Tuple[float, float]]:
    """
    Parse Flux2D .TRA file node coordinates.

    Expected section:
      "Number of nodes" line earlier (not strictly needed)
      "Coordinates of the nodes"
      then N lines: <id> <x> <y>
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()

    coord_idx = _find_line_idx(lines, "Coordinates of the nodes")
    nodes: Dict[int, Tuple[float, float]] = {}
    # Parse until we hit an empty line or a non-numeric starter token
    for s in lines[coord_idx + 1 :]:
        parts = s.split()
        if len(parts) < 3:
            break
        try:
            nid = int(parts[0])
            x = float(parts[1].replace("D", "E"))
            y = float(parts[2].replace("D", "E"))
        except ValueError:
            break
        nodes[nid] = (x, y)

    if not nodes:
        raise ValueError("No nodes parsed from TRA file.")
    return nodes

@dataclass(frozen=True)
class Tri6Element:
    """
    Flux2D 2D second-order (quadratic) triangle element:
      n1, n2, n3 are corner nodes
      n4 is mid node on edge (n1-n2)
      n5 is mid node on edge (n2-n3)
      n6 is mid node on edge (n3-n1)

    We export its boundary as a polyline:
      n1 -> n4 -> n2 -> n5 -> n3 -> n6 -> n1
    """

    eid: int
    region: int
    n1: int
    n2: int
    n3: int
    n4: int
    n5: int
    n6: int

    @property
    def poly_node_ids(self) -> List[int]:
        return [self.n1, self.n4, self.n2, self.n5, self.n3, self.n6, self.n1]


def parse_tra_elements(path: str) -> List[Tri6Element]:
    """
    Parse Flux2D .TRA element connectivity from section:
      "Description of elements"

    Each element record is two lines:
      <eid> <nper> <region>
      <n1> <n2> <n3> <n4> <n5> <n6>

    Where nper is expected to be 6 for quadratic triangles in this dataset.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()

    eidx = _find_line_idx(lines, "Description of elements")
    cidx = _find_line_idx(lines, "Coordinates of the nodes")

    elements: List[Tri6Element] = []
    i = eidx + 1
    while i < cidx:
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        parts = s.split()
        # Expect header line: 3 ints
        if len(parts) >= 3:
            try:
                eid = int(parts[0])
                nper = int(parts[1])
                region = int(parts[2])
            except ValueError:
                i += 1
                continue
            if nper != 6:
                raise ValueError(f"Unsupported element order: element {eid} has {nper} nodes (expected 6).")

            if i + 1 >= cidx:
                raise ValueError(f"Unexpected EOF while reading connectivity for element {eid}.")
            conn = lines[i + 1].split()
            if len(conn) < 6:
                raise ValueError(f"Bad connectivity line for element {eid}: {lines[i+1]!r}")
            try:
                n1, n2, n3, n4, n5, n6 = (int(conn[0]), int(conn[1]), int(conn[2]), int(conn[3]), int(conn[4]), int(conn[5]))
            except ValueError as e:
                raise ValueError(f"Bad node ids for element {eid}: {lines[i+1]!r}") from e

            elements.append(Tri6Element(eid=eid, region=region, n1=n1, n2=n2, n3=n3, n4=n4, n5=n5, n6=n6))
            i += 2
            continue

        i += 1

    if not elements:
        raise ValueError("No elements parsed from TRA file.")
    return elements

def write_dxf_polylines_r12(
    out_path: str,
    polylines: List[List[Tuple[float, float]]],
    layer: str,
    other_layers: Optional[List[str]] = None,
) -> None:
    """
    Write an ASCII DXF (R12/AC1009) with 2D POLYLINE entities.

    Each polyline is written as:
      POLYLINE (66=1, 70=1 for closed) -> VERTEX* -> SEQEND

    Note: This uses classic POLYLINE (R12) instead of LWPOLYLINE (R13+).
    """

    def dxf_polyline(points: List[Tuple[float, float]], closed: bool = True) -> List[str]:
        if len(points) < 2:
            return []
        lines: List[str] = [
            "0",
            "POLYLINE",
            "8",
            layer,
            "66",
            "1",
            "70",
            "1" if closed else "0",
            "10",
            "0.0",
            "20",
            "0.0",
            "30",
            "0.0",
        ]
        for (x, y) in points:
            lines.extend(
                [
                    "0",
                    "VERTEX",
                    "8",
                    layer,
                    "10",
                    f"{x:.12g}",
                    "20",
                    f"{y:.12g}",
                    "30",
                    "0.0",
                ]
            )
        lines.extend(["0", "SEQEND"])
        return lines

    layers = [layer]
    if other_layers:
        for l in other_layers:
            if l not in layers:
                layers.append(l)

    # Minimal but importer-friendly TABLES section (R12)
    layer_table: List[str] = [
        "0",
        "TABLE",
        "2",
        "LAYER",
        "70",
        str(len(layers)),
    ]
    for lname in layers:
        layer_table.extend(
            [
                "0",
                "LAYER",
                "2",
                lname,
                "70",
                "0",
                "62",
                "7",
                "6",
                "CONTINUOUS",
            ]
        )
    layer_table.extend(["0", "ENDTAB"])

    tables = [
        "0",
        "SECTION",
        "2",
        "TABLES",
        # LTYPE table with CONTINUOUS
        "0",
        "TABLE",
        "2",
        "LTYPE",
        "70",
        "1",
        "0",
        "LTYPE",
        "2",
        "CONTINUOUS",
        "70",
        "0",
        "3",
        "Solid line",
        "72",
        "65",
        "73",
        "0",
        "40",
        "0.0",
        "0",
        "ENDTAB",
        *layer_table,
        "0",
        "ENDSEC",
    ]

    blocks = [
        "0",
        "SECTION",
        "2",
        "BLOCKS",
        "0",
        "ENDSEC",
    ]

    entities: List[str] = []
    for pts in polylines:
        entities.extend(dxf_polyline(pts, closed=True))

    content = "\n".join(
        [
            "0",
            "SECTION",
            "2",
            "HEADER",
            "9",
            "$ACADVER",
            "1",
            "AC1009",
            "0",
            "ENDSEC",
            *tables,
            *blocks,
            "0",
            "SECTION",
            "2",
            "ENTITIES",
            *entities,
            "0",
            "ENDSEC",
            "0",
            "EOF",
            "",
        ]
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)


def build_element_polylines(
    tra_path: str,
    scale: float = DXF_SCALE,
) -> List[List[Tuple[float, float]]]:
    nodes_m = parse_tra_nodes(tra_path)
    elements = parse_tra_elements(tra_path)

    polylines: List[List[Tuple[float, float]]] = []
    for e in elements:
        pts: List[Tuple[float, float]] = []
        for nid in e.poly_node_ids:
            if nid not in nodes_m:
                raise ValueError(f"Element {e.eid} references missing node id: {nid}")
            x, y = nodes_m[nid]
            pts.append((x * scale, y * scale))
        polylines.append(pts)

    return polylines


def export_all_elements_to_dxf(
    tra_path: str,
    out_path: str,
    scale: float = DXF_SCALE,
) -> None:
    polylines = build_element_polylines(tra_path, scale=scale)
    write_dxf_polylines_r12(out_path, polylines, layer="ELEMENTS")


def write_png_preview(
    out_path: str,
    polylines: List[List[Tuple[float, float]]],
) -> bool:
    """
    Save a PNG preview using matplotlib (if available).
    Returns True if written, False if matplotlib is not available.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
    except Exception:
        return False

    fig = plt.figure(figsize=(7, 7), dpi=220)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.4, alpha=0.35)

    # Plot many element polylines with thin strokes
    for pts in polylines:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, "-", linewidth=0.35, color="#0B5FFF", alpha=0.55)

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(f"Mesh elements ({len(polylines)} polylines)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True

def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    tra_path = os.path.join(here, "rotor_2.TRA")

    # Export full mesh connectivity as DXF polylines (in mm)
    dxf_mesh_out = os.path.join(here, "rotor_mesh_elements.dxf")
    export_all_elements_to_dxf(tra_path, dxf_mesh_out, scale=DXF_SCALE)
    print("Wrote mesh elements DXF:")
    print(f"  {dxf_mesh_out}")

    # PNG preview for the mesh (same geometry as the DXF output)
    png_out = os.path.join(here, "rotor_mesh_elements_preview.png")
    polylines = build_element_polylines(tra_path, scale=DXF_SCALE)
    wrote_png = write_png_preview(png_out, polylines)
    if wrote_png:
        print("Wrote mesh preview PNG:")
        print(f"  {png_out}")


if __name__ == "__main__":
    main()
