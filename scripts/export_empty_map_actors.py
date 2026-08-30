from __future__ import annotations

import json
from pathlib import Path

import unreal


OUTPUT_PATH = Path(r"D:\AI\UrbanFly\data\scene_simworld_dense\empty_map_actors.json")


def vec3(value) -> list[float]:
    return [float(value.x), float(value.y), float(value.z)]


def rot3(value) -> list[float]:
    return [float(value.pitch), float(value.yaw), float(value.roll)]


def get_class_path(obj) -> str:
    try:
        return obj.get_path_name()
    except Exception:
        try:
            return obj.get_name()
        except Exception:
            return str(obj)


def collect_materials(component) -> list[str]:
    materials: list[str] = []
    try:
        count = int(component.get_num_materials())
    except Exception:
        return materials
    for idx in range(count):
        try:
            material = component.get_material(idx)
        except Exception:
            material = None
        if material:
            materials.append(get_class_path(material))
    return materials


def collect_mesh_components(actor) -> list[dict]:
    rows: list[dict] = []
    try:
        components = actor.get_components_by_class(unreal.StaticMeshComponent)
    except Exception:
        return rows

    for comp in components:
        mesh = None
        try:
            mesh = comp.get_editor_property("static_mesh")
        except Exception:
            mesh = None

        try:
            comp_loc = vec3(comp.get_component_location())
            comp_rot = rot3(comp.get_component_rotation())
            comp_scale = vec3(comp.get_component_scale())
        except Exception:
            comp_loc = None
            comp_rot = None
            comp_scale = None

        try:
            origin, extent = comp.get_local_bounds()
            local_bounds = {
                "origin": vec3(origin),
                "extent": vec3(extent),
                "size": [float(extent.x) * 2.0, float(extent.y) * 2.0, float(extent.z) * 2.0],
            }
        except Exception:
            local_bounds = None

        rows.append(
            {
                "name": comp.get_name(),
                "class": get_class_path(comp.get_class()),
                "mesh_path": get_class_path(mesh) if mesh else "",
                "location": comp_loc,
                "rotation": comp_rot,
                "scale": comp_scale,
                "local_bounds": local_bounds,
                "materials": collect_materials(comp),
            }
        )
    return rows


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    if actor_subsystem is None:
        raise RuntimeError("EditorActorSubsystem is not available")

    actors = actor_subsystem.get_all_level_actors()
    payload_actors: list[dict] = []
    for actor in actors:
        if actor is None:
            continue

        try:
            origin, extent = actor.get_actor_bounds(False)
            actor_bounds = {
                "origin": vec3(origin),
                "extent": vec3(extent),
                "size": [float(extent.x) * 2.0, float(extent.y) * 2.0, float(extent.z) * 2.0],
            }
        except Exception:
            actor_bounds = None

        try:
            folder_path = str(actor.get_folder_path())
        except Exception:
            folder_path = ""

        payload_actors.append(
            {
                "name": actor.get_name(),
                "label": actor.get_actor_label(),
                "class": get_class_path(actor.get_class()),
                "location": vec3(actor.get_actor_location()),
                "rotation": rot3(actor.get_actor_rotation()),
                "scale": vec3(actor.get_actor_scale3d()),
                "folder": folder_path,
                "bounds": actor_bounds,
                "mesh_components": collect_mesh_components(actor),
            }
        )

    payload = {
        "source_map": "/Game/Maps/Empty",
        "actor_count": len(payload_actors),
        "actors": payload_actors,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"EXPORT_OK={OUTPUT_PATH}")
    print(f"ACTOR_COUNT={len(payload_actors)}")


if __name__ == "__main__":
    main()
