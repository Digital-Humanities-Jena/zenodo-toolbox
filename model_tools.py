import logging
import math
import os
from pathlib import Path
from PIL import Image
from typing import Any, Dict, List, Optional, Tuple, Union

import bpy
from mathutils import Vector  # type: ignore
from pygltflib import GLTF2

from utilities import search_file


logger = logging.getLogger("zenodo-toolbox")


def auto_position_camera(
    obj: bpy.types.Object, camera_location: Vector, camera_rotation: Tuple[float, float, float]
) -> bpy.types.Object:
    """
    Creates and positions a camera in the Blender scene.

    Args:
        obj: The object to focus on.
        camera_location: Location of the camera.
        camera_rotation: Rotation of the camera.

    Returns:
        [0] (bpy.types.Object): The created camera object.
    """
    camera_data = bpy.data.cameras.new(name="Camera")
    camera_object = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera_object)
    bpy.context.scene.camera = camera_object
    camera_object.location = camera_location
    camera_object.rotation_euler = camera_rotation
    return camera_object


def convert_OBJ_to_GLB(
    obj_filepath: Union[str, Path],
    textures: List[str] = [],
    search_textures: bool = True,
    ignore_missing_textures: bool = False,
) -> str:
    """
    Converts an OBJ file to GLB format, including materials and textures.

    Args:
        obj_filepath: Path to the OBJ file.
        textures: List of texture filenames to include.
        search_textures: Whether to search for textures in the OBJ directory and subdirectories.
        ignore_missing_textures: Whether to ignore missing textures.

    Returns:
        [0] Path to the generated GLB file, or an empty string if conversion fails.
    """
    obj_filepath = Path(obj_filepath)
    output_dir = obj_filepath.parent

    try:
        # Clear existing objects in the scene
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete()

        # Import the OBJ file
        bpy.ops.wm.obj_import(filepath=str(obj_filepath))

        # Process materials and textures
        for obj in bpy.context.selected_objects:
            if obj.type == "MESH":
                for mat in obj.data.materials:
                    if mat and mat.use_nodes:
                        for node in mat.node_tree.nodes:
                            if node.type == "TEX_IMAGE":
                                image = node.image
                                if image:
                                    texture_path = None
                                    if search_textures:
                                        ext = Path(image.filepath).suffix
                                        texture_path = search_file(
                                            image.name,
                                            search_dir=obj_filepath.parent,
                                            extensions=[ext],
                                            search_subdirectories=True,
                                        )
                                    elif image.name in textures:
                                        texture_path = obj_filepath.parent / image.name

                                    if texture_path and texture_path.exists():
                                        image.filepath = str(texture_path)
                                    elif not ignore_missing_textures:
                                        raise FileNotFoundError(f"Texture file {image.name} not found.")
                                    else:
                                        print(f"Warning: Texture file {image.name} not found. Ignoring.")

        # Set the output file path
        glb_filepath = output_dir / f"{obj_filepath.stem}.glb"

        # Export to GLB
        bpy.ops.export_scene.gltf(filepath=str(glb_filepath), export_format="GLB", use_selection=True)

        return str(glb_filepath)

    except Exception as e:
        print(f"Conversion Error (OBJ -> GLB): {obj_filepath} (Reason: {str(e)})")
        return ""


def extract_glb_metadata(
    glb_path: Union[str, Path]
) -> Dict[str, Union[int, float, List[float], Dict[str, float], List[str]]]:
    """
    Extracts comprehensive metadata from a GLB file.

    Args:
    glb_path: Path to the GLB file.

    Returns:
    [0] (dict) A dictionary containing detailed metadata of the GLB file, including file size, scene information, mesh statistics, bounding box, dimensions, and used extensions.
    """
    glb_path = Path(glb_path)
    gltf = GLTF2().load(str(glb_path))

    stats = {
        "file_size": glb_path.stat().st_size,
        "scenes": len(gltf.scenes),
        "nodes": len(gltf.nodes),
        "meshes": len(gltf.meshes),
        "materials": len(gltf.materials),
        "textures": len(gltf.textures),
        "images": len(gltf.images),
        "animations": len(gltf.animations),
        "skins": len(gltf.skins),
        "cameras": len(gltf.cameras),
        "total_primitives": 0,
        "total_vertices": 0,
        "total_indices": 0,
        "bounding_box": {
            "min": [float("inf"), float("inf"), float("inf")],
            "max": [float("-inf"), float("-inf"), float("-inf")],
        },
        "used_extensions": set(),
    }

    # Process meshes and their primitives
    for mesh in gltf.meshes:
        for primitive in mesh.primitives:
            stats["total_primitives"] += 1

            # Count vertices
            position_accessor_index = getattr(primitive.attributes, "POSITION", None)
            if position_accessor_index is not None:
                accessor = gltf.accessors[position_accessor_index]
                stats["total_vertices"] += accessor.count

                # Update bounding box
                if accessor.min and accessor.max:
                    for i in range(3):
                        stats["bounding_box"]["min"][i] = min(stats["bounding_box"]["min"][i], accessor.min[i])
                        stats["bounding_box"]["max"][i] = max(stats["bounding_box"]["max"][i], accessor.max[i])

            # Count indices
            if primitive.indices is not None:
                stats["total_indices"] += gltf.accessors[primitive.indices].count

    # Calculate dimensions
    dimensions = [stats["bounding_box"]["max"][i] - stats["bounding_box"]["min"][i] for i in range(3)]
    stats["dimensions"] = {"width": dimensions[0], "height": dimensions[1], "depth": dimensions[2]}

    # Collect used extensions
    if gltf.extensionsUsed:
        stats["used_extensions"] = set(gltf.extensionsUsed)

    # Convert sets to sorted lists for consistency
    stats["used_extensions"] = sorted(stats["used_extensions"])

    return stats


def extract_mtl_metadata(obj_path: Union[str, Path], stats: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extracts metadata from the MTL file referenced in the OBJ statistics.

    Parses the associated MTL file to gather information about materials and textures.

    Args:
    obj_path: Path to the OBJ file
    stats: Statistics dictionary returned by extract_obj_statistics

    Returns:
    [0] (dict): Dictionary containing MTL metadata, including:
        - filename: Name of the MTL file
        - materials: List of dictionaries, each representing a material
        - textures: List of texture filenames
    [1] (None): If MTL file is not found or no material library is specified
    """
    TEXTURE_KEYS = ["map_Kd", "map_Ks", "map_Ns", "map_d", "map_bump", "bump", "disp", "map_refl"]
    obj_path = Path(obj_path) if isinstance(obj_path, str) else obj_path
    obj_dir = obj_path.parent

    if not stats["material_libraries"]:
        print("No material library found in the OBJ file.")
        return None

    mtl_filename = stats["material_libraries"][0]
    mtl_path = obj_dir / mtl_filename

    if not mtl_path.exists():
        print(f"Material file {mtl_filename} not found.")
        return None

    mtl_metadata = {"filename": mtl_filename, "materials": [], "textures": set()}

    current_material = None

    with open(mtl_path, "r") as mtl_file:
        for line in mtl_file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                continue

            key, value = parts

            if key == "newmtl":
                if current_material:
                    mtl_metadata["materials"].append(current_material)
                current_material = {"name": value, "properties": {}, "textures": {}}
            elif key in TEXTURE_KEYS and current_material:
                texture_file = value.split()[-1]  # Get the last part (filename) of the value
                current_material["textures"][key] = texture_file
                mtl_metadata["textures"].add(texture_file)
            elif current_material:
                current_material["properties"][key] = value

    if current_material:
        mtl_metadata["materials"].append(current_material)

    mtl_metadata["textures"] = list(mtl_metadata["textures"])

    return mtl_metadata


def extract_obj_statistics(
    obj_path: Union[str, Path]
) -> Dict[str, Union[int, float, List[str], List[float], Dict[str, Union[float, List[float]]]]]:
    """
    Extracts comprehensive statistics from an OBJ file.

    Args:
        obj_path: Path to the OBJ file.

    Returns:
        [0] (dict) A dictionary containing various statistics about the OBJ file, including:
            - Counts of geometric elements (vertices, faces, etc.)
            - Object and group information
            - Material and texture data
            - Bounding box and dimensional information
            - Face area and estimated volume
    """
    stats = {
        "geometric_vertices": 0,
        "texture_coordinates": 0,
        "vertex_normals": 0,
        "parameter_space_vertices": 0,
        "faces": 0,
        "triangular_faces": 0,
        "quad_faces": 0,
        "n_gon_faces": 0,
        "objects": 0,
        "groups": 0,
        "smoothing_groups": 0,
        "material_libraries": [],
        "material_names": set(),
        "comments": 0,
        "object_names": set(),
        "group_names": set(),
        "bounding_box": {
            "min": [float("inf"), float("inf"), float("inf")],
            "max": [float("-inf"), float("-inf"), float("-inf")],
        },
        "dimensions": {"width": 0, "height": 0, "depth": 0},
        "total_face_area": 0,
        "total_volume": 0,
    }

    vertices = []
    current_object = "default"
    current_group = "default"

    def update_bounding_box(vertex):
        for i in range(3):
            stats["bounding_box"]["min"][i] = min(stats["bounding_box"]["min"][i], vertex[i])
            stats["bounding_box"]["max"][i] = max(stats["bounding_box"]["max"][i], vertex[i])

    def calculate_face_area(face_vertices):
        if len(face_vertices) < 3:
            return 0
        a, b, c = face_vertices[:3]
        vector1 = [b[i] - a[i] for i in range(3)]
        vector2 = [c[i] - a[i] for i in range(3)]
        cross_product = [
            vector1[1] * vector2[2] - vector1[2] * vector2[1],
            vector1[2] * vector2[0] - vector1[0] * vector2[2],
            vector1[0] * vector2[1] - vector1[1] * vector2[0],
        ]
        return 0.5 * math.sqrt(sum(x * x for x in cross_product))

    with open(obj_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if not parts:
                continue

            if parts[0] == "v":
                if len(parts) >= 4:
                    vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                    vertices.append(vertex)
                    update_bounding_box(vertex)
                    stats["geometric_vertices"] += 1
            elif parts[0] == "vt":
                stats["texture_coordinates"] += 1
            elif parts[0] == "vn":
                stats["vertex_normals"] += 1
            elif parts[0] == "vp":
                stats["parameter_space_vertices"] += 1
            elif parts[0] == "f":
                stats["faces"] += 1
                face_vertices = []
                for vert in parts[1:]:
                    idx = int(vert.split("/")[0]) - 1
                    if 0 <= idx < len(vertices):
                        face_vertices.append(vertices[idx])

                vertex_count = len(face_vertices)
                if vertex_count == 3:
                    stats["triangular_faces"] += 1
                elif vertex_count == 4:
                    stats["quad_faces"] += 1
                else:
                    stats["n_gon_faces"] += 1

                stats["total_face_area"] += calculate_face_area(face_vertices)
            elif parts[0] == "o":
                stats["objects"] += 1
                current_object = " ".join(parts[1:])
                stats["object_names"].add(current_object)
            elif parts[0] == "g":
                stats["groups"] += 1
                current_group = " ".join(parts[1:])
                stats["group_names"].add(current_group)
            elif parts[0] == "s":
                stats["smoothing_groups"] += 1
            elif parts[0] == "mtllib":
                stats["material_libraries"].append(" ".join(parts[1:]))
            elif parts[0] == "usemtl":
                stats["material_names"].add(" ".join(parts[1:]))
            elif parts[0] == "#":
                stats["comments"] += 1

    # Calculate dimensions
    for i in range(3):
        stats["dimensions"]["width" if i == 0 else "height" if i == 1 else "depth"] = (
            stats["bounding_box"]["max"][i] - stats["bounding_box"]["min"][i]
        )

    # Estimate volume (assuming the model is somewhat solid)
    stats["total_volume"] = stats["dimensions"]["width"] * stats["dimensions"]["height"] * stats["dimensions"]["depth"]

    # Convert sets to sorted lists for consistency
    stats["material_names"] = sorted(stats["material_names"])
    stats["object_names"] = sorted(stats["object_names"])
    stats["group_names"] = sorted(stats["group_names"])

    return stats


def resize_image(image_path: Union[str, Path], sizes: List[Union[int, float]], quiet: bool = False) -> None:
    """
    Resizes an image to multiple specified sizes and saves the results.

    Args:
        image_path: Path to the input image file.
        sizes: List of sizes to resize the image to (width and height will be equal).
        quiet: If True, suppresses output messages.

    Returns:
        None

    Raises:
        IOError: If the image file cannot be opened or saved.
        ValueError: If the sizes list is empty or contains invalid values.
    """
    with Image.open(image_path) as img:
        for size in sizes:
            resized_img = img.resize((size, size), Image.LANCZOS)
            resized_img_path = f"{os.path.splitext(image_path)[0]}_{size}x{size}.png"
            resized_img.save(resized_img_path)
            if not quiet:
                print(f"Saved resized image to {resized_img_path}")


def setup_lighting(light_type: str, energy: float, angle: Optional[float] = None) -> bpy.types.Object:
    """
    Sets up lighting in the Blender scene.

    Args:
        light_type: Type of light to create.
        energy: Energy/intensity of the light.
        angle: Angle of the light source.

    Returns:
        [0] (bpy.types.Object): The created light object.
    """
    light_data = bpy.data.lights.new(name="Light", type=light_type)
    if angle:
        light_data.angle = math.radians(angle)
    light_data.energy = energy
    light_object = bpy.data.objects.new("Light", light_data)
    bpy.context.collection.objects.link(light_object)
    return light_object


def setup_render_settings(resolution: int, samples: int, use_gpu: bool, device_type: str = "CUDA") -> None:
    """
    Configures render settings in Blender.

    Args:
        resolution: Resolution of the render.
        samples: Number of samples for rendering.
        use_gpu: Whether to use GPU for rendering.
        device_type: Type of GPU device to use.

    Returns:
        None
    """
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.samples = samples
    bpy.context.scene.render.resolution_x = resolution
    bpy.context.scene.render.resolution_y = resolution
    bpy.context.scene.render.film_transparent = True

    if use_gpu:
        bpy.context.preferences.addons["cycles"].preferences.compute_device_type = device_type
        bpy.context.preferences.addons["cycles"].preferences.get_devices()
        for device in bpy.context.preferences.addons["cycles"].preferences.devices:
            device.use = True
        bpy.context.scene.cycles.device = "GPU"


def render_model_thumbnails(
    model_path: Union[str, Path],
    output_dir: Union[str, Path] = None,
    output_name: str = None,
    num_perspectives: int = 4,
    include_top_perspective: bool = True,
    resolution: int = 1000,
    samples: int = 128,
    use_gpu: bool = False,
    gpu_device_type: str = "CUDA",
    light_type: str = "SUN",
    light_energy: float = 5.0,
    light_angle: float = 112.0,
    camera_distance_factor: float = 2.0,
    resize: bool = False,
    resize_dimensions: List[int] = [512, 256, 128],
    material_ior: float = 1.45,
    material_specular: float = 0.5,
) -> None:
    """
    Renders thumbnails of a 3D model from multiple perspectives.

    Args:
        model_path: Path to the 3D model file.
        output_dir: Directory to save rendered images.
        output_name: Base name for output files.
        num_perspectives: Number of perspective views to render.
        include_top_perspective: Whether to include a top-down view.
        resolution: Resolution of rendered images.
        samples: Number of samples for rendering.
        use_gpu: Whether to use GPU for rendering.
        gpu_device_type: Type of GPU device to use.
        light_type: Type of light to use in the scene.
        light_energy: Energy/intensity of the light.
        light_angle: Angle of the light source.
        camera_distance_factor: Factor to determine camera distance.
        resize: Whether to resize the rendered images.
        resize_dimensions: Dimensions for resizing.
        specular_ior: Specular IOR (Index of Refraction) for materials. Default is 1.45.

    Returns:
        None
    """
    model_path = Path(model_path)

    if output_dir is None:
        output_dir = model_path.parent
    else:
        output_dir = Path(output_dir)

    if output_name is None:
        output_name = model_path.stem

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(model_path))
    obj = bpy.context.selected_objects[0]

    # Set the Specular IOR for all materials
    set_material_properties(ior=material_ior, specular=material_specular)

    bbox = obj.bound_box
    bbox_center = sum((obj.matrix_world @ Vector(b) for b in bbox), Vector()) / 8
    bbox_size = max(bbox[6][0] - bbox[0][0], bbox[6][1] - bbox[0][1], bbox[6][2] - bbox[0][2])
    camera_distance = bbox_size * camera_distance_factor

    setup_render_settings(resolution, samples, use_gpu, gpu_device_type)

    perspectives = num_perspectives + (1 if include_top_perspective else 0)
    for i in range(perspectives):
        if i < num_perspectives:
            angle = math.radians(i * 360 / num_perspectives)
            camera_location = bbox_center + Vector((math.sin(angle), -math.cos(angle), 0.5)) * camera_distance
            camera_rotation = (math.radians(60), 0, angle)
            perspective_name = f"perspective_{i+1}"
        else:
            camera_location = bbox_center + Vector((0, 0, bbox_size * 2.5))
            camera_rotation = (0, 0, 0)
            perspective_name = "perspective_top"

        camera_object = auto_position_camera(obj, camera_location, camera_rotation)
        light_object = setup_lighting(light_type, light_energy, light_angle)
        light_object.location = camera_object.location

        output_path = output_dir / f"{output_name}_{perspective_name}.png"
        bpy.context.scene.render.filepath = str(output_path)
        bpy.ops.render.render(write_still=True)
        print(f"Saved render to {output_path}")

        if resize:
            resize_image(output_path, resize_dimensions)

    bpy.ops.wm.read_factory_settings(use_empty=True)


def set_material_properties(ior: float = 1.45, specular: float = 0.5) -> None:
    """
    Sets the IOR and Specular properties for all materials in the scene.

    Args:
        ior (float): The IOR value to set. Default is 1.45.
        specular (float): The Specular value to set. Default is 0.5.

    Returns:
        None
    """
    for material in bpy.data.materials:
        if material.use_nodes:
            principled_bsdf = next((node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"), None)
            if principled_bsdf:
                if "IOR" in principled_bsdf.inputs:
                    principled_bsdf.inputs["IOR"].default_value = ior
                if "Specular" in principled_bsdf.inputs:
                    principled_bsdf.inputs["Specular"].default_value = specular
                print(f"Set properties for material {material.name}: IOR={ior}, Specular={specular}")
