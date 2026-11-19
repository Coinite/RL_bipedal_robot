from __future__ import annotations

import argparse
import time
from pathlib import Path


def resolve_xml(xml_file: str) -> Path:
    candidates = [
        Path(xml_file),
        Path.cwd() / xml_file,
        Path.cwd().parent / "project" / xml_file,
        Path.cwd().parent / "project_2" / xml_file,
        Path.cwd().parent / "project_2_modified" / xml_file,
    ]
    for candidate in candidates:
        path = candidate.expanduser().resolve()
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find XML file: {xml_file}")


def check_mjwarp(xml_path: Path, device: str, nworld: int, nconmax: int, njmax: int) -> None:
    import mujoco
    import mujoco_warp as mjw
    import warp as wp

    wp.init()
    wp.set_device(device)
    mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
    model = mjw.put_model(mj_model)
    data = mjw.make_data(mj_model, nworld=nworld, nconmax=nconmax, njmax=njmax)
    mjw.forward(model, data)
    print(f"MJWarp model check passed on {device}: nworld={nworld}, nq={mj_model.nq}, nv={mj_model.nv}, nu={mj_model.nu}")


def view_model(args: argparse.Namespace) -> None:
    import mujoco
    import mujoco.viewer

    xml_path = resolve_xml(args.xml_file)
    if args.mjwarp_check:
        check_mjwarp(xml_path, args.device, args.nworld, args.nconmax, args.njmax)

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)

    dt = float(model.opt.timestep)
    start = time.perf_counter()
    print(f"Opening MuJoCo viewer for: {xml_path}")
    print("Close the viewer window to stop. Use --run if you want free simulation instead of a static inspection pose.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = args.camera_distance
        viewer.cam.azimuth = args.camera_azimuth
        viewer.cam.elevation = args.camera_elevation
        viewer.cam.lookat[:] = [args.lookat_x, args.lookat_y, args.lookat_z]

        while viewer.is_running():
            loop_start = time.perf_counter()
            if args.duration > 0.0 and loop_start - start >= args.duration:
                break

            if args.run:
                mujoco.mj_step(model, data)
            else:
                data.qpos[:] = model.qpos0
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)

            viewer.sync()
            sleep_time = dt / max(args.real_time_factor, 1e-6) - (time.perf_counter() - loop_start)
            if sleep_time > 0.0:
                time.sleep(sleep_time)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open the robot XML in a simple MuJoCo/MJWarp model playground.")
    parser.add_argument("--xml-file", default="robot_segment_trainable.xml")
    parser.add_argument("--run", action="store_true", help="Step the MuJoCo simulation instead of holding the qpos0 pose.")
    parser.add_argument("--duration", type=float, default=0.0, help="Viewer duration in seconds. Zero means until closed.")
    parser.add_argument("--real-time-factor", type=float, default=1.0)
    parser.add_argument("--mjwarp-check", action="store_true", help="Instantiate the model with mujoco-warp before opening the viewer.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--nworld", type=int, default=1)
    parser.add_argument("--nconmax", type=int, default=64)
    parser.add_argument("--njmax", type=int, default=256)
    parser.add_argument("--camera-distance", type=float, default=2.2)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-18.0)
    parser.add_argument("--lookat-x", type=float, default=0.0)
    parser.add_argument("--lookat-y", type=float, default=0.0)
    parser.add_argument("--lookat-z", type=float, default=0.55)
    return parser.parse_args()


if __name__ == "__main__":
    view_model(parse_args())
