import mujoco_py
import os
import cv2
import numpy as np

# load scene in MuJoCo
mj_path, _ = mujoco_py.utils.discover_mujoco()
path = os.path.join(mj_path, 'model', 'humanoid.xml')
model = mujoco_py.load_model_from_path(path)
sim = mujoco_py.MjSim(model)

# to speed up computation we need the off screen rendering
viewer = mujoco_py.MjRenderContextOffscreen(sim, 0)
for i in range(3):
    viewer.render(420, 380, 0)
    data = np.asarray(viewer.read_pixels(420, 380, depth=False)[::-1, :, :], dtype=np.uint8)

    # save data
    if data is not None:
        cv2.imwrite("test{0}.png".format(i), data)

    print(i)
    sim.step()

