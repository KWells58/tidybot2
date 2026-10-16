
import mujoco
import numpy as np


class SingularityMonitor:
    def __init__(self, model_path="models/stanford_tidybot/scene.xml"):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.site_id = self.model.site("pinch_site").id
        self.base_dofs = self.model.body("base_link").jntnum.item()
        self.arm_dofs = 7

        self.jac_pos = np.zeros((3, self.model.nv))
        self.jac_rot = np.zeros((3, self.model.nv))

    def measure(self, arm_q, base_pose):
        self.data.qpos[:self.base_dofs] = base_pose
        self.data.qpos[
            self.base_dofs:self.base_dofs + self.arm_dofs
        ] = arm_q

        mujoco.mj_forward(self.model, self.data)

        mujoco.mj_jacSite(
            self.model,
            self.data,
            self.jac_pos,
            self.jac_rot,
            self.site_id
        )

        columns = slice(
            self.base_dofs,
            self.base_dofs + self.arm_dofs
        )

        J = np.vstack([
            self.jac_pos[:, columns],
            self.jac_rot[:, columns]
        ])

        singular_values = np.linalg.svd(
            J,
            compute_uv=False
        )

        return {
            "sigma_min": float(singular_values[-1]),
            "singular_values": singular_values,
        }
