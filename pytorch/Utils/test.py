from ursina import *
import numpy as np


def ursina_skeleton_animation(animations, parents, interval=33, use_cylinder=True):
    class SkeletonAnimator(Entity):
        def __init__(self, animations_data, bone_parents, **kwargs):
            super().__init__(**kwargs)
            self.anim_data = animations_data
            self.bone_parents = bone_parents
            self.bones = []
            self.joints = []
            self.current_frame = 0
            self.create_skeleton()

        def create_skeleton(self):
            # 创建基础骨骼结构
            for j in range(self.anim_data.shape[1]):
                # 关节实体（球体表示）
                joint = Entity(
                    model="sphere",
                    scale=0.05,
                    color=color.red,
                    position=self.get_joint_position(0, j)
                )
                self.joints.append(joint)

                # 骨骼连接（圆柱体或立方体）
                if self.bone_parents[j] != -1:
                    start_pos = self.get_joint_position(0, j)
                    end_pos = self.get_joint_position(0, self.bone_parents[j])

                    if use_cylinder:
                        bone = Entity(
                            model="cylinder",
                            scale=(0.05, distance(start_pos, end_pos), 0.05),
                            color=color.blue,
                            position=midpoint(start_pos, end_pos),
                            rotation=self.calculate_rotation(start_pos, end_pos)
                        )
                    else:
                        bone = Entity(
                            model="cube",
                            scale=(0.05, distance(start_pos, end_pos), 0.05),
                            color=color.green,
                            position=midpoint(start_pos, end_pos),
                            rotation=self.calculate_rotation(start_pos, end_pos)
                        )
                    self.bones.append(bone)

        def calculate_rotation(self, start, end):
            # 计算骨骼连接体的旋转角度
            direction = end - start
            return Vec3(
                np.degrees(np.arctan2(-direction[2], direction[1])),
                np.degrees(np.arctan2(direction[0], np.sqrt(direction[1] ** 2 + direction[2] ** 2))),
                0
            )

        def get_joint_position(self, frame, joint):
            # 转换坐标系适配Ursina的3D空间
            return Vec3(
                self.anim_data[frame, joint, 0],
                self.anim_data[frame, joint, 2],
                self.anim_data[frame, joint, 1]
            )

        def update(self):
            # 动画帧更新逻辑
            self.current_frame = (self.current_frame + 1) % len(self.anim_data)

            # 更新关节位置
            for j in range(len(self.joints)):
                self.joints[j].position = self.get_joint_position(self.current_frame, j)

            # 更新骨骼连接
            bone_index = 0
            for j in range(len(self.bone_parents)):
                if self.bone_parents[j] != -1:
                    start_pos = self.joints[j].position
                    end_pos = self.joints[self.bone_parents[j]].position

                    # 更新骨骼位置和旋转
                    self.bones[bone_index].position = midpoint(start_pos, end_pos)
                    self.bones[bone_index].rotation = self.calculate_rotation(start_pos, end_pos)
                    self.bones[bone_index].scale_y = distance(start_pos, end_pos)

                    bone_index += 1

    # 创建Ursina应用
    app = Ursina()
    # window.color = color.black
    window.color = color.rgba(0.9, 0.9, 0.9, 1)
    # Sky()
    EditorCamera()

    # 初始化动画器
    animator = SkeletonAnimator(animations, parents)

    # 设置动画更新频率
    invoke(setattr, animator, 'current_frame', 0, delay=interval / 1000)
    Sequence(animator.update, loop=True, duration=interval / 1000).start()

    app.run()


# 辅助函数
def midpoint(a, b):
    return (a + b) / 2


def distance(a, b):
    return (a - b).length()


import numpy as np


# 测试数据生成
def generate_test_animation():
    # 定义骨骼结构（4个关节的简单人形）
    parents = [-1, 0, 1, 2]  # 父节点关系：root -> torso -> upper_arm -> lower_arm

    # 生成30帧动画数据（形状：frames, joints, coordinates）
    frames = 30
    animations = np.zeros((frames, 4, 3))

    # 基础位置
    animations[..., 0, :] = [0, 0, 0]  # 根节点（骨盆位置）
    animations[..., 1, :] = [0, 5, 0]  # 躯干
    animations[..., 2, :] = [0, 10, 3]  # 上臂（右）
    animations[..., 3, :] = [0, 10, -3]  # 上臂（左）

    # 添加手臂摆动动画
    for i in range(frames):
        angle = np.radians(i * 12)  # 每帧旋转12度
        animations[i, 2, 1] = 10 + 3 * np.sin(angle)  # 右臂上下摆动
        animations[i, 3, 1] = 10 - 3 * np.sin(angle)  # 左臂反向摆动

    return animations, parents


# 运行测试
if __name__ == "__main__":
    # 生成测试数据
    test_anim, test_parents = generate_test_animation()

    # 运行圆柱体骨骼版本
    ursina_skeleton_animation(
        animations=test_anim,
        parents=test_parents,
        interval=50,  # 50ms per frame (20fps)
        use_cylinder=True
    )

    # 运行立方体骨骼版本（注释掉上一个先运行）
    '''
    ursina_skeleton_animation(
        animations=test_anim,
        parents=test_parents,
        interval=30,  # 30ms per frame (~33fps)
        use_cylinder=False
    )
    '''