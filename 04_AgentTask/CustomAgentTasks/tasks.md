# 环境任务总览

当前有 **8 类可运行独立任务**；图原型已经退役并移入 `archive/agent-interaction/legacy-demos/symbolic/`。当前为环境构建与功能验证阶段，尚未进行强化学习。

## 四类基础任务

|任务|环境|代理需要做什么|考察能力|
|---|---|---|---|
|[missing_object](missing_object/README.md)|AI2-THOR|探索并记忆场景；物体消失后，回答少了什么、原来在哪里。|空间记忆、变化检测、对象定位|
|[moved_object](moved_object/README.md)|AI2-THOR|比较变化前后的场景，找到移动物体并报告原位置和新位置。|主动搜索、位置记忆、前后对应|
|[shape_identification](shape_identification/README.md)|轻量三维几何环境|移动或转动相机，从候选中选出与目标相同的三维形状。|主动选视点、三维形状识别|
|[pose_alignment](pose_alignment/README.md)|轻量三维几何环境|独立控制相机与物体，将物体绕三轴旋转到参考姿态。|视角变换、旋转组合、姿态推理|

## 已完成的原型迁移

以下任务均使用真实 AI2-THOR 室内场景，已通过三个难度的 oracle 执行与轨迹回放。

|任务|任务流程|考察能力|
|---|---|---|
|[室内地标记忆导航（memory_maze）](memory_maze/README.md)|探索并记住地标；相机复位后按顺序重访，并返回指定目标。|空间记忆、有序导航|
|[容器依赖取物（dependency_doors）](dependency_doors/README.md)|打开容器、取出物品、逐件运送并放置，最后关闭来源容器。|操作先决条件、多步规划、持物约束|
|[空间关系重排（relational_rearrangement）](relational_rearrangement/README.md)|拿起目标，选择放置位置以满足方位/距离关系，同时保护锚点和其他物体。|空间关系落地、约束满足|
|[动态路径重规划（dynamic_replanning）](dynamic_replanning/README.md)|抵达检查点后，真实障碍封住原路；重新观察、绕行并到达目标。|变化感知、路径重规划|

## 任务的关键设置（trick）与动作

这里的 trick 指任务刻意设置的难点或依赖条件。**模拟器原生约束**与**benchmark 额外规则**分别说明；并不意味着每个案例都已证明“必须交互才能完成”。

室内任务的基础导航动作为 `MoveAhead`、`MoveBack`、`RotateLeft`、`RotateRight`、`LookUp`、`LookDown`。下列动作链省略中间的导航；物体编号取自当前 episode，不同案例不能混用。

|任务|环境里的关键设置|典型动作链|
|---|---|---|
|物体缺失 `missing_object`|AI2-THOR 中先给初始场景，再由环境停用一个未知目标；代理需记住对象及其承载物。L2 从看不到小目标的视角开始。|探索 → `MEMORIZE` → 重新观察 → `SUBMIT`，回答缺失对象和原位置。记忆预算耗尽也会自动切换阶段。|
|物体移动 `moved_object`|AI2-THOR 中目标仍存在，但被环境移到另一有效位置；只发现原处空了，还不足以回答新位置。|探索 → `MEMORIZE` → 搜索新位置 → `SUBMIT`，回答对象、原位置和当前位置。|
|形状识别 `shape_identification`|程序生成三维积木形状和不同几何的候选；在可构造时加入初始投影相似的干扰项，但不保证每例都有歧义。|`CAMERA_LEFT/RIGHT` 等平移、`CAMERA_YAW_LEFT/RIGHT` 等转向 → `SUBMIT_SHAPE(candidate_id)`。|
|姿态对齐 `pose_alignment`|相机姿态与物体姿态分开；仅换观察角度不会使物体真的对齐。几何对称的等价姿态也算正确。|按需调整相机 → `OBJECT_ROTATE_X_POS`、`OBJECT_ROTATE_Y_NEG` 等三轴正/负 90° 旋转→ `SUBMIT_POSE`。|
|地标记忆 `memory_maze`|AI2-THOR 中探索后把相机送回起点；地标没有标出完整坐标或路线。必须按顺序到访，仅远远看见不算完成。|探索 → `MEMORIZE`（或预算自动切换）→ 逐个靠近并 `MARK(object_id)` → 留在最后地标附近 `FINISH`。|
|容器取物 `dependency_doors`|AI2-THOR 的关闭橱柜内放置目标，实际验证关门时拿不到、开门后可取出；另设单手搬运、订单顺序和结束前关柜要求。|靠近来源 → `OpenObject` → `PickupObject` → 运到目的地 → `PutObject` → 关闭来源 `CloseObject` → 全部完成后 `FINISH`。|
|关系重排 `relational_rearrangement`|AI2-THOR 台面上有正确与错误的可放置点；方向按固定世界坐标定义，转身不改变东/西/南/北。锚点和其他物体不能被移动或操作。|`PickupObject(目标)` → 观察罗盘和局部点位 → `PLACE(slot_id)` → 空手 `FINISH`。PLACE 只执行所选点位，不自动选择正确关系。|
|动态重规划 `dynamic_replanning`|AI2-THOR 中抵达青色 C 检查点后，环境移动真实障碍封住原路；实际碰撞已验证，同时保留更长的可行绕路。代理不能搬走障碍。|导航到 C（自动触发变化）→ 观察、尝试与绕行 → 靠近目标 `MARK(object_id)` → 原地附近 `FINISH`。|

### dependency_doors：一个已运行的具体例子

**环境与布置：** `FloorPlan1` 厨房、种子 42。关闭的橱柜 `object_010` 中有酒瓶 `object_077`，需要送到台面 `object_017`。这些编号仅属于这个案例。

**先决条件如何生效：** 任务生成时实际尝试过“关柜拿酒瓶”，确认失败，再验证开柜、拿取和放置可成功；不是仅在提示词里要求先开门。场景内物品不足时，生成器会先用原生操作布置容器内容，再关闭容器。

到达相应物体前、调整到可见且 1.6 米以内后，依次执行下面的交互动作；拿起后到台面、放下后回橱柜的导航省略：

```jsonl
{"action":"OpenObject","object_id":"object_010"}
{"action":"PickupObject","object_id":"object_077"}
{"action":"PutObject","object_id":"object_017"}
{"action":"CloseObject","object_id":"object_010"}
{"action":"FINISH"}
```

注意 `PutObject` 的 `object_id` 是**目的地台面**，不是手中的酒瓶。原生开合、可见性、碰撞和单手持物限制操作；**订单顺序、结束前关好来源、空手提交**是 benchmark 的完成规则。L1/L2/L3 分别增加到 1/2/3 件订单，可以共用一个来源容器，不一定是三扇不同的门。

