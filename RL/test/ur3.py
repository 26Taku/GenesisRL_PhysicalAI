import argparse
import numpy as np
import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False,
                        help="Enable visualization window.")
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.cpu)

    ########################## create a scene ##########################
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, -0.8, 1.2),  # Adjusted camera position for better view of UR3
            camera_lookat=(0.0, 0.0, 0.3), # Adjusted lookat to center on robot base/workspace
            camera_fov=30,
            max_FPS=60,
        ),
        show_viewer=args.vis,
        vis_options=gs.options.VisOptions(
            visualize_mpm_boundary=True,
        ),
    )

    ########################## entities ##########################
    # Add a ground plane
    plane = scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(friction=1.0), # Add friction to the plane
    )

    # Add a soft cube for grasping
    cube = scene.add_entity(  
    gs.morphs.Box(  
        size=(0.04, 0.04, 0.04),  
        pos=(0.4, 0.0, 0.02),  
  
    )  
)  

    # Load the UR3 robot with Robotiq 2F-140 gripper
    # Make sure 'ur3_robotiq.xml' is in the 'xml/' directory or provide its full path
    ur3_gripper = scene.add_entity(
        gs.morphs.URDF(file="ur3_with_robotiq_2f_140_gripper.urdf", fixed=True), # Path to your saved URDF/MJCF file
        material=gs.materials.Rigid(coup_friction=1.0), # Rigid material for the robot
    )
    
    # Add a camera for recording video
    cam = scene.add_camera(
        res=(1280, 960),
        pos=(5, 0, 1.5),
        lookat=(0.0, 0.0, 0.0),
        fov=30,
        GUI=False,
    )

    ########################## build ##########################
    scene.build()

    # Get the total number of DOFs from the robot object
    num_total_dofs = ur3_gripper.n_dofs 
    print(f"Total DOFs detected by Genesis: {num_total_dofs}\n") 

    # --- Start of diagnostic code ---
    print("--- Joint to DOF Mapping ---")
    for joint in ur3_gripper.joints:
        # joint.dof_idx can be an empty list if it's a fixed joint or a 6-element array for a free base
        print(f"Joint Name: {joint.name}, DOF Index(es): {joint.dof_idx}")
    print("--------------------------\n")

    print("--- Link Names in Genesis ---")
    for link in ur3_gripper.links:
        print(f"Link Name: {link.name}")
    print("---------------------------\n")
    # --- End of diagnostic code ---

    motors_dof = np.arange(6)
    fingers_dof = np.arange(6, 7)  
    first_qpos = np.array([-0, -0.9,  -0.5,  -1.4,  -1.3,  -0.3])  
  
    kp = np.array([4500, 4500, 3500, 3500, 2000, 2000, 100, 100, 100, 100, 100, 100,]) 
    ur3_gripper.set_dofs_kp(  
        kp = kp,  
    )
    kv = np.array([450,   450,  350,  350,  200,  200, 10, 10, 10, 10, 10, 10])
    ur3_gripper.set_dofs_kv(  
        kv = kv,  
    )
    
    finger_qpos = [0.12]
    
    end_effector = ur3_gripper.get_link("wrist_3_link")  
 
    
    cam.start_recording()
    
    gripper_open = lambda: finger_qpos.__setitem__(0, 0.0)
    gripper_close = lambda: finger_qpos.__setitem__(0, 0.7)
    
    ur3_gripper.set_qpos(first_qpos, motors_dof)  
    scene.step()
    cam.render()
    
    gripper_open()
    
    for i in range(50):  
        ur3_gripper.control_dofs_position(first_qpos[0:6], motors_dof)  
        ur3_gripper.control_dofs_position(finger_qpos, fingers_dof)  
        scene.step()
        cam.render()   

    gripper_close()
    
    for i in range(50):  
        ur3_gripper.control_dofs_position(first_qpos[0:6], motors_dof)  
        ur3_gripper.control_dofs_position(finger_qpos, fingers_dof)  
        scene.step()
        cam.render()   
    
    """
    for i in range(50):  
        ur3_gripper.control_dofs_position(first_qpos[0:6], motors_dof)  
        ur3_gripper.control_dofs_position(finger_qpos, fingers_dof)  
        scene.step()
        cam.render()   
    
    """
    
    """

    #move to pre-grasp pose  
    qpos = ur3_gripper.inverse_kinematics(  
        link=end_effector,  
        pos=np.array([0.4, 0.0, 0.40]),  
        quat=np.array([0, 1, 0, 0]),  
        dofs_idx_local = motors_dof  
    )  
    print(qpos)  
    
    for i in range(20):  
        ur3_gripper.set_dofs_position(qpos[0:6], motors_dof)  
        ur3_gripper.set_dofs_position(finger_qpos, fingers_dof)  
        scene.step()
        cam.render()  
    
    
    
    for i in range(100): 
        qpos = ur3_gripper.inverse_kinematics(  
            link=end_effector,  
            pos=np.array([0.4, 0.0, 0.40 - 0.25 * i / 100]),  
            quat=np.array([0, 1, 0, 0]),  
            dofs_idx_local = motors_dof  
        )
        print("qpos", qpos)
        ur3_gripper.control_dofs_position(qpos[0:6], motors_dof)  
        ur3_gripper.control_dofs_position(finger_qpos, fingers_dof)  
        scene.step()
        cam.render() 
    
    
    for i in range(10):  
        ur3_gripper.control_dofs_position(qpos[0:6], motors_dof)  
        ur3_gripper.control_dofs_position(finger_qpos, fingers_dof)  
        scene.step()
        cam.render() 
    
    
    gripper_close()  
    for i in range(50):  
        ur3_gripper.control_dofs_position(qpos[0:6], motors_dof)  
        ur3_gripper.control_dofs_position(finger_qpos, fingers_dof)  
        scene.step()
        cam.render()  
    
    

    
    print(qpos)  
    #qpos[5] = finger_qpos[5]  
    
    for i in range(100):  
        print("lift", i)
        qpos = ur3_gripper.inverse_kinematics(  
            link=end_effector,  
            pos=np.array([0.4, 0.0, 0.15 + 0.25 * i / 100]),  
            quat=np.array([0, 1, 0, 0]),  
            #dofs_idx_local = motors_dof  
        )
        print("qpos", qpos)
        ur3_gripper.control_dofs_position(qpos[0:6], motors_dof)  
        ur3_gripper.control_dofs_position(finger_qpos, fingers_dof)  
        scene.step()
        cam.render()
    
    """
        
    cam.stop_recording(save_to_filename='ur3_robotiq_grasp2.mp4', fps=60)
    print("Simulation complete. Video saved as 'ur3_robotiq_grasp2.mp4'.")

if __name__ == "__main__":
    main()
