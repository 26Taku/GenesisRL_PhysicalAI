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
    plane = scene.add_entity(  
        gs.morphs.Plane(),  
    )  
    cube = scene.add_entity(  
        gs.morphs.Box(  
            size=(0.04, 0.04, 0.04),  
            pos=(0.65, 0.0, 0.02),  
    
        )  
    )  
    
    ur = scene.add_entity(  
        gs.morphs.URDF(file='ur5/ur5_robotiq85.urdf', fixed=True),  
    )  
    
    cam = scene.add_camera(  
        res=(640, 480),  
        pos=(3, -1, 1.5),  
        lookat=(0, 0, 0.5),  
        fov=30,  
        GUI=False  
    )  
    
    finger_qpos = [0.00, ]  
    
    def gripper_open():  
        global finger_qpos  
        finger_qpos = [0.12, ]  
    
    def gripper_close():  
        global finger_qpos  
        finger_qpos = [0.55, ]  
        
    ########################## build ##########################  
    scene.build()  
    
    motors_dof = np.arange(6)  
    fingers_dof = np.arange(6, 7)  
    first_qpos = np.array([  -0, -0.9,  -0.5,  -1.4,  -1.3,  -0.3, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04])  
    dofs_idx = list(np.arange(12))  
    
    # from franka_cube.py  
    kp = np.array([4500, 4500, 3500, 3500, 2000, 2000, 100, 100, 100, 100, 100, 100,])  
    kv = np.array([450,   450,  350,  350,  200,  200, 10, 10, 10, 10, 10, 10])  
    
    ur.set_dofs_kp(  
        kp,  
        dofs_idx  
    )  
    
    ur.set_dofs_kv(  
        kv,  
        dofs_idx  
    )  
    
    ur.set_dofs_force_range(  
        np.array([-87, -87, -87, -87, -87, -87, -12, -12, -12, -100, -100, -100]),  
        np.array([87, 87, 87, 87, 87, 87, 12, 12, 12, 100, 100, 100]),  
        dofs_idx  
    )  
    
    ur.set_qpos(first_qpos)  
    
    cam.start_recording()
    
    

    gripper_open()  
    finger_qpos = [0.12, ]  
    for i in range(50):  
        ur.control_dofs_position(first_qpos[0:6], motors_dof)  
        ur.control_dofs_position(finger_qpos, fingers_dof)  
        scene.step()  
        cam.render()
    
    gripper_close() 
    finger_qpos = [0.55, ]  
    for i in range(50):  
        ur.control_dofs_position(first_qpos[0:6], motors_dof)  
        ur.control_dofs_position(finger_qpos, fingers_dof)  
        scene.step()  
        cam.render()
    
    
    gripper_open()  
    finger_qpos = [0.12, ]  
    for i in range(50):  
        ur.control_dofs_position(first_qpos[0:6], motors_dof)  
        ur.control_dofs_position(finger_qpos, fingers_dof)  
        scene.step()  
        cam.render()
        
    cam.stop_recording(save_to_filename='ur5_demo.mp4')#, fps=60)
    print("Simulation complete. Video saved as 'ur5_demo.mp4'.")

if __name__ == "__main__":
    main()