import time
import pandas as pd
from openpyxl import load_workbook
from playsound import playsound
from pyomyo import Myo, emg_mode

# Para marcar solo un t_delay y un fin
vuelta = 0


# Para guardar los csv más fácilmente
def save(myo_data, myo_cols, filepath, sheet_name):
    new_df = pd.DataFrame(myo_data, columns=myo_cols)

    with pd.ExcelWriter(filepath, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
        new_df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Data saved to {sheet_name} in {filepath}")


def play_sound():
    playsound('C:\\Windows\\Media\\Windows Ding.wav')  # Reemplaza con la ruta a tu archivo de sonido


def data_worker(mode, seconds, collect, pos_buscada):
    # Myo Setup
    m = Myo(mode=mode)
    m.connect()

    tiempo = 0
    global vuelta

    # Guardar los datos EMG, IMU y de Pose
    myo_emg_data = []
    myo_imu_data = []
    myo_pose_data = []

    last_imu_data = (0, 0, 0, 0, 0, 0, 0, 0, 0)

    def save_emg(emg, movement):
        nonlocal last_imu_data
        if tiempo != 0:
            datos = (tiempo,) + emg + last_imu_data
            myo_emg_data.append(datos)

    def print_save_battery(bat):
        print("Battery level:", bat)

    def print_save_arm(arm, xdir):
        print("Arm data:", arm, xdir)
        datos = (tiempo,) + (arm.name,)
        myo_pose_data.append(datos)

    def print_save_imu(quat, acc, gyro):
        nonlocal last_imu_data
        if tiempo != 0:
            imu = quat + acc + gyro
            last_imu_data = imu
            myo_imu_data.append((tiempo,) + imu)

    def print_save_pose(pose):
        print("Pose data:", pose)
        datos = (tiempo,) + (pose.name,)
        myo_pose_data.append(datos)

    m.add_emg_handler(save_emg)
    m.add_battery_handler(print_save_battery)
    m.add_arm_handler(print_save_arm)
    m.add_imu_handler(print_save_imu)
    m.add_pose_handler(print_save_pose)

    m.set_leds([0, 128, 0], [0, 128, 0])
    play_sound()

    print("Data Worker started to collect")
    start_time = time.perf_counter()
    print("start_time:", start_time)

    while collect:
        if time.perf_counter() - start_time < seconds:
            tiempo = time.perf_counter() - start_time

            if tiempo % 10 < 0.1:
                if vuelta == 1 and tiempo > 1:
                    play_sound()
                    print(f"Termina {pos_buscada}, tiempo: {tiempo}")
                    if vuelta != 2:
                        vuelta = 0

            elif tiempo % 5 < 0.1 and vuelta == 0 and tiempo > 1:
                play_sound()
                print(f"Comienza {pos_buscada}, tiempo: {tiempo}")
                if vuelta != 2:
                    vuelta = 1
            m.run()
        else:
            collect = False
            vuelta = 2
            collection_time = time.perf_counter() - start_time
            print(f"Finished collecting.")
            play_sound()
            m.disconnect()

    print(f"Collection time: {collection_time}")
    return myo_emg_data, myo_imu_data, myo_pose_data


def combine_data(emg_df, imu_df):
    combined_df = emg_df.copy()
    imu_df.sort_values(by='Tiempo', inplace=True)

    for col in ["quat1", "quat2", "quat3", "quat4", "acc1", "acc2", "acc3", "gyro1", "gyro2", "gyro3"]:
        combined_df[col] = None

    imu_index = 0
    for i, emg_row in combined_df.iterrows():
        while imu_index < len(imu_df) and imu_df.iloc[imu_index]['Tiempo'] <= emg_row['Tiempo']:
            imu_index += 1
        if imu_index > 0:
            closest_imu_row = imu_df.iloc[imu_index - 1]
            for col in ["quat1", "quat2", "quat3", "quat4", "acc1", "acc2", "acc3", "gyro1", "gyro2", "gyro3"]:
                combined_df.at[i, col] = closest_imu_row[col]

    return combined_df


if __name__ == '__main__':
    gestos = 3
    seconds = gestos * 10 + 5

    arm = "de_"
    pos_brazo = "Estirado"
    pose = "UNO"
    num = "_1"

    name = "datos_jupyter/" + pos_brazo + "/" + pose + "/"
    file_name = str(name) + arm + pose + num + ".xlsx"

    mode = emg_mode.FILTERED

    emg, imu, pose = data_worker(mode, seconds, True, pose)

    print(len(emg), "frames collected")

    myo_emg_cols = ["Tiempo", "Channel_1", "Channel_2", "Channel_3", "Channel_4", "Channel_5", "Channel_6",
                    "Channel_7", "Channel_8", "quat1", "quat2", "quat3", "quat4", "acc1", "acc2", "acc3", "gyro1",
                    "gyro2", "gyro3"]
    myo_imu_cols = ["Tiempo", "quat1", "quat2", "quat3", "quat4", "acc1", "acc2", "acc3", "gyro1", "gyro2", "gyro3"]
    myo_pose_cols = ["Tiempo", "pose"]

    emg_df = pd.DataFrame(emg, columns=myo_emg_cols)
    imu_df = pd.DataFrame(imu, columns=myo_imu_cols)
    pose_df = pd.DataFrame(pose, columns=myo_pose_cols)

    combined_df = combine_data(emg_df, imu_df)

    save(emg_df[
             ["Tiempo", "Channel_1", "Channel_2", "Channel_3", "Channel_4", "Channel_5", "Channel_6", "Channel_7",
              "Channel_8"]],
         ["Tiempo", "Channel_1", "Channel_2", "Channel_3", "Channel_4", "Channel_5", "Channel_6", "Channel_7",
          "Channel_8"], file_name, "EMG")
    save(imu_df, myo_imu_cols, file_name, "IMU")
    save(pose_df, myo_pose_cols, file_name, "Pose")
    save(combined_df, myo_emg_cols, file_name, "Datos")
