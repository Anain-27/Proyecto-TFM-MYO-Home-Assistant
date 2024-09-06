# Simplistic data recording
import time
import multiprocessing
import numpy as np
import pandas as pd

from pyomyo import Myo, emg_mode

# Ana: para marcar solo un t_delay y un fin
vuelta = True

#Para guardar los csv más facilmente
def save(myo_data,myo_cols,filepath):
	# Add columns and save to df
	myo_df = pd.DataFrame(myo_data, columns=myo_cols)
	myo_df.to_excel(filepath, index=False)
	print("CSV Saved at: ", filepath)


def data_worker(mode, seconds, filepath):
	collect = True

	# ------------ Myo Setup ---------------
	m = Myo(mode=mode)
	m.connect()

	tiempo = 0

	#Ana: Guardo los datos EMG
	myo_emg_data = []


	# Ana: Guardo los demás datos
	myo_data = []

	def save_emg(emg, movement):
		global vuelta
		datos= (tiempo,)+ emg + (0,0,0,0,0,0,0,0,0,0,0)

		#print("EMG:", emg)

		# Ana: Para saber cuando comenzar a hacer el gesto, en principio, lo haremos 3 veces.
		# Pero lo vamos a implementar por si queremos más lo marcaremos cada 5 segundos el comenzar y terminar gestos
		if tiempo % 10 < 0.1:
			if not vuelta and tiempo > 1:
				m.vibrate(1)
				print("Termina")
				vuelta = True
		elif tiempo % 5 <0.1 and vuelta and tiempo >1:
			m.vibrate(1)
			print("Comienza")
			vuelta=False


		# Ana: Pintamos los datos una vez reestructurados
		#print(datos)
		# Ana: Además de pintarlo lo guardamos
		myo_data.append(datos)
		myo_emg_data.append(emg)

	m.add_emg_handler(save_emg)

	#Ana: Pintamos y guardamos
	def print_save_battery(bat):
		print("Battery level:", bat)

		#Ana: Además de pintarlo lo guardamos
		#myo_data.append(bat)
	def print_save_arm(arm,xdir):
		print("Arm data:", arm, xdir)
		datos = (tiempo,) + (0,0,0,0,0,0,0,0,) + (0, 0, 0, 0, 0, 0, 0, 0, 0,0,) + (arm.name,)

		#Ana: Pintamos los datos una vez reestructurados
		#print(datos)

		#Ana: Además de pintarlo lo guardamos
		myo_data.append(datos)

	def print_save_imu(quat, acc, gyro):
		#print("IMU data:", quat, acc, gyro)
		imu=quat+acc+gyro
		datos = (tiempo,) + (0, 0, 0, 0, 0, 0, 0, 0,) + imu + (0,)

		# Ana: Pintamos los datos una vez reestructurados
		# print(datos)

		#Ana: Además de pintarlo lo guardamos
		myo_data.append(datos)

	def print_save_pose(pose):
		print("Pose data:", pose)
		datos = (tiempo,) + (0, 0, 0, 0, 0, 0, 0, 0,) + (0, 0, 0, 0, 0, 0, 0, 0, 0, 0,) + (pose.name,)

		# Ana: Pintamos los datos una vez reestructurados
		# print(datos)

		# Ana: Además de pintarlo lo guardamos
		myo_data.append(datos)


	# Ana: Añadimos los demás handlers para captar todos los datos
	m.add_battery_handler(print_save_battery)
	m.add_arm_handler(print_save_arm)
	m.add_imu_handler(print_save_imu)
	m.add_pose_handler(print_save_pose)

	 # Its go time
	m.set_leds([0, 128, 0], [0, 128, 0])
	# Vibrate to know we connected okay
	m.vibrate(1)

	print("Data Worker started to collect")
	# Start collecing data.
	start_time = time.perf_counter()

	print("start_time:",start_time)

	while collect:
		#Ana: Por si queremos ir pintando los tiempos
		#print(time.perf_counter() - start_time)
		if time.perf_counter() - start_time < seconds:
			m.run()
			tiempo= time.perf_counter()- start_time
		else:
			collect = False
			collection_time = time.perf_counter() - start_time
			print("Finished collecting.")
			print(f"Collection time: {collection_time}")
			print(len(myo_emg_data), "frames collected")

			# Add columns and save to df
			myo_emg_cols = ["Tiempo","Channel_1", "Channel_2", "Channel_3", "Channel_4", "Channel_5", "Channel_6", "Channel_7", "Channel_8"]

			myo_imu_cols = ["quat1","quat2","quat3","quat4", "acc1","acc2","acc3", "gyro1","gyro2","gyro3"]
			myo_pose_cols = ["pose"]
			cols= myo_emg_cols+myo_imu_cols+myo_pose_cols

			save(myo_data, cols, filepath)


# -------- Main Program Loop -----------
if __name__ == '__main__':
	#Ana: tiempo de captación
	gestos=3
	seconds= gestos*10 +5

	#Ana: Añado el nombre arm para ponerlo en el guardado de datos
	arm="de_"
	pos_brazo= "Estirado"
	pose="FIST"
	num = "_1"

	#Ana: Guardar los datos en carpetas
	name = "datos_jupyter/" +pos_brazo + "/" + pose + "/"
	file_name = str(name)+arm+pose+num+".xlsx"

	#Ana: Guardar los datos para pruebas
	#file_name = "datos/data_prueba_todos.xlsx"

	#Cambiado para ver el raw
	mode = emg_mode.FILTERED
	p = multiprocessing.Process(target=data_worker, args=(mode, seconds, file_name))
	p.start()
