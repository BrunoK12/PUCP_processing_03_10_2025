import numpy as np 
import pandas as pd
import os
import sys
import pickle


## FUNCTIONS
def txt_to_df(path,xlims=None,ylims=None,inclined=True):
    # Lists to save the data
    ids = []
    x_values   = []
    y_values   = []
    t_values   = []
    px_values  = []
    py_values  = []
    pz_values  = []
    ek_values  = []
    w_values   = []
    lev_values = []

    # accessing the .txt
    with open(path, 'r') as archivo:
        for linea in archivo:
            try:
                # Divide la línea en partes usando el espacio como separador
                partes = linea.split()

                # Extrae los valores que contienen 'x=', 'y=', 't=', etc.
                id_valor = int(partes[1])
                x_valor = float(partes[2].split('=')[1])/(100)   #en metros
                y_valor = float(partes[3].split('=')[1])/(100)   #en metros          
                t_valor = float(partes[4].split('=')[1])
                px_valor = float(partes[5].split('=')[1])
                py_valor = float(partes[6].split('=')[1])
                pz_valor = float(partes[7].split('=')[1])
                if inclined==True:
                    x_valor,y_valor= (-y_valor),x_valor
                    px_valor,py_valor= (-py_valor),px_valor
                    pz_valor=-pz_valor
                    #Now Y means upwards the inclined plane and X means to the right 
                ek_valor = float(partes[8].split('=')[1])
                w_valor = float(partes[9].split('=')[1])
                lev_valor = int(partes[10].split('=')[1])

                #if (det_X_inf<=x_valor<=det_X_sup) and (det_Y_inf<=y_valor<=det_Y_sup):
                    # Agrega los valores a las listas
                ids.append(id_valor)
                x_values.append(x_valor)
                y_values.append(y_valor)
                t_values.append(t_valor)
                px_values.append(px_valor)
                py_values.append(py_valor)
                pz_values.append(pz_valor)
                ek_values.append(ek_valor)
                w_values.append(w_valor)
                lev_values.append(lev_valor)
            except:
                pass

    # Crea un DataFrame de Pandas
    data = {
        'id': ids,
        'x': x_values,
        'y': y_values,
        't': t_values,
        'px': px_values,
        'py': py_values,
        'pz': pz_values,
        'ek': ek_values,
        'w': w_values,
        'lev': lev_values,
        'detector': np.nan
    }

    all_data = pd.DataFrame(data).astype({'detector':object})
    if xlims != None:
        all_data = all_data[(all_data['x']>= xlims[0]) & (all_data['x']<= xlims[1])].reset_index(drop=True)
    if ylims != None:
        all_data = all_data[(all_data['y']>= ylims[0]) & (all_data['y']<= ylims[1])].reset_index(drop=True)

    return all_data

def list_dats(path):
    dat_list = []
    for name in os.listdir(path):
        full_path = os.path.join(path, name)
        if full_path[-4:]=='.txt':
            dat_list.append(name)
    return dat_list

def list_directories(parent_directory):
    directories = [d for d in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, d))]
    return directories

def get_shower_info(nombre_archivo):
    valores_encontrados = {}

    palabras_clave = ["PRMPAR = ", "PRME = ", "THETAP = ", "PHIP = "]

    # Abre el archivo y lo lee línea por línea
    with open(nombre_archivo, 'r') as archivo:
        for linea in archivo:
            for palabra in palabras_clave:
                # Busca la palabra clave seguida de un número
                if palabra in linea:
                    # Encuentra el número al costado de la palabra clave
                    indice_palabra = linea.index(palabra)
                    inicio_numero = indice_palabra + len(palabra)
                    
                    # Busca el número después de la palabra clave
                    numero = ""
                    for char in linea[inicio_numero:]:
                        if char.isdigit() or char == '.' or char == '-':
                            numero += char
                        else:
                            break
                    
                    if numero:
                        valores_encontrados[palabra[:-3]] = float(numero) if '.' in numero else int(numero)
                        
    return valores_encontrados

def filter_geometry(all_particles_df, allowed_particles, length_triangle, a, b):
    mask_charged = all_particles_df['id'].isin(allowed_particles)
    mask_d0 = ((all_particles_df['x'] >=  (- b / 2)) & 
               (all_particles_df['x'] <=  (+ b / 2)) & 
               (all_particles_df['y'] >=  (- a / 2)) & 
               (all_particles_df['y'] <=  (+ a / 2)))

    mask_d1 = ((all_particles_df['x'] >= (- b / 2)) & 
               (all_particles_df['x'] <= (+ b / 2)) & 
               (all_particles_df['y'] >= ((length_triangle*np.sqrt(3)/3) - a / 2)) & 
               (all_particles_df['y'] <= ((length_triangle*np.sqrt(3)/3) + a / 2)))
    
    mask_d2 = ((all_particles_df['x'] >= ((-length_triangle/2) - b / 2)) & 
               (all_particles_df['x'] <= ((-length_triangle/2) + b / 2)) & 
               (all_particles_df['y'] >= ((-length_triangle*np.sqrt(3)/6) - a / 2)) & 
               (all_particles_df['y'] <= ((-length_triangle*np.sqrt(3)/6) + a / 2)))
    
    mask_d3 = ((all_particles_df['x'] >= ((length_triangle/2) - b / 2)) & 
               (all_particles_df['x'] <= ((length_triangle/2) + b / 2)) & 
               (all_particles_df['y'] >= ((-length_triangle*np.sqrt(3)/6) - a / 2)) & 
               (all_particles_df['y'] <= ((-length_triangle*np.sqrt(3)/6) + a / 2)))

    all_particles_df['Detector'] = np.nan
    all_particles_df.loc[mask_charged & mask_d0, 'Detector'] = 0
    all_particles_df.loc[mask_charged & mask_d1, 'Detector'] = 1
    all_particles_df.loc[mask_charged & mask_d2, 'Detector'] = 2
    all_particles_df.loc[mask_charged & mask_d3, 'Detector'] = 3

    filtered_df = all_particles_df[all_particles_df['Detector'] != np.nan][['id', 't', 'Detector','x','y','ek']].reset_index(drop=True)
    df_detector_0 = filtered_df[filtered_df['Detector'] == 0].reset_index(drop=True)
    df_detector_1 = filtered_df[filtered_df['Detector'] == 1].reset_index(drop=True)
    df_detector_2 = filtered_df[filtered_df['Detector'] == 2].reset_index(drop=True)
    df_detector_3 = filtered_df[filtered_df['Detector'] == 3].reset_index(drop=True)

    return df_detector_0, df_detector_1, df_detector_2, df_detector_3

## INDUCE BIAS
def create_bias(max_bias):
    theta = np.random.uniform(0, 2*np.pi)
    r = max_bias*np.sqrt(np.random.uniform(0, 1))
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    return x,y

def reset_time(df0,df1,df2,df3):
    if len(df1) + len(df2) + len(df3) + len(df0) <=1:
        return df0,df1,df2,df3
    t1min= df1["t"].min()
    t2min= df2["t"].min()
    t3min= df3["t"].min()
    t0min= df0["t"].min()
    tmin= np.nanmin([t1min,t2min,t3min,t0min])
    df3['t']=df3['t']-tmin
    df2['t']=df2['t']-tmin
    df1['t']=df1['t']-tmin
    df0['t']=df0['t']-tmin
    return df0,df1,df2,df3


## PROCESS_DATA FUNCTION
def process_data(txt_path,length_triangle,a,b,allowed_particles,max_bias,x_lim=None,y_lim=None):
    ## TXT to Dataframe
    all_particles_df=txt_to_df(txt_path,inclined=False)
    if (x_lim is not None and y_lim is not None):
        all_particles_df_limited = txt_to_df(txt_path, x_lim, y_lim, inclined=False)
    else:
        all_particles_df_limited = None

    x_bias,y_bias=create_bias(max_bias)
    all_particles_df['x']=all_particles_df['x'] + x_bias
    all_particles_df['y']=all_particles_df['y'] + y_bias

    shower_info=get_shower_info(txt_path)
    df_det0,df_det1,df_det2,df_det3=filter_geometry(all_particles_df,allowed_particles,length_triangle,a,b)
    df_det0,df_det1,df_det2,df_det3=reset_time(df_det0,df_det1,df_det2,df_det3)
    return shower_info,df_det1,df_det2,df_det3,df_det0,all_particles_df_limited


def rotating_showers_bruno2(txt_path, length_triangle, a, b, allowed_particles, max_bias,n_divisions,x_lim=None, y_lim=None):
    all_particles_df = txt_to_df(txt_path, inclined=False)
    all_particles_df_limited = (txt_to_df(txt_path, x_lim, y_lim, inclined=False) if (x_lim is not None and y_lim is not None) else None)

    bx, by = create_bias(max_bias)
    base = all_particles_df.copy()
    base["x"] += bx
    base["y"] += by

    def rot2d(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s], [s,  c]], dtype=float)

    def apply_rot_xy(df, theta):
        out = df.copy()
        R = rot2d(theta)
        xy = out[["x","y"]].to_numpy()
        out[["x","y"]] = xy @ R.T
        return out

    angles = np.deg2rad(np.linspace(0,360,n_divisions + 1)[:-1])

    shower_info = get_shower_info(txt_path)
    phi0_raw = shower_info.get("phi", shower_info.get("PHIP"))
    if phi0_raw is None:
        raise KeyError("phi/PHIP not found in shower_info")
    phi0 = np.deg2rad(phi0_raw) if phi0_raw > 2*np.pi else float(phi0_raw)

    def wrap(phi):
        return (phi + 2*np.pi) % (2*np.pi)

    det0_list, det1_list, det2_list, det3_list = [], [], [], []
    phi_rots = []  

    for k, ang in enumerate(angles):
        dfr = base if ang == 0.0 else apply_rot_xy(base, ang)

        d0, d1, d2, d3 = filter_geometry(dfr, allowed_particles, length_triangle, a, b)
        d0, d1, d2, d3 = reset_time(d0, d1, d2, d3)

        new_phi = wrap(phi0 + ang)
        phi_rots.append(new_phi)

        for d in (d0, d1, d2, d3):
            d["rot_idx"] = k
            d["rotation_rad"] = ang
            d["phi"] = new_phi

        det0_list.append(d0); det1_list.append(d1)
        det2_list.append(d2); det3_list.append(d3)

    for i, phi_i in enumerate(phi_rots):
        shower_info[f"phi_{i}"] = phi_i

    df_det0 = pd.concat(det0_list, ignore_index=True)
    df_det1 = pd.concat(det1_list, ignore_index=True)
    df_det2 = pd.concat(det2_list, ignore_index=True)
    df_det3 = pd.concat(det3_list, ignore_index=True)

    return shower_info, df_det1, df_det2, df_det3, df_det0, all_particles_df_limited




#MODIFICADO PARA OBTENER LA DATA PURA SIN PROCESAR DE CORSIKA

if __name__ == "__main__":
    ## USER
    print('------------ USER SELECTION ------------\n')
    users=['bruno']
    print(f"1| Bruno\n ")
    user_id=int(input("Enter user ID: "))
    user=users[user_id-1]
    print('')
    ## DETECTOR PARAMETERS
    print('----------- ARRAY PARAMETERS -----------\n')
    a = float(input('Enter the width of the detector (m): '))
    b = float(input('Enter the lenght of the detector (m): '))
    length_triangle= float(input('Enter the separation of the detectors (m): '))
    limited_data = 'n'
    limited_data = input('Limited pure data required? [y/n], default [n]: ')
    if limited_data not in ['y', 'n']:
        print("Error: Please enter 'y' or 'n'"); sys.exit(1)

    if limited_data=='y':
        x_lim = input('Enter the limit in x (x0,x1): ')
        y_lim = input('Enter the limit in y (y0,y1): ')
        try:
            x0, x1 = map(float, x_lim.split(','))
            y0, y1 = map(float, y_lim.split(','))
            x_lim = (x0, x1)
            y_lim = (y0, y1)
        except ValueError:
            print('Error: Values must be separate ')
    if limited_data=='n':
        pass
    rot = input("Use rotated showers for detectors-only data? [y/n]: ").strip().lower()
    if rot in ("y", "yes"):
        rotation = True
        n_divisions = int(input("How many divisions (Ex: 4 (0°,90°,180°,270°)): "))
    elif rot in ("n", "no"):
        rotation = False
        pass
    else:
        print("Error: please enter 'y' or 'n'."); sys.exit(1)
    
    max_bias= float(input('Enter the maximum bias for the center of the shower (m): '))
    allowed_particles=(1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 19, 21, 23, 24, 27, 29, 31, 32, 52, 53, 54, 55, 57, 58, 59, 61, 63, 64, 117, 118, 120, 121, 124, 125, 127, 128, 131, 132, 137, 138, 140, 141, 143, 149, 150, 152, 153, 155, 161, 162, 171, 172, 177, 178, 182, 183, 185, 186, 188, 189, 191, 192, 194, 195)
    print("")
    ## DATA FILES PARAMETERS
    print('------------ DATA SELECTION ------------\n')
    #parent_directory = r'C:\Users\cg_h2\Documents\pucp_array\data'
    parent_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)),'data')
    #print found directories
    directories= list_directories(parent_directory)
    print('Avaliable directories:\n')
    for n,d in zip(list(range(1,len(directories)+1)),directories):
        print(f"{n}| {d} ")
    print("")
    sim_dir_ids=input('Enter the IDs of the directories to process (separated by comas \',\' ): ')
    sim_dir_ids=sim_dir_ids.split(',')
    sim_dir_ids=[int(dir_id) for dir_id in sim_dir_ids]
    sim_dirs=[directories[id-1] for id in sim_dir_ids]
    print("")
    print('Selected directories: ')
    print(sim_dirs)
    print("")
    confirm_dirs=input('confirm_dirs [y/n]')
    if confirm_dirs=='y':
        pass
    else:
        sim_dirs=[]
    print("")
    for sim_dir in sim_dirs:    #sim_dirs son los directorios de cada simulación, y sim_dir es la simulación escogida
        print(f'{sim_dir} directory is being processed')
        data_directory=os.path.join(parent_directory,sim_dir)
        ## DATA PROCESSING
        dat_list= list_dats(data_directory) #dat_list es la lista de las lluvias de sim_dir
        print(f"{len(dat_list)} DAT files found")
        exceptions=[]
        count=0
        all_showers_data=[]
        all_particles_dfs = []
        # PROCESS_DATA TRABAJA POR LLUVIA
        for dat in dat_list:  # se recorre cada lluvia en el dat_list
            print(f'{dat} is being processed.')
            try:
                dat_path= os.path.join(data_directory, dat)
                ## call process_data function
                if limited_data=='y':
                   if rotation:
                       shower_info,df_det1,df_det2,df_det3,det_0,all_particles_df=rotating_showers_bruno2(dat_path,length_triangle,a,b,allowed_particles,max_bias,n_divisions,x_lim,y_lim)
                   else: 
                       shower_info,df_det1,df_det2,df_det3,det_0,all_particles_df=process_data(dat_path,length_triangle,a,b,allowed_particles,max_bias,x_lim,y_lim)

                if limited_data=='n':
                    if rotation:
                       shower_info,df_det1,df_det2,df_det3,det_0,all_particles_df=rotating_showers_bruno2(dat_path,length_triangle,a,b,allowed_particles,max_bias,n_divisions)
                    else:
                        shower_info,df_det1,df_det2,df_det3,det_0,all_particles_df=process_data(dat_path,length_triangle,a,b,allowed_particles,max_bias)
                    
                ## append dataframes and shower info
                shower_summary=(shower_info,df_det1,df_det2,df_det3,det_0)
                complete_data=(shower_info,all_particles_df)
                all_showers_data.append(shower_summary)
                all_particles_dfs.append(complete_data)
                print(f'\n{dat} successful.')
            except Exception as e :
                print(f'\n{dat} Failed.')
                exceptions.append((dat,e))
            count+=1
            left=len(dat_list)-count
            print(f'{left} dats remaining')
        print('Data has been processed')
        print(f'The following exceptions have been encountered: \n{exceptions}')

        ## DATA SAVING
        if not exceptions:
            save_flag = 'y'
        else:
            save_flag = input('Save data?[y/n]: ')

        if not save_flag=='n':
            pickle_rel_dir=user+'/pickles'
            pickle_dir_path=os.path.join(os.path.dirname(os.path.realpath(__file__)),pickle_rel_dir)
            ## declare pickle file paths. pimaries_path=os.path.join(pickle_path, 'primaries.pickle')

            pickle_name=sim_dir
            pickle_name=pickle_name+'_'+str(length_triangle)+'m_'+'.pickle'
            pickle_file_path=os.path.join(pickle_dir_path,pickle_name)
            with open(pickle_file_path, 'wb') as file:
                pickle.dump(all_showers_data, file)
            print(f'pickle file saved as {pickle_file_path}')
            if limited_data=='y':
                pure_data_pickle = "PURE_DATA"+'_'+ sim_dir +'.pickle'
                pickle_pure_data_path=os.path.join(pickle_dir_path,pure_data_pickle)
                with open(pickle_pure_data_path, 'wb') as file:
                    pickle.dump(all_particles_dfs, file)
                print(f'pickle file saved as {pickle_pure_data_path}')
        else:
            print('Not saved')
    input('Press Enter to continue')
