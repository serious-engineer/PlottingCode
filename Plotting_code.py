import csv
import math
import sys
import os
import progressbar
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

distances=[]


#free_csv=sys.argv[1]
#est_csv=sys.argv[2]

def participant_estimate(path):
  part_no = os.path.basename(path)
  print("\nEstimate for Participant {} \n".format(part_no))
  csv_plot('FL'+ part_no +'_free.csv','FL'+ part_no +'_est.csv',path)
  csv_plot('FL'+ part_no +'_palmer.csv','FL'+ part_no +'_est.csv',path)
  csv_plot('FL'+ part_no +'_dorsal.csv','FL'+ part_no +'_est.csv',path)
  csv_plot('FL'+ part_no +'_palmeraway.csv','FL'+ part_no +'_est.csv',path)
  csv_plot('UL'+ part_no +'_free.csv','FL'+ part_no +'_est.csv',path)
  csv_plot('UL'+ part_no +'_palmer.csv','FL'+ part_no +'_est.csv',path)
  csv_plot('UL'+ part_no +'_dorsal.csv','FL'+ part_no +'_est.csv',path)
  csv_plot('UL'+ part_no +'_palmeraway.csv','FL'+ part_no +'_est.csv',path)

def csv_plot(free_csv,est_csv,path = os.getcwd()):
  global distances
  distances = []
  free_csv = os.path.join(path,free_csv)
  est_csv = os.path.join(path,est_csv)
  if not os.path.exists(free_csv):
    print("Dear user",free_csv," does not exist!!")
    return
  if not os.path.exists(est_csv):
    print("Dear user",free_csv," does not exist!!")
    return

  std_out=""
  df =pd.read_csv(free_csv)
  real = pd.read_csv(est_csv)
  #print(df)
  saorted = df.sort_values('X_value')
  #print(saorted)

  X_coor = saorted['X'].tolist()
  Y_coor = saorted['Y'].tolist()
  Target_value = saorted['X_value'].tolist()
  #print(len(Target_value))
  #print(len(X_coor))
  #print(len(Y_coor))

  x_esd = []

  for i in range(0,4):
      val = 0
      for j in range(0,5):
          val = val + X_coor[j+i*5]
          avg_val = val/5
      x_esd.append(avg_val)

  std_out += "Estimated x coordinates " + str(x_esd) + '\n'

  y_esd = []
  for i in range(0,4):
      val = 0
      for j in range(0,5):
          val = val + Y_coor[j+i*5]
          avg_val = val/5
      y_esd.append(avg_val)

  std_out += "Estimated y coordinates " + str(y_esd) + '\n'


  x_real = real['X'].tolist()
  y_real = real['Y'].tolist()
  std_out += "Real x coordinates " + str(x_real) + '\n'
  std_out += "Real y coordinates " + str(y_real) + '\n'

  plt.scatter(x_esd, y_esd, c ="blue")
  plt.scatter(x_real, y_real, c ="red")
  plt.scatter(X_coor, Y_coor, c ="green")
  # To show the plot

  connectpoints(x_esd,y_esd,0,1,'k--')
  connectpoints(x_esd,y_esd,2,3,'k--')
  connectpoints(x_esd,y_esd,0,2,'k--')
  connectpoints(x_esd,y_esd,1,3,'k--')

  connectpoints(x_real, y_real,0,1,'r-')
  connectpoints(x_real, y_real,2,3,'r-')
  connectpoints(x_real, y_real,0,2,'r-')
  connectpoints(x_real, y_real,1,3,'r-')

  plt.axis('equal')
  scatter_path = os.path.join(path,'scatter_plot' + os.path.basename(free_csv) + os.path.basename(est_csv) +'_.png')
  plt.savefig(scatter_path, dpi=1200)
  plt.close()

  # Calculate estimated and real width
  width_est = max(x_esd) - min(x_esd)
  width_real = max(x_real) - min(x_real)

  # Calculate real and estimated height
  height_est = max(y_esd) - min(y_esd)
  height_real = max(y_real) - min(y_real)


  # Display the plot
  #print(distances)
  #plt.show()
  Digits = np.arange(6)
  width = 0.4


  lengths = distances[0:4]
  real_lengths = distances[4:8]
  #print(real_lengths)

  lengths.append(width_est)
  lengths.append(height_est)
  real_lengths.append(width_real)
  real_lengths.append(height_real)


  plt.bar(Digits-0.2, lengths, width)
  plt.bar(Digits+0.2, real_lengths, width)
  plt.xticks(Digits, ['AB','CD','AC','BD','W','H'])
  plt.legend(["estimated", "Real"])
  #plt.show()

  std_out += "Estimated lengths(cm)" + str(lengths) + '\n'
  std_out += "Real lengths(cm)" + str(real_lengths) + '\n'
  barplot_path = os.path.join(path,'barplot' + os.path.basename(free_csv) + os.path.basename(est_csv) + '_.png')
  plt.savefig(barplot_path, dpi=1200)
  plt.close()

  percent = []

  for i in range(4):
      p = (lengths[i]/real_lengths[i])*100
      percent.append(p)

  std_out += "Percentage estimate(%)" + str(percent) + '\n'

  # Calculate percentage estimate for width and height
  percent_width = (width_est / width_real) * 100
  percent_height = (height_est / height_real) * 100

  std_out += "Percentage Estimate for Width(%)" + str(percent_width) + '\n'
  std_out += "Percentage Estimate for Height(%)" + str(percent_height) + '\n'

  resultFilePath= os.path.join(path,'Results' + os.path.basename(free_csv) + os.path.basename(est_csv) + '.txt')
  print("Saving the Results File in ",resultFilePath)
  with open(resultFilePath,"w") as file:
      file.write(std_out)
  file.close()

def connectpoints(X,Y,p1,p2,color):
    x1, x2 = X[p1], X[p2]
    y1, y2 = Y[p1], Y[p2]
    d = math.dist([x1,y1], [x2,y2])
#    print(x1,x2)
#    print (d)
    global distances
    distances.append(d)
    plt.plot([x1,x2],[y1,y2],color)

def main():
    #print("python main function")
    print("*"*50)
    parent_path = r'C:\Users\user\OneDrive - iitgn.ac.in\Body Model - analysis Nihaala 2024\Main_Experiment'
    print("Parent Directory - ",parent_path)
    path_list = []
    for subdir, dirs, files in os.walk(parent_path):
      if subdir != parent_path:
        path_list.append(subdir)

    for path in path_list:
      participant_estimate(path)

    print("*"*50)

if __name__ == '__main__':
    main()
