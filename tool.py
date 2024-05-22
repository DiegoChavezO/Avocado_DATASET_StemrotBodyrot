#Instalar el paquete para remover el fondo.
!pip install rembg

#Importar bibliotecas
import cv2
import math
import numpy as np
from rembg import remove
from PIL import Image
import matplotlib.pyplot as plt
#DSC_0233 copia.JPG DSC_0149.JPG

#Obtener imagen
path = '/content/drive/MyDrive/TT/Stem end rot/'
# Se carga una imagen a escala de grises
img = cv2.imread(path+"DSC_0814.JPG")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#Removemos el fondo
img_remove = remove(img)

#Filtro Gaussiano
img_gaussiana = cv2.GaussianBlur(img_remove, (15,15), 0)

#Escala de grises
img_gris = cv2.cvtColor(img_gaussiana, cv2.COLOR_BGR2GRAY)

#Umbralización area del fruto
ret, th = cv2.threshold(img_gris, 1, 255, cv2.THRESH_BINARY)

#Contar pixeles blancos del área del fruto
pixeles_blancos = np.where(th == 255, 1, 0)
numero_pixeles_blancos = pixeles_blancos.sum()

img_height, img_width, channels = img.shape
total_pixels = img_height * img_width
print("Total de pixeles = ",total_pixels)
print("Total de pixeles blancos = ", numero_pixeles_blancos)

#Dibujar contorno
#Algoritmo de Canny
canny_img_ext = cv2.Canny(th, 5, 15)
img_contCanny = th.copy()
cont, ret = cv2.findContours(canny_img_ext, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img_contCanny, cont, -1, (0,0,0), 200)

#Operación OR
op_or_ext = cv2.bitwise_and(img_gaussiana , img_gaussiana, mask=img_contCanny)

#Segmetación por color
# Se seleccionan los rangos de HSV
#stem end rot
rango_min = np.array([18, 0, 0], np.uint8)
rango_max = np.array([23, 255, 255], np.uint8)

#body rot
#rango_min = np.array([18, 0, 0], np.uint8)
#rango_max = np.array([25, 255, 255], np.uint8)

# Lectura de la imagen y conversión a HSV
imagen_HSV = cv2.cvtColor(op_or_ext, cv2.COLOR_RGB2HSV)

# Creación de la mascara de color
mascara = cv2.inRange(imagen_HSV, rango_min, rango_max)
segmentada_color = cv2.bitwise_and(img, img, mask=mascara)

#Umbralización el area dañanda
img_gris_dañada = cv2.cvtColor(segmentada_color, cv2.COLOR_BGR2GRAY)
ret, areAfectada = cv2.threshold(img_gris_dañada, 1, 255, cv2.THRESH_BINARY)

print(areAfectada.shape)
#Calculó del porcentaje del área dañada

#Contar pixeles blancos del área dañada
pixeles_blancos_dañados = np.where(areAfectada == 255, 1, 0)
numero_pixeles_blancos_dañados = pixeles_blancos_dañados.sum()

areaDañada = ((numero_pixeles_blancos_dañados*100)/numero_pixeles_blancos)
print("Total de pixeles dañados = ", numero_pixeles_blancos_dañados)


areAfectada_3canales = cv2.cvtColor(areAfectada, cv2.COLOR_GRAY2RGB)
#Operación OR
op_or = cv2.bitwise_or(img, areAfectada_3canales)
plt.title('OR')
plt.imshow(op_or)
plt.show()

print(f"Porcentaje área dañada = {areaDañada:.2f} %")