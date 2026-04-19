from PIL import Image
try:
    print('Original:', Image.open('c:/Users/giris/Desktop/Elephant_Output/Unknown_1/Makhna_17_DSCN6949.JPG').size)
    print('Crop:', Image.open('c:/Users/giris/Desktop/Elephant_Output/Unknown_1/.crops/Makhna_17_DSCN6949.JPG').size)
except Exception as e:
    print(e)
