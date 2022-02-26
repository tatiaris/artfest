from PIL import Image

img = Image.open(input('enter img path: '))
img = img.convert("RGBA")
output_file = open(input('name of output file: ') + '.txt', 'w+')

pixdata = img.load()

for i in range(img.height):
    for j in range(img.width):
        coordinate = j, i
        pixel_rgb = img.getpixel(coordinate)
        r = pixel_rgb[0]
        g = pixel_rgb[1]
        b = pixel_rgb[2]
        a = pixel_rgb[3]
        avg = int((r+g+b)/3)
        if (avg < 20):
            output_file.write(' ')
        elif (avg < 40):
            output_file.write('.')
        elif (avg < 60):
            output_file.write(',')
        elif (avg < 80):
            output_file.write(':')
        elif (avg < 100):
            output_file.write(';')
        elif (avg < 120):
            output_file.write('|')
        elif (avg < 140):
            output_file.write('?')
        elif (avg < 160):
            output_file.write('%')
        elif (avg < 180):
            output_file.write('#')
        elif (avg < 200):
            output_file.write('@')
        elif (avg < 220):
            output_file.write(' ')
        elif (avg < 240):
            output_file.write(' ')
        elif (avg < 260):
            output_file.write(' ')
        elif (avg < 280):
            output_file.write('U')
        elif (avg < 300):
            output_file.write('S')
        elif (avg < 340):
            output_file.write('0')
        else:
            output_file.write('B')
    output_file.write('\n')
