from deepface import DeepFace

result = DeepFace.represent(img_path='./assets/cara1.png')

# print(result)
# print(len(result))
# for k in result[0]:
    # print(k)

codification = result[0]['embedding']
print(codification)
