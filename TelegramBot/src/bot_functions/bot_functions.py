from data.meals_pfc import food_nutrition

from ultralytics import YOLO

import os
import torch
from PIL import Image


IOU_THRESHOLD = 0.5
SCORE_THRESHOLD = 0.2

async def wakeUp(update, context):
    chat = update.effective_chat
    await context.bot.send_message(
        chat_id=chat.id, 
        text=f'Привет, {chat.first_name}! \nЯ бот для расчета БЖУ и каллорий по фото! Отправь мне фото твоего блюда и я попробую угадать, сколько в нем белков, жиров и углеводов!')


async def sayHi(update, context):
    chat = update.effective_chat

    photo = update.message.photo[-1]
    file = await photo.get_file()

    if not os.path.exists('data/images'):
        os.makedirs('data/images')

    file_path = 'data/images/{name}.jpg'
    with open(file_path.format(name='base'), 'wb') as f:
        await file.download_to_memory(f)

    # Preprocess the image: resize to 640x640
    with Image.open(file_path.format(name='base')) as img:
        img_resized = img.resize((640, 640))
        img_resized.save(file_path.format(name='resized'))

    yolo_model = YOLO("../FoodScanner/yolo11_pretrained_last/weights/best.pt").to('mps')
    with torch.no_grad():
        preds = yolo_model.predict([file_path.format(name='resized')], conf= SCORE_THRESHOLD, iou=IOU_THRESHOLD)
        for pred in preds:
            boxes = torch.tensor(pred.boxes.xyxy, dtype=torch.float32)
            scores = torch.tensor(pred.boxes.conf, dtype=torch.float32)
            labels = torch.tensor(preds[0].boxes.cls, dtype=torch.int64)


    labels = list(set([yolo_model.names[label.cpu().item()] for label in preds[0].boxes.cls]))
    if len(labels) == 0:
        text = 'Sorry, right now I don\'t know what is in the photo'
    else:
        text = f'In the photo I see {", ".join(labels)}\n\n'
        text += '--- P/F/C and calories (100g) ---\n'

        proteins = 0
        fats = 0
        carbohydrates = 0
        calories = 0

        weight = 2

        for label in labels:
            current = food_nutrition[label]
            text += label + ' - ' + str(current['proteins']) + 'g / ' + str(current['fats']) + 'g / ' + str(current['carbohydrates']) + 'g / ' + str(current['calories']) + 'kcal\n'
            proteins += float(current['proteins']) * weight
            fats += float(current['fats']) * weight
            carbohydrates += float(current['carbohydrates']) * weight
            calories += float(current['calories']) * weight
        
        text += '--- Total ---\n'
        text += f'Proteins: {proteins:.2f}g\nFats: {fats:.2f}g\nCarbohydrates: {carbohydrates:.2f}g\nCalories: {calories}kcal'

    await context.bot.send_message(chat_id=chat.id, text=text)
