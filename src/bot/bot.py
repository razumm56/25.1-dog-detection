import os
import asyncio
import cv2
from telegram import Bot

class DogAlertBot:
    def __init__(self, token, chat_id):
        self.bot = Bot(token=token)
        self.chat_id = chat_id

    async def push_notificate(self, dog_count, frame=None):
            message = f"Внимание! На фото есть собаки"

            # Если передан кадр (frame), конвертируем его в формат, подходящий для отправки
            if frame is not None:
                # Конвертируем кадр в байты (в формате JPEG)
                _, buffer = cv2.imencode('.jpg', frame)
                photo_bytes = buffer.tobytes()

                # Отправляем фото с подписью
                await self.bot.send_photo(
                    chat_id=self.chat_id,
                    photo=photo_bytes,
                    caption=message
                )
            else:
                # Если кадр не передан, отправляем только текст
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=message
                )

if __name__ == "__main__":
    TOKEN = os.environ.get("YOUR_TELEGRAM_BOT_TOKEN")
    CHAT_ID = os.environ.get("YOUR_CHAT_ID")

    if not TOKEN or not CHAT_ID:
        print("Please set the environment variables YOUR_TELEGRAM_BOT_TOKEN and YOUR_CHAT_ID")
        exit(1)

    bot = DogAlertBot(TOKEN, CHAT_ID)
    # bro, here's just an example you can connect it with our project later
    asyncio.run(bot.push_notificate(7))