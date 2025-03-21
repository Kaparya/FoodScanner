import src.bot_functions.bot_functions as bot_functions

from dotenv import load_dotenv
from functools import partial
from telegram.ext import ApplicationBuilder, CommandHandler, filters, MessageHandler, Updater

import logging
import os

def main():
    load_dotenv()
    
    bot_token = os.getenv('BOT_TOKEN')

    application = ApplicationBuilder().token(bot_token).build()
    application.add_handler(
        CommandHandler('start', bot_functions.wakeUp))
    application.add_handler(
        MessageHandler(filters.PHOTO, bot_functions.sayHi))

    print('Running...')
    application.run_polling()
    print('Finished')



if __name__ == '__main__':
    main()
