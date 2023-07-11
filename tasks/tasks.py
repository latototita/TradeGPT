
from tasks.views import run_tasks
from tasks.telegramfile import telbot
from apscheduler.schedulers.background import BackgroundScheduler


# Create an instance of the scheduler
scheduler = BackgroundScheduler()

# Add your task to the scheduler and schedule it to
#  run every 5 seconds asyncio.run(send_message(message))
#scheduler.add_job(run_tasks, 'interval', seconds=7200)
scheduler.add_job(telbot(), 'interval', seconds=3600)
# Start the scheduler
scheduler.start()

