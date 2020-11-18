import asyncio
import aio_pika


def save_message(msg):
    print(msg)


async def get_one_message(loop):
    connection = await aio_pika.connect_robust(
        "amqp://testuser:testuser@127.0.0.1/", 
        loop=loop
    )
    queue_name = "logir"

    async with connection:
        # Creating channel
        channel = await connection.channel()

        # Declaring queue
        queue = await channel.declare_queue(queue_name, auto_delete=False)

        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                async with message.process():
                    save_message(message.body)

def get_message():
    loop = asyncio.get_event_loop()
    #loop.create_task(get_one_message(loop))
    #loop.run_forever()
    loop.run_until_complete(get_one_message(loop))
    loop.close()


get_message()