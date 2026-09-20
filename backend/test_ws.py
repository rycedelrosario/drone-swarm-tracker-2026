import asyncio
import websockets


async def main():
    async with websockets.connect("ws://127.0.0.1:8000/ws/tracks") as ws:
        print(await ws.recv())


asyncio.run(main())
