import asyncio
import websockets
import json

async def test_websocket():
    session_id = "test_session"
    uri = f"ws://localhost:8000/ws/{session_id}"
    
    async with websockets.connect(uri) as websocket:
        while True:
            user_input = input("Enter message to send (or 'quit' to exit): ")
            
            message = {
                "message": user_input,
                "action": "close" if user_input.lower() == "quit" else "message",
                "session_id": session_id
            }
            
            await websocket.send(json.dumps(message))
            print(f"Sent: {message}")

            response = await websocket.recv()
            print(f"Received: {response}")
            
            if user_input.lower() == "quit":
                break

if __name__ == "__main__":
    asyncio.run(test_websocket())
